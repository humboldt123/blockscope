package com.blockscope.recording;

import com.blockscope.model.*;
import com.blockscope.upload.UploaderThread;
import com.blockscope.upload.ChunkedVideoUploader;
import com.blockscope.util.Config;
import com.blockscope.util.KeybindingsExporter;
import com.blockscope.util.PlayerAnonymizer;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.gui.screen.ingame.HandledScreen;
import net.minecraft.client.gui.screen.recipebook.RecipeBookProvider;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;
import net.minecraft.entity.ItemEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.EquipmentSlot;
import net.minecraft.item.ItemStack;
import net.minecraft.block.BlockState;
import net.minecraft.block.entity.BlockEntity;
import net.minecraft.block.entity.AbstractFurnaceBlockEntity;
import net.minecraft.block.entity.SignBlockEntity;
import net.minecraft.inventory.Inventory;
import net.minecraft.state.property.Property;
import net.minecraft.screen.ScreenHandler;
import net.minecraft.screen.slot.Slot;
import net.minecraft.screen.slot.SlotActionType;
import net.minecraft.registry.Registries;
import net.minecraft.util.Identifier;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

public class RecordingManager {
    private static RecordingManager instance;

    private final Config config;
    private final AsyncWriter writer;
    private VideoEncoder videoEncoder;
    private SegmentedTSEncoder segmentedEncoder;
    private StreamingDataWriter ticksWriter;
    private StreamingDataWriter inputsWriter;
    private StreamingDataWriter blockChangesWriter;
    private StreamingDataWriter frameMappingWriter;

    private final AtomicBoolean isRecording = new AtomicBoolean(false);
    private final AtomicLong currentTick = new AtomicLong(0);
    private long tickStartTime;

    private String sessionId;
    private Path sessionDir;
    private Path ticksFile;
    private Path inputsFile;
    private Path videoFile;
    private Path worldEventsFile;  // New: event-based world tracking
    private Path frameMappingFile; // Maps video frame index -> tick number
    private SessionMetadata metadata;


    // ReplayMod approach: Save raw chunk NBT + camera state in ticks.jsonl
    // Visualizer computes visible blocks from chunks + camera frustum

    // Chunk upload infrastructure
    private final java.util.concurrent.ConcurrentHashMap<String, Boolean> savedChunks = new java.util.concurrent.ConcurrentHashMap<>();
    private final java.util.concurrent.BlockingQueue<ChunkUploadTask> chunkUploadQueue = new java.util.concurrent.LinkedBlockingQueue<>(100);
    private Thread chunkUploaderThread;

    private static class ChunkUploadTask {
        final String dimension;
        final int chunkX;
        final int chunkZ;
        final long tick;
        final byte[] nbtData;

        ChunkUploadTask(String dimension, int chunkX, int chunkZ, long tick, byte[] nbtData) {
            this.dimension = dimension;
            this.chunkX = chunkX;
            this.chunkZ = chunkZ;
            this.tick = tick;
            this.nbtData = nbtData;
        }
    }

    private RecordingManager() {
        this.config = Config.getInstance();
        this.writer = new AsyncWriter();
    }

    public static RecordingManager getInstance() {
        if (instance == null) {
            instance = new RecordingManager();
        }
        return instance;
    }

    public void startRecording() {
        if (isRecording.get()) {
            return;
        }

        // Reset player anonymizer for new session
        PlayerAnonymizer.getInstance().resetSession();

        sessionId = "session_" + Instant.now().getEpochSecond();
        sessionDir = Paths.get(config.recordingDirectory, sessionId);
        ticksFile = sessionDir.resolve("ticks.jsonl");
        inputsFile = sessionDir.resolve("inputs.jsonl");
        videoFile = sessionDir.resolve("video.mp4");
        worldEventsFile = sessionDir.resolve("world_events.jsonl");  // Phase 4.1: Event-based tracking
        frameMappingFile = sessionDir.resolve("frame_mapping.jsonl");

        currentTick.set(0);
        tickStartTime = System.currentTimeMillis();

        // ReplayMod approach: no master maps, just raw chunk data + camera

        // Initialize session on server FIRST
        try {
            java.net.URL initUrl = new java.net.URL(config.serverUrl + "/init-session?session_id=" + sessionId);
            java.net.HttpURLConnection conn = (java.net.HttpURLConnection) initUrl.openConnection();
            conn.setRequestMethod("POST");
            int responseCode = conn.getResponseCode();
            if (responseCode == 200) {
                System.out.println("[Blockscope] Initialized session on server");
            } else {
                System.err.println("[Blockscope] Failed to initialize session: HTTP " + responseCode);
            }
            conn.disconnect();
        } catch (Exception e) {
            System.err.println("[Blockscope] Error initializing session: " + e.getMessage());
        }

        // Initialize streaming data writers for ticks, inputs, and block_changes
        ticksWriter = new StreamingDataWriter(sessionId, config.serverUrl, "ticks");
        inputsWriter = new StreamingDataWriter(sessionId, config.serverUrl, "inputs");
        blockChangesWriter = new StreamingDataWriter(sessionId, config.serverUrl, "block_changes");
        frameMappingWriter = new StreamingDataWriter(sessionId, config.serverUrl, "frame_mapping");
        System.out.println("[Blockscope] Streaming data writers initialized");

        // Initialize segmented TS encoder
        try {
            System.out.println("[Blockscope] Initializing segmented video recording...");
            segmentedEncoder = new SegmentedTSEncoder(config, sessionId, config.serverUrl, sessionDir);
            segmentedEncoder.startRecording();
            System.out.println("[Blockscope] Segmented TS encoder started successfully");
        } catch (Exception e) {
            System.err.println("[Blockscope] CRITICAL: Failed to start video recording!");
            System.err.println("[Blockscope] Error type: " + e.getClass().getName());
            System.err.println("[Blockscope] Error message: " + e.getMessage());
            e.printStackTrace();
            segmentedEncoder = null;

            // Alert user in chat
            if (config.showChatMessages) {
                MinecraftClient.getInstance().inGameHud.getChatHud().addMessage(
                    net.minecraft.text.Text.literal("§c[Blockscope] Video recording failed to start! Check logs."));
            }
        }

        // Initialize metadata
        metadata = new SessionMetadata();
        metadata.sessionId = sessionId;
        metadata.startTimestamp = Instant.now().getEpochSecond();
        metadata.minecraftVersion = "1.19.4";
        metadata.modVersion = "0.1.0-alpha";

        SessionMetadata.RecordingConfig recordingConfig = new SessionMetadata.RecordingConfig();
        recordingConfig.resolutionWidth = config.resolutionWidth;
        recordingConfig.resolutionHeight = config.resolutionHeight;
        recordingConfig.aspectRatioMode = config.aspectRatioMode;
        recordingConfig.targetFps = config.targetFps;
        recordingConfig.recordingDirectory = config.recordingDirectory;
        metadata.config = recordingConfig;

        // Ensure session directory exists FIRST
        try {
            java.nio.file.Files.createDirectories(sessionDir);
            System.out.println("[Blockscope] Created session directory: " + sessionDir);
        } catch (Exception e) {
            System.err.println("[Blockscope] Failed to create session directory: " + e.getMessage());
        }

        // Upload metadata to server (NO local writes)
        try {
            uploadMetadataToServer();
            System.out.println("[Blockscope] Uploaded metadata to server");
        } catch (Exception e) {
            System.err.println("[Blockscope] CRITICAL: Failed to upload metadata!");
            System.err.println("[Blockscope] Error: " + e.getMessage());
            e.printStackTrace();
        }

        writer.start();

        // Start chunk uploader thread
        savedChunks.clear();
        chunkUploadQueue.clear();
        chunkUploaderThread = new Thread(this::chunkUploaderLoop, "Blockscope-ChunkUploader");
        chunkUploaderThread.setDaemon(true);
        chunkUploaderThread.start();

        isRecording.set(true);

        // Capture all already-loaded chunks (chunks near player that loaded before recording started)
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player != null && client.world != null) {
            net.minecraft.client.world.ClientChunkManager chunkManager = client.world.getChunkManager();
            int renderDistance = client.options.getViewDistance().getValue();
            net.minecraft.util.math.BlockPos playerPos = client.player.getBlockPos();
            int playerChunkX = playerPos.getX() >> 4;
            int playerChunkZ = playerPos.getZ() >> 4;

            System.out.println("[Blockscope] Capturing already-loaded chunks in radius " + renderDistance);
            for (int cx = playerChunkX - renderDistance; cx <= playerChunkX + renderDistance; cx++) {
                for (int cz = playerChunkZ - renderDistance; cz <= playerChunkZ + renderDistance; cz++) {
                    net.minecraft.world.chunk.WorldChunk chunk = chunkManager.getWorldChunk(cx, cz);
                    if (chunk != null) {
                        onChunkLoad(chunk, null);
                    }
                }
            }
            System.out.println("[Blockscope] Finished capturing already-loaded chunks");
        }

        // Blocks will be captured as player explores (block changes tracked via WorldChunkMixin)

        if (config.showChatMessages) {
            MinecraftClient.getInstance().inGameHud.getChatHud().addMessage(
                net.minecraft.text.Text.literal("[Blockscope] Recording started: " + sessionId));
        }

        System.out.println("[Blockscope] Recording started: " + sessionId);
    }

    public void stopRecording() {
        if (!isRecording.get()) {
            return;
        }

        isRecording.set(false);

        // Stop streaming data writers
        if (ticksWriter != null) {
            ticksWriter.stop();
            ticksWriter = null;
        }
        if (inputsWriter != null) {
            inputsWriter.stop();
            inputsWriter = null;
        }
        if (blockChangesWriter != null) {
            blockChangesWriter.stop();
            blockChangesWriter = null;
        }
        if (frameMappingWriter != null) {
            frameMappingWriter.stop();
            frameMappingWriter = null;
        }

        // Wait for chunk uploads to complete
        if (chunkUploaderThread != null) {
            try {
                chunkUploaderThread.join(30000); // 30 second timeout
                System.out.println("[Blockscope] Chunk uploader finished");
            } catch (InterruptedException e) {
                System.err.println("[Blockscope] Chunk uploader interrupted");
            }
        }

        // Stop segmented encoder (will upload remaining segments)
        if (segmentedEncoder != null) {
            segmentedEncoder.stopRecording();
            segmentedEncoder = null;
        }

        // Write final metadata locally (update with end timestamp)
        metadata.endTimestamp = Instant.now().getEpochSecond();
        try {
            // Upload final metadata to server (NO local writes)
            uploadMetadataToServer();
        } catch (Exception e) {
            System.err.println("[Blockscope] Failed to upload final metadata: " + e.getMessage());
        }

        writer.stop();

        if (config.showChatMessages) {
            MinecraftClient.getInstance().inGameHud.getChatHud().addMessage(
                net.minecraft.text.Text.literal("[Blockscope] Recording stopped: " + sessionId));
        }

        System.out.println("[Blockscope] Recording stopped: " + sessionId);

        // Tell server to finalize video (concat segments to MP4)
        // Note: Server will auto-finalize after 15s of no segments anyway
        finalizeVideo();

        // All data (ticks, inputs, video) already streamed - no upload needed!
    }

    public void onClientTick() {
        if (!isRecording.get()) {
            return;
        }

        MinecraftClient client = MinecraftClient.getInstance();

        // Auto-stop recording if player left the world
        if (client.player == null || client.world == null) {
            System.out.println("[Blockscope] Player left world - auto-stopping recording...");
            stopRecording();
            return;
        }

        long tick = currentTick.getAndIncrement();
        tickStartTime = System.currentTimeMillis();

        TickData tickData = new TickData();
        tickData.tick = tick;
        tickData.player = capturePlayerState(client.player);
        tickData.world = captureWorldState(client.world);

        // Capture inventory every tick (better for ML - self-contained samples)
        tickData.inventory = captureInventoryState(client.player);

        tickData.gui = captureGuiState(client.currentScreen);
        tickData.crosshairTarget = captureCrosshairTarget(client);
        tickData.gamemode = client.interactionManager.getCurrentGameMode().getName();

        // Camera state ALREADY in tickData (x,y,z,pitch,yaw,fov,cameraPerspective)
        // Chunks saved as binary NBT files via onChunkLoad
        // Block changes saved to block_changes.jsonl via onBlockChanged

        // Note: Phase 4 old code (captureNearbyEntitiesState/captureNearbyBlocksState) is deprecated
        // Now using event-based master map + diffs (see ChunkBuilderMixin, WorldChunkMixin)

        // Stream tick data to server instead of writing locally
        if (ticksWriter != null) {
            ticksWriter.writeLine(tickData.toJson());
        }
    }

    // Cached reflection fields to avoid repeated lookups
    private static java.lang.reflect.Field focusedSlotField = null;
    private static java.lang.reflect.Field burnTimeField = null;
    private static java.lang.reflect.Field cookTimeField = null;
    static {
        try {
            focusedSlotField = HandledScreen.class.getDeclaredField("focusedSlot");
            focusedSlotField.setAccessible(true);
        } catch (Exception e) {
            System.err.println("[Blockscope] Failed to cache focusedSlot field: " + e.getMessage());
        }
        try {
            burnTimeField = AbstractFurnaceBlockEntity.class.getDeclaredField("burnTime");
            burnTimeField.setAccessible(true);
            cookTimeField = AbstractFurnaceBlockEntity.class.getDeclaredField("cookTime");
            cookTimeField.setAccessible(true);
        } catch (Exception e) {
            System.err.println("[Blockscope] Failed to cache furnace fields: " + e.getMessage());
        }
    }

    public void onRenderTick() {
        if (!isRecording.get()) {
            return;
        }

        MinecraftClient client = MinecraftClient.getInstance();

        // Skip frame capture if game is paused in singleplayer
        // (In multiplayer, isPaused() is always false, so this won't trigger)
        if (client.isPaused()) {
            return;
        }

        // Capture and encode frame via segmented TS encoder
        if (segmentedEncoder != null) {
            boolean captured = segmentedEncoder.captureAndEncodeFrame();
            if (captured) {
                // Record which tick this frame corresponds to (stream to server)
                int frameIndex = segmentedEncoder.getFrameCount() - 1;
                long tick = currentTick.get();
                if (frameMappingWriter != null) {
                    String frameMappingLine = "{\"frame\":" + frameIndex + ",\"tick\":" + tick + "}";
                    frameMappingWriter.writeLine(frameMappingLine);
                }
            }
        }
    }

    public void recordInputEvent(InputEvent event) {
        if (!isRecording.get()) {
            return;
        }

        event.tick = currentTick.get();
        event.offsetMs = System.currentTimeMillis() - tickStartTime;

        // Stream input event to server instead of writing locally
        if (inputsWriter != null) {
            inputsWriter.writeLine(event.toJson());
        }
    }

    public boolean isRecording() {
        return isRecording.get();
    }

    public long getCurrentTick() {
        return currentTick.get();
    }

    public int getFrameCount() {
        return videoEncoder != null ? videoEncoder.getFrameCount() : 0;
    }

    private TickData.PlayerState capturePlayerState(ClientPlayerEntity player) {
        MinecraftClient client = MinecraftClient.getInstance();
        TickData.PlayerState state = new TickData.PlayerState();

        // Position and rotation
        state.x = player.getX();
        state.y = player.getY();
        state.z = player.getZ();
        state.pitch = player.getPitch(1.0f);
        state.yaw = player.getYaw(1.0f);

        // Visual settings (can change during gameplay)
        state.fov = (float) client.options.getFov().getValue();
        state.mouseSensitivity = client.options.getMouseSensitivity().getValue();

        // Player state
        state.health = player.getHealth();
        state.hunger = player.getHungerManager().getFoodLevel();
        state.saturation = player.getHungerManager().getSaturationLevel();
        state.armor = player.getArmor();
        state.hotbarIndex = player.getInventory().selectedSlot;

        ItemStack heldStack = player.getMainHandStack();
        if (!heldStack.isEmpty()) {
            state.heldItem = createItemStack(heldStack);
        }

        // UI state flags
        state.chatOpen = (client.currentScreen != null &&
                         client.currentScreen.getClass().getSimpleName().equals("ChatScreen"));
        state.isPaused = client.isPaused(); // Only true in singleplayer when paused
        state.menuOpen = (client.currentScreen != null);

        // Detect text input (chat, books, signs, anvils, creative search, command blocks, etc.)
        state.textInputActive = false;
        if (client.currentScreen != null) {
            String screenName = client.currentScreen.getClass().getSimpleName();
            state.textInputActive = screenName.equals("ChatScreen") ||
                                   screenName.equals("BookEditScreen") ||
                                   screenName.equals("BookScreen") ||
                                   screenName.equals("SignEditScreen") ||
                                   screenName.equals("AbstractSignEditScreen") ||
                                   screenName.equals("AnvilScreen") ||
                                   screenName.equals("CommandBlockScreen") ||
                                   screenName.equals("StructureBlockScreen") ||
                                   screenName.equals("JigsawBlockScreen") ||
                                   // Creative inventory has search field
                                   (screenName.equals("CreativeInventoryScreen") &&
                                    client.currentScreen.getFocused() != null);
        }

        state.hideGui = client.options.hudHidden; // F1 pressed
        state.debugScreenVisible = client.options.debugEnabled; // F3 pressed

        // Camera settings (F5 perspective)
        switch (client.options.getPerspective()) {
            case FIRST_PERSON:
                state.cameraPerspective = "first_person";
                break;
            case THIRD_PERSON_BACK:
                state.cameraPerspective = "third_person_back";
                break;
            case THIRD_PERSON_FRONT:
                state.cameraPerspective = "third_person_front";
                break;
            default:
                state.cameraPerspective = "first_person";
        }

        // Active status effects (potions)
        state.statusEffects = captureStatusEffects(player);

        // Accessibility/gameplay settings
        state.showSubtitles = client.options.getShowSubtitles().getValue();
        state.autoJumpEnabled = client.options.getAutoJump().getValue();

        return state;
    }

    private TickData.StatusEffect[] captureStatusEffects(ClientPlayerEntity player) {
        java.util.Collection<net.minecraft.entity.effect.StatusEffectInstance> effects =
            player.getStatusEffects();

        if (effects.isEmpty()) {
            return null; // Omit field if no effects
        }

        java.util.List<TickData.StatusEffect> effectList = new java.util.ArrayList<>();

        for (net.minecraft.entity.effect.StatusEffectInstance effect : effects) {
            TickData.StatusEffect statusEffect = new TickData.StatusEffect();

            // Get effect ID (e.g. "minecraft:speed")
            Identifier effectId = Registries.STATUS_EFFECT.getId(effect.getEffectType());
            statusEffect.effectId = effectId != null ? effectId.toString() : "unknown";

            // Effect properties
            statusEffect.amplifier = effect.getAmplifier(); // 0 = level I, 1 = level II, etc.
            statusEffect.duration = effect.getDuration(); // Remaining ticks
            statusEffect.ambient = effect.isAmbient(); // From beacon (fewer particles)
            statusEffect.showParticles = effect.shouldShowParticles();
            statusEffect.showIcon = effect.shouldShowIcon();

            effectList.add(statusEffect);
        }

        return effectList.toArray(new TickData.StatusEffect[0]);
    }

    private TickData.WorldState captureWorldState(World world) {
        TickData.WorldState state = new TickData.WorldState();
        state.worldTime = world.getTime();
        state.timeOfDay = world.getTimeOfDay();

        Identifier dimensionId = world.getRegistryKey().getValue();
        state.dimension = dimensionId.toString();

        ClientPlayerEntity player = MinecraftClient.getInstance().player;
        if (player != null) {
            BlockPos pos = player.getBlockPos();
            Identifier biomeId = world.getBiome(pos).getKey().map(key -> key.getValue()).orElse(null);
            state.biome = biomeId != null ? biomeId.toString() : "unknown";
        }

        if (world.isRaining()) {
            state.weather = world.isThundering() ? "thunder" : "rain";
        } else {
            state.weather = "clear";
        }

        return state;
    }

    private TickData.InventoryState captureInventoryState(ClientPlayerEntity player) {
        TickData.InventoryState state = new TickData.InventoryState();
        java.util.List<TickData.SlotItem> slotsList = new java.util.ArrayList<>();

        // Minecraft slot layout:
        // Slots 0-8: Hotbar
        // Slots 9-35: Main inventory (27 slots)
        // Slots 36-39: Armor (boots, leggings, chestplate, helmet)
        // Slot 40: Offhand

        // Capture main inventory + hotbar (slots 0-35)
        for (int i = 0; i < 36; i++) {
            ItemStack stack = player.getInventory().getStack(i);
            if (!stack.isEmpty()) {
                TickData.SlotItem slotItem = new TickData.SlotItem();
                slotItem.slot = i;
                slotItem.item = createItemStack(stack);
                slotsList.add(slotItem);
            }
        }

        // Capture armor (slots 36-39)
        for (int i = 0; i < 4; i++) {
            ItemStack stack = player.getInventory().armor.get(i);
            if (!stack.isEmpty()) {
                TickData.SlotItem slotItem = new TickData.SlotItem();
                slotItem.slot = 36 + i;
                slotItem.item = createItemStack(stack);
                slotsList.add(slotItem);
            }
        }

        // Capture offhand (slot 40)
        ItemStack offhandStack = player.getOffHandStack();
        if (!offhandStack.isEmpty()) {
            TickData.SlotItem slotItem = new TickData.SlotItem();
            slotItem.slot = 40;
            slotItem.item = createItemStack(offhandStack);
            slotsList.add(slotItem);
        }

        state.slots = slotsList.toArray(new TickData.SlotItem[0]);

        // Capture cursor stack (item held by mouse)
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.currentScreen instanceof HandledScreen) {
            HandledScreen<?> handledScreen = (HandledScreen<?>) client.currentScreen;
            ItemStack cursorStack = handledScreen.getScreenHandler().getCursorStack();
            if (!cursorStack.isEmpty()) {
                state.cursorStack = createItemStack(cursorStack);
            }
        }

        return state;
    }

    private TickData.GuiState captureGuiState(Screen currentScreen) {
        TickData.GuiState state = new TickData.GuiState();

        if (currentScreen == null) {
            state.screenType = null;
            return state;
        }

        state.screenType = currentScreen.getClass().getSimpleName();

        MinecraftClient client = MinecraftClient.getInstance();
        double mouseX = client.mouse.getX() * client.getWindow().getScaledWidth() / client.getWindow().getWidth();
        double mouseY = client.mouse.getY() * client.getWindow().getScaledHeight() / client.getWindow().getHeight();

        state.cursorX = (int) mouseX;
        state.cursorY = (int) mouseY;

        // Enhanced GUI state for HandledScreen (containers, crafting, etc.)
        if (currentScreen instanceof HandledScreen) {
            HandledScreen<?> handledScreen = (HandledScreen<?>) currentScreen;
            ScreenHandler handler = handledScreen.getScreenHandler();

            // Get hovered slot using cached reflection field
            if (focusedSlotField != null) {
                try {
                    Slot focusedSlot = (Slot) focusedSlotField.get(handledScreen);
                    if (focusedSlot != null) {
                        state.hoveredSlotIndex = focusedSlot.id;
                    }
                } catch (Exception e) {
                    // Failed to get focused slot, skip it
                }
            }

            // Capture container contents (slots that don't belong to player inventory)
            captureContainerContents(state, handler, client.player);

            // Capture crafting grid if present
            captureCraftingState(state, handler);

            // Check recipe book state
            if (currentScreen instanceof RecipeBookProvider) {
                RecipeBookProvider recipeScreen = (RecipeBookProvider) currentScreen;
                state.recipeBookOpen = recipeScreen.getRecipeBookWidget().isOpen();
            }
        }

        return state;
    }

    private TickData.CrosshairTarget captureCrosshairTarget(MinecraftClient client) {
        TickData.CrosshairTarget target = new TickData.CrosshairTarget();

        HitResult hitResult = client.crosshairTarget;
        if (hitResult == null) {
            target.type = "none";
            return target;
        }

        if (hitResult.getType() == HitResult.Type.BLOCK) {
            BlockHitResult blockHit = (BlockHitResult) hitResult;
            BlockPos pos = blockHit.getBlockPos();

            target.type = "block";
            target.blockPos = new TickData.BlockPos(pos.getX(), pos.getY(), pos.getZ());

            Identifier blockId = Registries.BLOCK.getId(client.world.getBlockState(pos).getBlock());
            target.blockId = blockId.toString();

        } else if (hitResult.getType() == HitResult.Type.ENTITY) {
            EntityHitResult entityHit = (EntityHitResult) hitResult;
            Entity entity = entityHit.getEntity();

            target.type = "entity";
            target.entityUuid = PlayerAnonymizer.getInstance().anonymizeUuid(entity.getUuidAsString());
            target.entityType = entity.getType().toString();

        } else {
            target.type = "none";
        }

        return target;
    }

    private TickData.ItemStack createItemStack(ItemStack stack) {
        Identifier itemId = Registries.ITEM.getId(stack.getItem());
        String id = itemId.toString();
        int count = stack.getCount();

        TickData.ItemStack item = new TickData.ItemStack(id, count);

        // Durability (damage and percentage)
        if (stack.isDamageable()) {
            item.damage = stack.getDamage();
            item.maxDamage = stack.getMaxDamage();

            // Calculate durability as 0-1 (1.0 = pristine, 0.0 = broken)
            // durability = (maxDamage - damage) / maxDamage
            if (item.maxDamage > 0) {
                item.durability = (float) (item.maxDamage - item.damage) / item.maxDamage;
            }
        }

        // Enchantments
        if (stack.hasEnchantments()) {
            net.minecraft.nbt.NbtList enchantList = stack.getEnchantments();
            java.util.List<TickData.Enchantment> enchantments = new java.util.ArrayList<>();

            for (int i = 0; i < enchantList.size(); i++) {
                net.minecraft.nbt.NbtCompound enchant = enchantList.getCompound(i);
                String enchantId = enchant.getString("id");
                int level = enchant.getInt("lvl");

                TickData.Enchantment e = new TickData.Enchantment();
                e.id = enchantId;
                e.level = level;
                enchantments.add(e);
            }

            item.enchantments = enchantments.toArray(new TickData.Enchantment[0]);
        }

        return item;
    }

    // Note: Inventory comparison methods removed - we now capture inventory every tick (no delta encoding)

    private void captureContainerContents(TickData.GuiState state, ScreenHandler handler, ClientPlayerEntity player) {
        // Count non-player slots (container slots start at beginning, player inventory at end)
        int playerInvStart = handler.slots.size() - 36; // Last 36 slots are usually player inventory
        if (playerInvStart <= 0) {
            return; // No container slots
        }

        // Detect container type from handler class name
        String handlerName = handler.getClass().getSimpleName();
        String containerType = handlerName.replace("ScreenHandler", "").toLowerCase();
        containerType = "minecraft:" + containerType;

        // Capture container data
        TickData.ContainerData container = new TickData.ContainerData();
        container.type = containerType;
        container.size = playerInvStart;

        // Capture non-empty container slots
        java.util.List<TickData.SlotItem> containerSlots = new java.util.ArrayList<>();
        for (int i = 0; i < playerInvStart; i++) {
            Slot slot = handler.slots.get(i);
            ItemStack stack = slot.getStack();
            if (!stack.isEmpty()) {
                TickData.SlotItem slotItem = new TickData.SlotItem();
                slotItem.slot = i;
                slotItem.item = createItemStack(stack);
                containerSlots.add(slotItem);
            }
        }

        container.slots = containerSlots.toArray(new TickData.SlotItem[0]);
        state.container = container;
    }

    private void captureCraftingState(TickData.GuiState state, ScreenHandler handler) {
        String handlerName = handler.getClass().getSimpleName();

        // Detect crafting screens (CraftingTableScreenHandler, PlayerScreenHandler)
        if (handlerName.contains("Crafting") || handlerName.contains("Player")) {
            TickData.CraftingData crafting = new TickData.CraftingData();

            // Crafting table: 3x3 grid (slots 1-9) + result (slot 0)
            // Player inventory: 2x2 grid (slots 1-4) + result (slot 0)
            boolean isCraftingTable = handlerName.contains("CraftingTable");
            crafting.largeCraftingGrid = isCraftingTable;

            int gridSize = isCraftingTable ? 9 : 4;
            java.util.List<TickData.SlotItem> gridSlots = new java.util.ArrayList<>();

            // Capture non-empty crafting grid slots
            for (int i = 1; i <= gridSize; i++) {
                if (i < handler.slots.size()) {
                    ItemStack stack = handler.slots.get(i).getStack();
                    if (!stack.isEmpty()) {
                        TickData.SlotItem slotItem = new TickData.SlotItem();
                        slotItem.slot = i - 1; // 0-indexed for crafting grid
                        slotItem.item = createItemStack(stack);
                        gridSlots.add(slotItem);
                    }
                }
            }

            crafting.craftingGrid = gridSlots.toArray(new TickData.SlotItem[0]);

            // Capture crafting result (slot 0)
            if (!handler.slots.isEmpty()) {
                ItemStack resultStack = handler.slots.get(0).getStack();
                if (!resultStack.isEmpty()) {
                    crafting.result = createItemStack(resultStack);
                }
            }

            state.crafting = crafting;
        }
    }

    // Phase 4: Block State Serialization (CRITICAL for rotation, facing, etc.)
    private String serializeBlockStateProperties(BlockState blockState) {
        // Convert block state properties to string format: "facing=north,half=bottom,powered=true"
        java.util.Map<Property<?>, Comparable<?>> properties = blockState.getEntries();

        if (properties.isEmpty()) {
            return null;
        }

        java.util.List<String> propertyStrings = new java.util.ArrayList<>();
        for (java.util.Map.Entry<Property<?>, Comparable<?>> entry : properties.entrySet()) {
            String propertyName = entry.getKey().getName();
            String propertyValue = entry.getValue().toString();
            propertyStrings.add(propertyName + "=" + propertyValue);
        }

        // Sort for consistency
        java.util.Collections.sort(propertyStrings);

        return String.join(",", propertyStrings);
    }

    // Phase 4: Capture nearby blocks with state properties
    private TickData.NearbyBlocksState captureNearbyBlocksState(ClientPlayerEntity player, World world) {
        if (!config.captureNearbyBlocks) {
            return null;
        }

        TickData.NearbyBlocksState state = new TickData.NearbyBlocksState();
        java.util.List<TickData.BlockData> blockList = new java.util.ArrayList<>();

        // Get player position for radius check
        BlockPos playerPos = player.getBlockPos();
        int radius = config.blockCaptureRadius;

        // Iterate blocks in cubic region around player
        for (int x = -radius; x <= radius; x++) {
            for (int y = -radius; y <= radius; y++) {
                for (int z = -radius; z <= radius; z++) {
                    BlockPos blockPos = playerPos.add(x, y, z);

                    // Get block state
                    BlockState blockState = world.getBlockState(blockPos);

                    // Skip air blocks
                    if (blockState.isAir()) {
                        continue;
                    }

                    TickData.BlockData blockData = new TickData.BlockData();

                    // Position (absolute or relative based on config)
                    if (config.useRelativeBlockPositions) {
                        blockData.x = x;
                        blockData.y = y;
                        blockData.z = z;
                    } else {
                        blockData.x = blockPos.getX();
                        blockData.y = blockPos.getY();
                        blockData.z = blockPos.getZ();
                    }

                    // Block ID
                    Identifier blockId = Registries.BLOCK.getId(blockState.getBlock());
                    blockData.blockId = blockId.toString();

                    // CRITICAL: Block state properties (rotation, facing, etc.)
                    blockData.blockStateProperties = serializeBlockStateProperties(blockState);

                    // Block entity data (chests, furnaces, signs, etc.)
                    if (config.captureBlockEntities) {
                        BlockEntity blockEntity = world.getBlockEntity(blockPos);
                        if (blockEntity != null) {
                            blockData.blockEntity = captureBlockEntityData(blockEntity);
                        }
                    }

                    blockList.add(blockData);
                }
            }
        }

        state.blocks = blockList.toArray(new TickData.BlockData[0]);
        return state;
    }

    // Phase 4: Capture block entity data (chests, furnaces, signs)
    private TickData.BlockEntityData captureBlockEntityData(BlockEntity blockEntity) {
        TickData.BlockEntityData data = new TickData.BlockEntityData();

        Identifier typeId = Registries.BLOCK_ENTITY_TYPE.getId(blockEntity.getType());
        data.type = typeId != null ? typeId.toString() : "unknown";

        // Container (chest, barrel, shulker box, etc.)
        if (blockEntity instanceof Inventory) {
            Inventory inventory = (Inventory) blockEntity;
            int size = inventory.size();
            data.items = new TickData.ItemStack[size];

            for (int i = 0; i < size; i++) {
                ItemStack stack = inventory.getStack(i);
                if (!stack.isEmpty()) {
                    data.items[i] = createItemStack(stack);
                }
            }
        }

        // Furnace specific (use cached reflection fields)
        if (blockEntity instanceof AbstractFurnaceBlockEntity) {
            AbstractFurnaceBlockEntity furnace = (AbstractFurnaceBlockEntity) blockEntity;
            if (burnTimeField != null && cookTimeField != null) {
                try {
                    data.burnTime = (Integer) burnTimeField.get(furnace);
                    data.cookTime = (Integer) cookTimeField.get(furnace);
                } catch (Exception e) {
                    // Failed to get furnace data, skip it
                }
            }
        }

        // Sign specific (sign API changed in 1.20+, using reflection for 1.19.4 compatibility)
        if (blockEntity instanceof SignBlockEntity) {
            SignBlockEntity sign = (SignBlockEntity) blockEntity;
            data.signText = new String[4];
            try {
                // Try to get messages using reflection (field names may vary)
                for (int i = 0; i < 4; i++) {
                    // In 1.19.4, signs still use getTextOnRow method
                    try {
                        java.lang.reflect.Method getTextOnRow = SignBlockEntity.class.getMethod("getTextOnRow", int.class);
                        net.minecraft.text.Text text = (net.minecraft.text.Text) getTextOnRow.invoke(sign, i);
                        data.signText[i] = text != null ? text.getString() : "";
                    } catch (NoSuchMethodException e) {
                        // If method doesn't exist, try accessing text field directly
                        data.signText[i] = "";
                    }
                }
            } catch (Exception e) {
                // Failed to get sign text, skip it
            }
        }

        return data;
    }

    // Phase 4: Capture nearby entities
    private TickData.NearbyEntitiesState captureNearbyEntitiesState(ClientPlayerEntity player, World world) {
        if (!config.captureNearbyEntities) {
            return null;
        }

        TickData.NearbyEntitiesState state = new TickData.NearbyEntitiesState();
        java.util.List<TickData.EntityData> entityList = new java.util.ArrayList<>();

        // Get player position for radius check
        double playerX = player.getX();
        double playerY = player.getY();
        double playerZ = player.getZ();
        double radius = config.entityCaptureRadius;
        double radiusSq = radius * radius;

        // Create bounding box around player for entity lookup
        net.minecraft.util.math.Box searchBox = new net.minecraft.util.math.Box(
            playerX - radius, playerY - radius, playerZ - radius,
            playerX + radius, playerY + radius, playerZ + radius
        );

        // Get entities within bounding box
        java.util.List<Entity> entities = world.getEntitiesByClass(Entity.class, searchBox, entity -> true);

        // Iterate entities
        for (Entity entity : entities) {
            // Skip the player themselves
            if (entity.equals(player)) {
                continue;
            }

            // Distance check (squared to avoid sqrt)
            double dx = entity.getX() - playerX;
            double dy = entity.getY() - playerY;
            double dz = entity.getZ() - playerZ;
            double distSq = dx * dx + dy * dy + dz * dz;

            if (distSq > radiusSq) {
                continue;
            }

            // Filter by entity type based on config
            if (!config.captureAllEntities && !(entity instanceof LivingEntity)) {
                if (!config.captureItemEntities || !(entity instanceof ItemEntity)) {
                    continue;
                }
            }

            // Create entity data
            TickData.EntityData entityData = new TickData.EntityData();
            entityData.uuid = PlayerAnonymizer.getInstance().anonymizeUuid(entity.getUuidAsString());
            entityData.type = Registries.ENTITY_TYPE.getId(entity.getType()).toString();
            entityData.x = entity.getX();
            entityData.y = entity.getY();
            entityData.z = entity.getZ();
            entityData.pitch = entity.getPitch(1.0f);
            entityData.yaw = entity.getYaw(1.0f);

            // Common entity states
            entityData.onFire = entity.isOnFire();
            entityData.invisible = entity.isInvisible();
            if (entity.hasCustomName()) {
                // Only anonymize if it's a player entity
                String customName = entity.getCustomName().getString();
                if (entity instanceof net.minecraft.entity.player.PlayerEntity) {
                    customName = PlayerAnonymizer.getInstance().anonymizeName(customName);
                }
                entityData.customName = customName;
            }

            // Living entity specific data
            if (entity instanceof LivingEntity) {
                LivingEntity living = (LivingEntity) entity;
                entityData.health = living.getHealth();
                entityData.maxHealth = living.getMaxHealth();
                entityData.equipment = captureEquipment(living);
            }

            // Item entity specific data
            if (entity instanceof ItemEntity) {
                ItemEntity itemEntity = (ItemEntity) entity;
                entityData.item = createItemStack(itemEntity.getStack());
                entityData.age = itemEntity.getItemAge();
            }

            entityList.add(entityData);
        }

        state.entities = entityList.toArray(new TickData.EntityData[0]);
        return state;
    }

    // Phase 4: Capture equipment from living entities
    private TickData.EquipmentData captureEquipment(LivingEntity entity) {
        TickData.EquipmentData equipment = new TickData.EquipmentData();

        // Main hand
        ItemStack mainHand = entity.getMainHandStack();
        if (!mainHand.isEmpty()) {
            equipment.mainHand = createItemStack(mainHand);
        }

        // Offhand
        ItemStack offhand = entity.getOffHandStack();
        if (!offhand.isEmpty()) {
            equipment.offhand = createItemStack(offhand);
        }

        // Armor slots
        for (EquipmentSlot slot : EquipmentSlot.values()) {
            if (slot.getType() == EquipmentSlot.Type.ARMOR) {
                ItemStack armorStack = entity.getEquippedStack(slot);
                if (!armorStack.isEmpty()) {
                    switch (slot) {
                        case HEAD:
                            equipment.helmet = createItemStack(armorStack);
                            break;
                        case CHEST:
                            equipment.chestplate = createItemStack(armorStack);
                            break;
                        case LEGS:
                            equipment.leggings = createItemStack(armorStack);
                            break;
                        case FEET:
                            equipment.boots = createItemStack(armorStack);
                            break;
                    }
                }
            }
        }

        return equipment;
    }

    // Phase 4: Delta encoding comparison for blocks
    private boolean blocksEquals(TickData.NearbyBlocksState a, TickData.NearbyBlocksState b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        if (a.blocks.length != b.blocks.length) return false;

        // Create position-based map for efficient comparison
        java.util.Map<String, TickData.BlockData> mapA = new java.util.HashMap<>();
        for (TickData.BlockData block : a.blocks) {
            String key = block.x + "," + block.y + "," + block.z;
            mapA.put(key, block);
        }

        for (TickData.BlockData blockB : b.blocks) {
            String key = blockB.x + "," + blockB.y + "," + blockB.z;
            TickData.BlockData blockA = mapA.get(key);
            if (blockA == null || !blockDataEquals(blockA, blockB)) {
                return false;
            }
        }

        return true;
    }

    private boolean blockDataEquals(TickData.BlockData a, TickData.BlockData b) {
        // Compare position
        if (a.x != b.x || a.y != b.y || a.z != b.z) {
            return false;
        }

        // Compare block ID
        if (!a.blockId.equals(b.blockId)) {
            return false;
        }

        // Compare block state properties (CRITICAL for detecting rotations, etc.)
        if (a.blockStateProperties == null ? b.blockStateProperties != null :
                !a.blockStateProperties.equals(b.blockStateProperties)) {
            return false;
        }

        // Block entity comparison would go here...
        // For MVP, block ID and state properties are most important

        return true;
    }

    // Phase 4: Delta encoding comparison for entities
    private boolean entitiesEquals(TickData.NearbyEntitiesState a, TickData.NearbyEntitiesState b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        if (a.entities.length != b.entities.length) return false;

        // Create maps by UUID for efficient comparison
        java.util.Map<String, TickData.EntityData> mapA = new java.util.HashMap<>();
        for (TickData.EntityData entity : a.entities) {
            mapA.put(entity.uuid, entity);
        }

        for (TickData.EntityData entityB : b.entities) {
            TickData.EntityData entityA = mapA.get(entityB.uuid);
            if (entityA == null || !entityDataEquals(entityA, entityB)) {
                return false;
            }
        }

        return true;
    }

    private boolean entityDataEquals(TickData.EntityData a, TickData.EntityData b) {
        // Compare position (with small epsilon for floating point)
        double epsilon = 0.001;
        if (Math.abs(a.x - b.x) > epsilon ||
                Math.abs(a.y - b.y) > epsilon ||
                Math.abs(a.z - b.z) > epsilon) {
            return false;
        }

        // Compare rotation
        if (Math.abs(a.pitch - b.pitch) > epsilon ||
                Math.abs(a.yaw - b.yaw) > epsilon) {
            return false;
        }

        // Compare type and UUID
        if (!a.type.equals(b.type) || !a.uuid.equals(b.uuid)) {
            return false;
        }

        // Compare health (if applicable)
        if (a.health != null && b.health != null) {
            if (Math.abs(a.health - b.health) > epsilon) {
                return false;
            }
        }

        // For MVP, position/rotation/health changes are most important

        return true;
    }

    // ========================================
    // Phase 4.1: Event-based World Tracking
    // ========================================

    // Reusable Gson instance to avoid creating new ones every flush
    private static final com.google.gson.Gson gson = new com.google.gson.Gson();

    /**
     * Record camera state each tick for visualizer to compute visibility.
     * Replaces old block iteration - now visualizer computes what's visible from chunks + camera.
     */

    /**
     * Called when a block changes (player breaks/places, piston moves, etc.).
     * Records block updates for ML to learn player actions.
     */
    public void onBlockChanged(World world, BlockPos pos, BlockState oldState, BlockState newState) {
        if (!isRecording.get()) {
            return;
        }

        try {
            Identifier dimensionId = world.getRegistryKey().getValue();
            String dimension = dimensionId.toString();

            // Create simple block change event (JSONL format)
            com.google.gson.JsonObject event = new com.google.gson.JsonObject();
            event.addProperty("tick", (int) currentTick.get());
            event.addProperty("dimension", dimension);
            event.addProperty("x", pos.getX());
            event.addProperty("y", pos.getY());
            event.addProperty("z", pos.getZ());
            event.addProperty("blockId", Registries.BLOCK.getId(newState.getBlock()).toString());

            String props = serializeBlockStateProperties(newState);
            if (props != null) {
                event.addProperty("blockStateProperties", props);
            }

            if (blockChangesWriter != null) {
                blockChangesWriter.writeLine(event.toString());
            }
        } catch (Exception e) {
            System.err.println("[Blockscope] Error in onBlockChanged: " + e.getMessage());
        }
    }

    /**
     * Background thread that uploads chunks to server
     */
    private void chunkUploaderLoop() {
        while (isRecording.get() || !chunkUploadQueue.isEmpty()) {
            try {
                ChunkUploadTask task = chunkUploadQueue.poll(1, java.util.concurrent.TimeUnit.SECONDS);
                if (task != null) {
                    uploadChunk(task);
                }
            } catch (InterruptedException e) {
                break;
            }
        }
        System.out.println("[Blockscope] Chunk uploader thread exiting");
    }

    /**
     * Upload a chunk to the server
     */
    private void uploadChunk(ChunkUploadTask task) {
        try {
            String url = config.serverUrl + "/upload-chunk?session_id=" + sessionId +
                         "&chunkX=" + task.chunkX + "&chunkZ=" + task.chunkZ +
                         "&tick=" + task.tick + "&dimension=" + java.net.URLEncoder.encode(task.dimension, "UTF-8");

            java.net.HttpURLConnection conn = (java.net.HttpURLConnection) new java.net.URL(url).openConnection();
            conn.setDoOutput(true);
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/octet-stream");
            conn.setRequestProperty("Content-Length", String.valueOf(task.nbtData.length));

            try (java.io.OutputStream out = conn.getOutputStream()) {
                out.write(task.nbtData);
            }

            int responseCode = conn.getResponseCode();
            if (responseCode == 200) {
                System.out.println("[Blockscope] Uploaded chunk (" + task.chunkX + "," + task.chunkZ + ")");
            } else {
                System.err.println("[Blockscope] Chunk upload failed: HTTP " + responseCode);
            }

            conn.disconnect();
        } catch (Exception e) {
            System.err.println("[Blockscope] Error uploading chunk: " + e.getMessage());
        }
    }

    /**
     * Write a world event to world_events.jsonl
     */
    private void uploadMetadataToServer() {
        try {
            java.net.URL url = new java.net.URL(config.serverUrl + "/upload-metadata?session_id=" + sessionId);
            java.net.HttpURLConnection conn = (java.net.HttpURLConnection) url.openConnection();

            conn.setDoOutput(true);
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");

            byte[] metadataBytes = metadata.toJson().getBytes();

            try (java.io.OutputStream out = conn.getOutputStream()) {
                out.write(metadataBytes);
            }

            int responseCode = conn.getResponseCode();
            if (responseCode == 200) {
                System.out.println("[Blockscope] Uploaded metadata.json to server");
            } else {
                System.err.println("[Blockscope] Failed to upload metadata: HTTP " + responseCode);
            }

            conn.disconnect();
        } catch (Exception e) {
            System.err.println("[Blockscope] Error uploading metadata to server: " + e.getMessage());
        }
    }




    /**
     * Called when a chunk loads from server packet (from ClientChunkManagerMixin).
     * Also called on already-loaded chunks when recording starts.
     * Saves chunk block data as compressed NBT (nbt parameter unused, we serialize from chunk).
     */
    public void onChunkLoad(net.minecraft.world.chunk.WorldChunk chunk, net.minecraft.nbt.NbtCompound nbt) {
        if (!isRecording.get() || chunk == null) {
            return;
        }

        try {
            long chunkArrivalTick = currentTick.get();
            net.minecraft.world.World world = chunk.getWorld();
            if (world == null) {
                return;
            }

            net.minecraft.util.Identifier dimensionId = world.getRegistryKey().getValue();
            String dimension = dimensionId.toString();
            net.minecraft.util.math.ChunkPos chunkPos = chunk.getPos();

            // Check if we've already saved this chunk (only save once)
            String chunkKey = dimension + ":" + chunkPos.x + ":" + chunkPos.z;
            if (savedChunks.containsKey(chunkKey)) {
                return; // Already saved
            }

            // Create NBT compound with chunk sections
            net.minecraft.nbt.NbtCompound chunkNbt = new net.minecraft.nbt.NbtCompound();
            net.minecraft.nbt.NbtList sectionsNbt = new net.minecraft.nbt.NbtList();

            // Get chunk sections (1.19.4)
            net.minecraft.world.chunk.ChunkSection[] sections = chunk.getSectionArray();
            int bottomY = world.getBottomY();

            for (int i = 0; i < sections.length; i++) {
                net.minecraft.world.chunk.ChunkSection section = sections[i];
                if (section == null || section.isEmpty()) {
                    continue;
                }

                net.minecraft.nbt.NbtCompound sectionNbt = new net.minecraft.nbt.NbtCompound();
                sectionNbt.putByte("Y", (byte) (bottomY / 16 + i));

                // Save block states
                net.minecraft.nbt.NbtCompound blockStatesNbt = new net.minecraft.nbt.NbtCompound();
                net.minecraft.nbt.NbtList paletteNbt = new net.minecraft.nbt.NbtList();

                // Build palette of unique block states in this section
                java.util.Map<net.minecraft.block.BlockState, Integer> palette = new java.util.HashMap<>();
                java.util.List<net.minecraft.block.BlockState> paletteList = new java.util.ArrayList<>();

                for (int y = 0; y < 16; y++) {
                    for (int z = 0; z < 16; z++) {
                        for (int x = 0; x < 16; x++) {
                            net.minecraft.block.BlockState state = section.getBlockState(x, y, z);
                            if (!palette.containsKey(state)) {
                                palette.put(state, paletteList.size());
                                paletteList.add(state);
                            }
                        }
                    }
                }

                // Write palette
                for (net.minecraft.block.BlockState state : paletteList) {
                    net.minecraft.nbt.NbtCompound stateNbt = new net.minecraft.nbt.NbtCompound();
                    stateNbt.putString("Name", net.minecraft.registry.Registries.BLOCK.getId(state.getBlock()).toString());

                    // Add properties if present
                    if (!state.getEntries().isEmpty()) {
                        net.minecraft.nbt.NbtCompound propsNbt = new net.minecraft.nbt.NbtCompound();
                        for (java.util.Map.Entry<net.minecraft.state.property.Property<?>, Comparable<?>> entry : state.getEntries().entrySet()) {
                            propsNbt.putString(entry.getKey().getName(), entry.getValue().toString());
                        }
                        stateNbt.put("Properties", propsNbt);
                    }

                    paletteNbt.add(stateNbt);
                }

                blockStatesNbt.put("palette", paletteNbt);

                // Pack block indices into data array (standard Minecraft NBT format)
                if (paletteList.size() > 1) {
                    // Calculate bits needed per block
                    int bitsPerBlock = Math.max(4, 32 - Integer.numberOfLeadingZeros(paletteList.size() - 1));
                    int blocksPerLong = 64 / bitsPerBlock;

                    // Pack block indices into long array (Y-Z-X order like Minecraft)
                    long[] data = new long[(4096 + blocksPerLong - 1) / blocksPerLong];
                    int index = 0;
                    for (int y = 0; y < 16; y++) {
                        for (int z = 0; z < 16; z++) {
                            for (int x = 0; x < 16; x++) {
                                net.minecraft.block.BlockState state = section.getBlockState(x, y, z);
                                int paletteIndex = palette.get(state);
                                int longIndex = index / blocksPerLong;
                                int bitIndex = (index % blocksPerLong) * bitsPerBlock;
                                data[longIndex] |= (long)paletteIndex << bitIndex;
                                index++;
                            }
                        }
                    }
                    blockStatesNbt.putLongArray("data", data);
                }
                // If palette size == 1, no data array needed (entire section is one block)

                sectionNbt.put("block_states", blockStatesNbt);
                sectionsNbt.add(sectionNbt);
            }

            chunkNbt.put("sections", sectionsNbt);

            // Serialize to compressed binary
            java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
            try {
                net.minecraft.nbt.NbtIo.writeCompressed(chunkNbt, baos);
                byte[] nbtBytes = baos.toByteArray();

                // Queue for async upload
                ChunkUploadTask task = new ChunkUploadTask(dimension, chunkPos.x, chunkPos.z, chunkArrivalTick, nbtBytes);
                if (!chunkUploadQueue.offer(task)) {
                    System.err.println("[Blockscope] Chunk upload queue full! Dropped chunk " + chunkPos);
                } else {
                    savedChunks.put(chunkKey, true);
                    System.out.println("[Blockscope] Queued chunk " + chunkPos + " (" + nbtBytes.length + " bytes)");
                }
            } catch (java.io.IOException e) {
                System.err.println("[Blockscope] Failed to serialize chunk NBT: " + e.getMessage());
            }
        } catch (Exception e) {
            System.err.println("[Blockscope] Error in onChunkLoad: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Tell server to concatenate TS segments into final MP4
     */
    private void finalizeVideo() {
        try {
            java.net.URL url = new java.net.URL(config.serverUrl + "/finalize-video?session_id=" + sessionId);
            java.net.HttpURLConnection conn = (java.net.HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("User-Agent", "Blockscope/1.0");

            int responseCode = conn.getResponseCode();
            if (responseCode == 200) {
                System.out.println("[Blockscope] Video finalized successfully");
            } else {
                System.err.println("[Blockscope] Video finalization failed: HTTP " + responseCode);
            }

            conn.disconnect();
        } catch (Exception e) {
            System.err.println("[Blockscope] Error finalizing video: " + e.getMessage());
        }
    }
}
