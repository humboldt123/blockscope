package com.blockscope.recording;

import com.blockscope.model.*;
import com.blockscope.util.Config;
import com.blockscope.util.KeybindingsExporter;
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
import net.minecraft.util.Identifier;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.registry.Registry;
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

    // Inventory change detection
    private TickData.InventoryState previousInventory;

    // Phase 4: Entity/block change detection (delta encoding)
    private TickData.NearbyEntitiesState previousNearbyEntities;
    private TickData.NearbyBlocksState previousNearbyBlocks;

    // Phase 4.1: Master map tracking (event-based architecture)
    // Master map: "dimension:x:y:z" -> block data (accumulates all seen blocks)
    private java.util.HashMap<String, WorldEvent.BlockData> masterBlockMap = new java.util.HashMap<>();
    // Entity map: UUID -> entity data (tracks all seen entities)
    private java.util.HashMap<String, WorldEvent.EntityData> masterEntityMap = new java.util.HashMap<>();
    // Last snapshot tick (for periodic full dumps every 1200 ticks)
    private long lastSnapshotTick = 0;

    // Batching for block_seen events (to prevent queue overflow)
    private final java.util.concurrent.ConcurrentHashMap<String, WorldEvent.BlockData> blockBatchBuffer = new java.util.concurrent.ConcurrentHashMap<>();
    private final Object batchLock = new Object();
    private static final int BATCH_SIZE = 500; // Flush after 500 blocks

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

        sessionId = "session_" + Instant.now().getEpochSecond();
        sessionDir = Paths.get(config.recordingDirectory, sessionId);
        ticksFile = sessionDir.resolve("ticks.jsonl");
        inputsFile = sessionDir.resolve("inputs.jsonl");
        videoFile = sessionDir.resolve("video.mp4");
        worldEventsFile = sessionDir.resolve("world_events.jsonl");  // Phase 4.1: Event-based tracking
        frameMappingFile = sessionDir.resolve("frame_mapping.jsonl");

        currentTick.set(0);
        tickStartTime = System.currentTimeMillis();
        previousInventory = null; // Reset inventory tracking
        previousNearbyEntities = null; // Reset entity tracking (Phase 4)
        previousNearbyBlocks = null; // Reset block tracking (Phase 4)

        // Phase 4.1: Clear master maps for new session
        masterBlockMap.clear();
        masterEntityMap.clear();
        lastSnapshotTick = 0;
        blockBatchBuffer.clear();

        // Initialize video encoder
        videoEncoder = new VideoEncoder(config);
        try {
            videoEncoder.startRecording(videoFile.toFile());
        } catch (java.io.IOException e) {
            System.err.println("[Blockscope] Failed to start video encoder: " + e.getMessage());
            e.printStackTrace();
            videoEncoder = null;
        }

        // Initialize metadata
        metadata = new SessionMetadata();
        metadata.sessionId = sessionId;
        metadata.startTimestamp = Instant.now().getEpochSecond();
        metadata.minecraftVersion = "1.16.5";
        metadata.modVersion = "0.1.0-alpha";

        SessionMetadata.RecordingConfig recordingConfig = new SessionMetadata.RecordingConfig();
        recordingConfig.resolutionWidth = config.resolutionWidth;
        recordingConfig.resolutionHeight = config.resolutionHeight;
        recordingConfig.aspectRatioMode = config.aspectRatioMode;
        recordingConfig.targetFps = config.targetFps;
        recordingConfig.recordingDirectory = config.recordingDirectory;
        metadata.config = recordingConfig;

        writer.start();
        isRecording.set(true);

        // Trigger chunk rebuilds to capture all currently visible blocks
        // This ensures blocks rendered before pressing R are captured
        triggerChunkRebuilds();

        // Write initial metadata
        writer.writeBytes(sessionDir.resolve("metadata.json"),
            metadata.toJson().getBytes(), true);

        // Write keybindings (to understand non-default controls)
        writer.writeBytes(sessionDir.resolve("keybindings.json"),
            KeybindingsExporter.exportKeybindings().getBytes(), true);

        if (config.showChatMessages) {
            MinecraftClient.getInstance().inGameHud.getChatHud().addMessage(
                new net.minecraft.text.LiteralText("[Blockscope] Recording started: " + sessionId));
        }

        System.out.println("[Blockscope] Recording started: " + sessionId);
    }

    public void stopRecording() {
        if (!isRecording.get()) {
            return;
        }

        // Flush any remaining batched blocks
        flushBlockBatch();

        isRecording.set(false);

        // Stop video encoder and finalize video file
        if (videoEncoder != null) {
            videoEncoder.stopRecording();
            videoEncoder = null;
        }

        // Write final metadata
        metadata.endTimestamp = Instant.now().getEpochSecond();
        writer.writeBytes(sessionDir.resolve("metadata.json"),
            metadata.toJson().getBytes(), true);

        writer.stop();

        if (config.showChatMessages) {
            MinecraftClient.getInstance().inGameHud.getChatHud().addMessage(
                new net.minecraft.text.LiteralText("[Blockscope] Recording stopped: " + sessionId));
        }

        System.out.println("[Blockscope] Recording stopped: " + sessionId);
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

        // Only capture inventory if it changed (optimization for disk space and performance)
        TickData.InventoryState currentInventory = captureInventoryState(client.player);
        if (previousInventory == null || !inventoryEquals(previousInventory, currentInventory)) {
            tickData.inventory = currentInventory;
            previousInventory = currentInventory;
        }
        // If inventory unchanged, tickData.inventory will be null (JSON will omit the field)

        tickData.gui = captureGuiState(client.currentScreen);
        tickData.crosshairTarget = captureCrosshairTarget(client);
        tickData.gamemode = client.interactionManager.getCurrentGameMode().getName();

        // Phase 4.1: Event-based world tracking (blocks/entities captured via mixins and events)

        // Flush block batch every 20 ticks (1 second) to ensure timely writes
        if (tick % 20 == 0 && !blockBatchBuffer.isEmpty()) {
            flushBlockBatch();
        }

        // Periodic snapshot every 1200 ticks (~1 minute) to prevent unbounded diff chains
        if (tick - lastSnapshotTick >= 1200) {
            writePeriodicSnapshot(client.player, client.world, tick);
            lastSnapshotTick = tick;
        }

        // Note: Phase 4 old code (captureNearbyEntitiesState/captureNearbyBlocksState) is deprecated
        // Now using event-based master map + diffs (see ChunkBuilderMixin, WorldChunkMixin)

        writer.writeJsonLine(ticksFile, tickData.toJson());
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

        // Capture and encode frame directly to video
        if (videoEncoder != null) {
            boolean captured = videoEncoder.captureAndEncodeFrame();
            if (captured) {
                // Record which tick this frame corresponds to
                int frameIndex = videoEncoder.getFrameCount() - 1;
                long tick = currentTick.get();
                writer.writeJsonLine(frameMappingFile,
                    "{\"frame\":" + frameIndex + ",\"tick\":" + tick + "}");
            }
        }
    }

    public void recordInputEvent(InputEvent event) {
        if (!isRecording.get()) {
            return;
        }

        event.tick = currentTick.get();
        event.offsetMs = System.currentTimeMillis() - tickStartTime;

        writer.writeJsonLine(inputsFile, event.toJson());
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
        state.fov = (float) client.options.fov;
        state.mouseSensitivity = client.options.mouseSensitivity;

        // Player state
        state.health = player.getHealth();
        state.hunger = player.getHungerManager().getFoodLevel();
        state.saturation = player.getHungerManager().getSaturationLevel();
        state.armor = player.getArmor();
        state.hotbarIndex = player.inventory.selectedSlot;

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

        state.hideGui = client.options.hudHidden; // F2 pressed
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

        // Accessibility/gameplay settings
        state.showSubtitles = client.options.showSubtitles;
        state.autoJumpEnabled = client.options.autoJump;

        return state;
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
            Identifier biomeId = world.getRegistryManager().get(Registry.BIOME_KEY).getId(world.getBiome(pos));
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

        // Main inventory (36 slots)
        state.mainInventory = new TickData.ItemStack[36];
        for (int i = 0; i < 36; i++) {
            ItemStack stack = player.inventory.getStack(i);
            if (!stack.isEmpty()) {
                state.mainInventory[i] = createItemStack(stack);
            }
        }

        // Armor (4 slots)
        state.armor = new TickData.ItemStack[4];
        for (int i = 0; i < 4; i++) {
            ItemStack stack = player.inventory.armor.get(i);
            if (!stack.isEmpty()) {
                state.armor[i] = createItemStack(stack);
            }
        }

        // Offhand
        ItemStack offhandStack = player.getOffHandStack();
        if (!offhandStack.isEmpty()) {
            state.offhand = createItemStack(offhandStack);
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

            // Get hovered slot using reflection (focusedSlot is protected in 1.16.5)
            try {
                java.lang.reflect.Field focusedSlotField = HandledScreen.class.getDeclaredField("focusedSlot");
                focusedSlotField.setAccessible(true);
                Slot focusedSlot = (Slot) focusedSlotField.get(handledScreen);
                if (focusedSlot != null) {
                    state.hoveredSlotIndex = focusedSlot.id;
                }
            } catch (Exception e) {
                // Failed to get focused slot, skip it
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

            Identifier blockId = Registry.BLOCK.getId(client.world.getBlockState(pos).getBlock());
            target.blockId = blockId.toString();

        } else if (hitResult.getType() == HitResult.Type.ENTITY) {
            EntityHitResult entityHit = (EntityHitResult) hitResult;
            Entity entity = entityHit.getEntity();

            target.type = "entity";
            target.entityUuid = entity.getUuidAsString();
            target.entityType = entity.getType().toString();

        } else {
            target.type = "none";
        }

        return target;
    }

    private TickData.ItemStack createItemStack(ItemStack stack) {
        Identifier itemId = Registry.ITEM.getId(stack.getItem());
        String id = itemId.toString();
        int count = stack.getCount();

        // Extract metadata
        Integer damage = null;
        Integer maxDamage = null;
        if (stack.isDamageable()) {
            damage = stack.getDamage();
            maxDamage = stack.getMaxDamage();
        }

        String customName = null;
        if (stack.hasCustomName()) {
            customName = stack.getName().getString();
        }

        String[] enchantments = null;
        if (stack.hasEnchantments()) {
            net.minecraft.nbt.NbtList enchantList = stack.getEnchantments();
            enchantments = new String[enchantList.size()];
            for (int i = 0; i < enchantList.size(); i++) {
                net.minecraft.nbt.NbtCompound enchant = enchantList.getCompound(i);
                String enchantId = enchant.getString("id");
                int level = enchant.getInt("lvl");
                enchantments[i] = enchantId + ":" + level;
            }
        }

        return new TickData.ItemStack(id, count, damage, maxDamage, customName, enchantments);
    }

    private boolean inventoryEquals(TickData.InventoryState a, TickData.InventoryState b) {
        // Compare main inventory
        if (!itemArrayEquals(a.mainInventory, b.mainInventory)) return false;
        // Compare armor
        if (!itemArrayEquals(a.armor, b.armor)) return false;
        // Compare offhand
        return itemStackEquals(a.offhand, b.offhand);
    }

    private boolean itemArrayEquals(TickData.ItemStack[] a, TickData.ItemStack[] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (!itemStackEquals(a[i], b[i])) return false;
        }
        return true;
    }

    private boolean itemStackEquals(TickData.ItemStack a, TickData.ItemStack b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;

        if (!a.id.equals(b.id)) return false;
        if (a.count != b.count) return false;

        // Compare damage
        if (a.damage == null ? b.damage != null : !a.damage.equals(b.damage)) return false;
        if (a.maxDamage == null ? b.maxDamage != null : !a.maxDamage.equals(b.maxDamage)) return false;

        // Compare custom name
        if (a.customName == null ? b.customName != null : !a.customName.equals(b.customName)) return false;

        // Compare enchantments
        if (a.enchantments == null && b.enchantments == null) return true;
        if (a.enchantments == null || b.enchantments == null) return false;
        if (a.enchantments.length != b.enchantments.length) return false;
        for (int i = 0; i < a.enchantments.length; i++) {
            if (!a.enchantments[i].equals(b.enchantments[i])) return false;
        }

        return true;
    }

    private void captureContainerContents(TickData.GuiState state, ScreenHandler handler, ClientPlayerEntity player) {
        // Count non-player slots (container slots)
        int playerInvStart = handler.slots.size() - 36; // Last 36 slots are usually player inventory
        if (playerInvStart <= 0) {
            return; // No container slots
        }

        // Capture container slots only (exclude player inventory)
        java.util.List<TickData.ItemStack> containerItems = new java.util.ArrayList<>();
        for (int i = 0; i < playerInvStart; i++) {
            Slot slot = handler.slots.get(i);
            ItemStack stack = slot.getStack();
            if (!stack.isEmpty()) {
                containerItems.add(createItemStack(stack));
            } else {
                containerItems.add(null); // Preserve slot indices
            }
        }

        if (!containerItems.isEmpty()) {
            state.containerContents = containerItems.toArray(new TickData.ItemStack[0]);
        }
    }

    private void captureCraftingState(TickData.GuiState state, ScreenHandler handler) {
        String handlerName = handler.getClass().getSimpleName();

        // Detect crafting screens (CraftingTableScreenHandler, PlayerScreenHandler)
        if (handlerName.contains("Crafting") || handlerName.contains("Player")) {
            // Find crafting grid and result slots
            // Crafting table: 3x3 grid (slots 1-9) + result (slot 0)
            // Player inventory: 2x2 grid (slots 1-4) + result (slot 0)

            int gridSize = handlerName.contains("CraftingTable") ? 9 : 4;
            java.util.List<TickData.ItemStack> gridItems = new java.util.ArrayList<>();

            for (int i = 1; i <= gridSize; i++) {
                if (i < handler.slots.size()) {
                    ItemStack stack = handler.slots.get(i).getStack();
                    gridItems.add(!stack.isEmpty() ? createItemStack(stack) : null);
                }
            }

            if (!gridItems.isEmpty()) {
                state.craftingGrid = gridItems.toArray(new TickData.ItemStack[0]);
            }

            // Capture crafting result (slot 0)
            if (!handler.slots.isEmpty()) {
                ItemStack resultStack = handler.slots.get(0).getStack();
                if (!resultStack.isEmpty()) {
                    state.craftingOutput = createItemStack(resultStack);
                }
            }
        }
    }

    // Phase 4: Block State Serialization (CRITICAL for rotation, facing, etc.)
    private String serializeBlockStateProperties(BlockState blockState) {
        // Convert block state properties to string format: "facing=north,half=bottom,powered=true"
        com.google.common.collect.ImmutableMap<Property<?>, Comparable<?>> properties = blockState.getEntries();

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
                    Identifier blockId = Registry.BLOCK.getId(blockState.getBlock());
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

        Identifier typeId = net.minecraft.block.entity.BlockEntityType.getId(blockEntity.getType());
        data.type = typeId.toString();

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

        // Furnace specific (use reflection for private fields)
        if (blockEntity instanceof AbstractFurnaceBlockEntity) {
            AbstractFurnaceBlockEntity furnace = (AbstractFurnaceBlockEntity) blockEntity;
            try {
                java.lang.reflect.Field burnTimeField = AbstractFurnaceBlockEntity.class.getDeclaredField("burnTime");
                burnTimeField.setAccessible(true);
                data.burnTime = (Integer) burnTimeField.get(furnace);

                java.lang.reflect.Field cookTimeField = AbstractFurnaceBlockEntity.class.getDeclaredField("cookTime");
                cookTimeField.setAccessible(true);
                data.cookTime = (Integer) cookTimeField.get(furnace);
            } catch (Exception e) {
                // Failed to get furnace data, skip it
            }
        }

        // Sign specific
        if (blockEntity instanceof SignBlockEntity) {
            SignBlockEntity sign = (SignBlockEntity) blockEntity;
            data.signText = new String[4];
            for (int i = 0; i < 4; i++) {
                data.signText[i] = sign.getTextOnRow(i).getString();
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
            entityData.uuid = entity.getUuidAsString();
            entityData.type = Registry.ENTITY_TYPE.getId(entity.getType()).toString();
            entityData.x = entity.getX();
            entityData.y = entity.getY();
            entityData.z = entity.getZ();
            entityData.pitch = entity.getPitch(1.0f);
            entityData.yaw = entity.getYaw(1.0f);

            // Common entity states
            entityData.onFire = entity.isOnFire();
            entityData.invisible = entity.isInvisible();
            if (entity.hasCustomName()) {
                entityData.customName = entity.getCustomName().getString();
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

    /**
     * Called when a block is rendered (added to chunk render mesh).
     * This captures what blocks the player can actually SEE.
     * Blocks are batched to prevent queue overflow.
     */
    public void onBlockRendered(World world, BlockPos pos, BlockState state) {
        if (!isRecording.get()) {
            return;
        }

        try {
            Identifier dimensionId = world.getRegistryKey().getValue();
            String dimension = dimensionId.toString();
            String blockKey = buildBlockKey(dimension, pos.getX(), pos.getY(), pos.getZ());

            // Check if we've already seen this block
            if (masterBlockMap.containsKey(blockKey)) {
                return; // Already in master map, no need to re-record
            }

            // Create block data
            WorldEvent.BlockData blockData = new WorldEvent.BlockData();
            blockData.dimension = dimension;
            blockData.x = pos.getX();
            blockData.y = pos.getY();
            blockData.z = pos.getZ();
            blockData.blockId = Registry.BLOCK.getId(state.getBlock()).toString();
            blockData.blockStateProperties = serializeBlockStateProperties(state);

            // Capture block entity data if present
            BlockEntity blockEntity = world.getBlockEntity(pos);
            if (blockEntity != null && config.captureBlockEntities) {
                blockData.blockEntity = captureBlockEntityData(blockEntity);
            }

            // Add to master map
            masterBlockMap.put(blockKey, blockData);

            // Add to batch buffer instead of writing immediately
            blockBatchBuffer.put(blockKey, blockData);

            // Flush batch if it gets too large
            if (blockBatchBuffer.size() >= BATCH_SIZE) {
                flushBlockBatch();
            }
        } catch (Exception e) {
            System.err.println("[Blockscope] Error in onBlockRendered: " + e.getMessage());
        }
    }

    /**
     * Flush accumulated block_seen events to disk.
     * Writes all batched blocks as individual events (for compatibility with existing data format).
     */
    private void flushBlockBatch() {
        synchronized (batchLock) {
            if (blockBatchBuffer.isEmpty()) {
                return;
            }

            // Write all batched blocks
            int tick = (int) currentTick.get();
            for (WorldEvent.BlockData blockData : blockBatchBuffer.values()) {
                WorldEvent event = new WorldEvent();
                event.tick = tick;
                event.event = "block_seen";
                event.dimension = blockData.dimension;
                event.x = blockData.x;
                event.y = blockData.y;
                event.z = blockData.z;
                event.blockId = blockData.blockId;
                event.blockStateProperties = blockData.blockStateProperties;
                event.blockEntity = blockData.blockEntity;

                // Use direct write to bypass queue
                try {
                    com.google.gson.Gson gson = new com.google.gson.Gson();
                    String json = gson.toJson(event);
                    writer.writeDirectly(worldEventsFile, json);
                } catch (Exception e) {
                    System.err.println("[Blockscope] Error writing batched block event: " + e.getMessage());
                }
            }

            int batchSize = blockBatchBuffer.size();
            blockBatchBuffer.clear();
            System.out.println("[Blockscope] Flushed " + batchSize + " block_seen events");
        }
    }

    /**
     * Called when a block changes (player breaks/places, piston moves, etc.).
     * This captures block updates as diffs.
     */
    public void onBlockChanged(World world, BlockPos pos, BlockState oldState, BlockState newState) {
        if (!isRecording.get()) {
            return;
        }

        try {
            Identifier dimensionId = world.getRegistryKey().getValue();
            String dimension = dimensionId.toString();
            String blockKey = buildBlockKey(dimension, pos.getX(), pos.getY(), pos.getZ());

            // Update master map
            if (newState.isAir()) {
                // Block was removed
                masterBlockMap.remove(blockKey);
            } else {
                // Block was changed/placed
                WorldEvent.BlockData blockData = new WorldEvent.BlockData();
                blockData.dimension = dimension;
                blockData.x = pos.getX();
                blockData.y = pos.getY();
                blockData.z = pos.getZ();
                blockData.blockId = Registry.BLOCK.getId(newState.getBlock()).toString();
                blockData.blockStateProperties = serializeBlockStateProperties(newState);

                BlockEntity blockEntity = world.getBlockEntity(pos);
                if (blockEntity != null && config.captureBlockEntities) {
                    blockData.blockEntity = captureBlockEntityData(blockEntity);
                }

                masterBlockMap.put(blockKey, blockData);
            }

            // Write "block_changed" event
            WorldEvent event = new WorldEvent();
            event.tick = (int) currentTick.get();
            event.event = "block_changed";
            event.dimension = dimension;
            event.x = pos.getX();
            event.y = pos.getY();
            event.z = pos.getZ();
            event.blockId = Registry.BLOCK.getId(newState.getBlock()).toString();
            event.blockStateProperties = serializeBlockStateProperties(newState);

            writeWorldEvent(event);
        } catch (Exception e) {
            System.err.println("[Blockscope] Error in onBlockChanged: " + e.getMessage());
        }
    }

    /**
     * Write a world event to world_events.jsonl
     */
    private void writeWorldEvent(WorldEvent event) {
        try {
            com.google.gson.Gson gson = new com.google.gson.Gson();
            String json = gson.toJson(event);
            writer.writeJsonLine(worldEventsFile, json);
        } catch (Exception e) {
            System.err.println("[Blockscope] Error writing world event: " + e.getMessage());
        }
    }

    /**
     * Build a unique key for the block map: "dimension:x:y:z"
     */
    private String buildBlockKey(String dimension, int x, int y, int z) {
        return dimension + ":" + x + ":" + y + ":" + z;
    }

    /**
     * Write a periodic snapshot of nearby blocks and entities.
     * This prevents unbounded diff chains and ensures data integrity.
     * Called every 1200 ticks (~1 minute).
     */
    private void writePeriodicSnapshot(ClientPlayerEntity player, World world, long tick) {
        try {
            Identifier dimensionId = world.getRegistryKey().getValue();
            String dimension = dimensionId.toString();

            BlockPos playerPos = player.getBlockPos();
            int radius = config.blockCaptureRadius;

            // Collect blocks within 8x8x8 cube around player
            java.util.List<WorldEvent.BlockData> nearbyBlocks = new java.util.ArrayList<>();
            for (int x = -radius; x <= radius; x++) {
                for (int y = -radius; y <= radius; y++) {
                    for (int z = -radius; z <= radius; z++) {
                        BlockPos pos = playerPos.add(x, y, z);
                        String blockKey = buildBlockKey(dimension, pos.getX(), pos.getY(), pos.getZ());

                        // Check if this block is in our master map (we've seen it)
                        WorldEvent.BlockData blockData = masterBlockMap.get(blockKey);
                        if (blockData != null) {
                            nearbyBlocks.add(blockData);
                        }
                    }
                }
            }

            // Collect nearby entities (reuse existing code for consistency)
            // Note: This still uses the old Phase 4 code temporarily
            // TODO: Refactor entity tracking to event-based system like blocks
            java.util.List<WorldEvent.EntityData> nearbyEntities = new java.util.ArrayList<>();
            if (config.captureNearbyEntities) {
                TickData.NearbyEntitiesState entitiesState = captureNearbyEntitiesState(player, world);
                if (entitiesState != null && entitiesState.entities != null) {
                    for (TickData.EntityData entityData : entitiesState.entities) {
                        WorldEvent.EntityData eventEntity = new WorldEvent.EntityData();
                        eventEntity.uuid = entityData.uuid;
                        eventEntity.type = entityData.type;
                        eventEntity.x = entityData.x;
                        eventEntity.y = entityData.y;
                        eventEntity.z = entityData.z;
                        eventEntity.pitch = entityData.pitch;
                        eventEntity.yaw = entityData.yaw;
                        eventEntity.health = entityData.health;
                        eventEntity.maxHealth = entityData.maxHealth;
                        // Note: Equipment, item, age, etc. can be added if needed
                        nearbyEntities.add(eventEntity);
                    }
                }
            }

            // Create snapshot event
            WorldEvent snapshotEvent = new WorldEvent();
            snapshotEvent.tick = (int) tick;
            snapshotEvent.event = "snapshot";

            WorldEvent.SnapshotData snapshotData = new WorldEvent.SnapshotData();
            snapshotData.nearbyBlocks = nearbyBlocks.toArray(new WorldEvent.BlockData[0]);
            snapshotData.nearbyEntities = nearbyEntities.toArray(new WorldEvent.EntityData[0]);
            snapshotEvent.snapshot = snapshotData;

            writeWorldEvent(snapshotEvent);

            System.out.println("[Blockscope] Periodic snapshot written: " + nearbyBlocks.size() + " blocks, " + nearbyEntities.size() + " entities");
        } catch (Exception e) {
            System.err.println("[Blockscope] Error writing periodic snapshot: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Trigger chunk rebuilds to capture all currently visible blocks.
     * This is called when recording starts to ensure blocks rendered
     * before pressing R are captured.
     */
    private void triggerChunkRebuilds() {
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.worldRenderer != null) {
            // Schedule chunk rebuilds to trigger on next render
            // This will cause ChunkBuilderMixin to fire for all visible chunks
            client.worldRenderer.reload();
            System.out.println("[Blockscope] Triggered chunk rebuilds to capture initial visible blocks");
        }
    }
}
