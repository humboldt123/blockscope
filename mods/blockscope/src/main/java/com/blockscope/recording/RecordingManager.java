package com.blockscope.recording;

import com.blockscope.model.*;
import com.blockscope.util.Config;
import com.blockscope.util.KeybindingsExporter;
import com.blockscope.util.PlayerAnonymizer;
import com.replaymod.core.ReplayMod;
import com.replaymod.recording.ReplayModRecording;
import com.replaymod.recording.Setting;
import com.replaymod.recording.packet.PacketListener;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.text.Text;
import net.minecraft.client.gui.screen.ingame.HandledScreen;
import net.minecraft.client.gui.screen.recipebook.RecipeBookProvider;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;
import net.minecraft.entity.ItemEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.EquipmentSlot;
import net.minecraft.item.ItemStack;
import net.minecraft.block.entity.AbstractFurnaceBlockEntity;
import net.minecraft.block.entity.SignBlockEntity;
import net.minecraft.inventory.Inventory;
import net.minecraft.screen.ScreenHandler;
import net.minecraft.screen.slot.Slot;
import net.minecraft.registry.Registries;
import net.minecraft.util.Identifier;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Records per-tick metadata (player state, inventory, world state) alongside
 * a ReplayMod recording. Also handles video streaming via SegmentedTSEncoder
 * and uploads everything to the server at the end.
 */
public class RecordingManager {
    private static RecordingManager instance;

    private final Config config;
    private final AsyncWriter writer;
    private SegmentedTSEncoder segmentedEncoder;

    private final AtomicBoolean isRecording = new AtomicBoolean(false);
    private final AtomicLong currentTick = new AtomicLong(0);
    private long tickStartTime;
    private boolean startMessagePending = false;
    private long sessionStartTime = 0;
    private volatile boolean mcprUploaded = false;
    // Exact path PacketListener is writing to — grabbed via reflection while listener is live
    private volatile java.nio.file.Path capturedReplayPath = null;

    private String sessionId;
    private Path sessionDir;
    private SessionMetadata metadata;

    // Where ReplayMod writes its .mcpr files
    private String replayFolder;

    // Cached reflection fields
    private static java.lang.reflect.Field focusedSlotField = null;
    private static java.lang.reflect.Field outputPathField = null;
    static {
        try {
            focusedSlotField = HandledScreen.class.getDeclaredField("focusedSlot");
            focusedSlotField.setAccessible(true);
        } catch (Exception e) {
            System.err.println("[Blockscope] Failed to cache focusedSlot field: " + e.getMessage());
        }
        try {
            outputPathField = PacketListener.class.getDeclaredField("outputPath");
            outputPathField.setAccessible(true);
        } catch (Exception e) {
            System.err.println("[Blockscope] Failed to cache outputPath field: " + e.getMessage());
        }
    }

    private RecordingManager() {
        this.config = Config.getInstance();
        this.writer = new AsyncWriter();
    }

    public static RecordingManager getInstance() {
        if (instance == null) instance = new RecordingManager();
        return instance;
    }

    public void startRecording(String replayFolder) {
        if (isRecording.get()) return;

        this.replayFolder = replayFolder;
        PlayerAnonymizer.getInstance().resetSession();

        // Skip rename dialog and post-processing — we just want the raw .mcpr quickly
        try {
            ReplayMod.instance.getSettingsRegistry().set(Setting.RENAME_DIALOG, false);
            ReplayMod.instance.getSettingsRegistry().set(Setting.AUTO_POST_PROCESS, false);
        } catch (Exception e) {
            System.err.println("[Blockscope] Could not configure ReplayMod settings: " + e.getMessage());
        }

        sessionId = "session_" + Instant.now().getEpochSecond();
        sessionDir = Paths.get(config.recordingDirectory, sessionId);
        currentTick.set(0);
        tickStartTime = System.currentTimeMillis();
        sessionStartTime = tickStartTime;
        mcprUploaded = false;
        capturedReplayPath = null;

        // Create session directory so ffmpeg can write segments to it
        try {
            Files.createDirectories(sessionDir);
        } catch (Exception e) {
            System.err.println("[Blockscope] Failed to create session directory: " + e.getMessage());
        }

        // Init session on server — fail loud if unreachable
        if (!serverPost("/init-session?session_id=" + sessionId, null, null)) {
            sendChatMessage("§c[Blockscope] §fCannot reach server at " + config.serverUrl +
                " — ticks/video won't save!");
        }

        // Start streaming writers for ticks, inputs, frame mapping
        writer.start(sessionId, config.serverUrl);

        // Start video encoder
        try {
            segmentedEncoder = new SegmentedTSEncoder(config, sessionId, config.serverUrl, sessionDir);
            segmentedEncoder.startRecording();
        } catch (Exception e) {
            String msg = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
            System.err.println("[Blockscope] Failed to start video: " + msg);
            segmentedEncoder = null;
            // Tell the player immediately — don't let this fail silently
            final String chatMsg = msg;
            MinecraftClient.getInstance().execute(() ->
                sendChatMessage("§c[Blockscope] §fVideo error: " + chatMsg));
        }

        // Build and upload metadata
        metadata = new SessionMetadata();
        metadata.sessionId = sessionId;
        metadata.startTimestamp = Instant.now().getEpochSecond();
        metadata.minecraftVersion = "1.19.4";
        metadata.modVersion = "0.1.0-alpha";
        SessionMetadata.RecordingConfig rc = new SessionMetadata.RecordingConfig();
        rc.resolutionWidth = config.resolutionWidth;
        rc.resolutionHeight = config.resolutionHeight;
        rc.targetFps = config.targetFps;
        metadata.config = rc;
        uploadMetadata();

        isRecording.set(true);
        startMessagePending = true;
        System.out.println("[Blockscope] Recording started: " + sessionId);
    }

    public void stopRecording() {
        if (!isRecording.get()) return;
        isRecording.set(false);

        // Stop video
        if (segmentedEncoder != null) {
            segmentedEncoder.stopRecording();
            segmentedEncoder = null;
        }

        // Flush writers
        writer.stop();

        // Update and upload final metadata
        metadata.endTimestamp = Instant.now().getEpochSecond();
        uploadMetadata();

        // Tell server to finalize video segments → video.mp4
        serverPost("/finalize-video?session_id=" + sessionId, null, null);

        // Start background watcher to upload the .mcpr as soon as ReplayMod writes it.
        // We know the exact target path from outputPath reflection — no directory scanning.
        startMcprWatcher();

        System.out.println("[Blockscope] Recording stopped: " + sessionId);
        sendChatMessage("§c[Blockscope] §fRecording saved");
    }

    public void onTick(MinecraftClient client) {
        if (!isRecording.get()) return;
        if (client.player == null || client.world == null) return;

        if (startMessagePending) {
            startMessagePending = false;
            sendChatMessage("§c[Blockscope] §fRecording started");
        }

        // Grab PacketListener's outputPath once while it's still live
        if (capturedReplayPath == null && outputPathField != null) {
            try {
                PacketListener listener = ReplayModRecording.instance
                        .getConnectionEventHandler().getPacketListener();
                if (listener != null) {
                    capturedReplayPath = (java.nio.file.Path) outputPathField.get(listener);
                }
            } catch (Exception ignored) {}
        }

        long tick = currentTick.getAndIncrement();
        tickStartTime = System.currentTimeMillis();

        TickData tickData = new TickData();
        tickData.tick = tick;
        tickData.replayMs = getReplayDurationMs();
        tickData.player = capturePlayerState(client.player);
        tickData.world = captureWorldState(client.world);
        tickData.inventory = captureInventoryState(client.player);
        tickData.gui = captureGuiState(client.currentScreen);
        tickData.crosshairTarget = captureCrosshairTarget(client);
        tickData.gamemode = client.interactionManager.getCurrentGameMode().getName();

        writer.writeTick(tickData.toJson());
    }

    public void onRenderFrame() {
        if (!isRecording.get() || segmentedEncoder == null) return;
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.isPaused()) return;

        boolean captured = segmentedEncoder.captureAndEncodeFrame();
        if (captured) {
            int frameIndex = segmentedEncoder.getFrameCount() - 1;
            long tick = currentTick.get();
            writer.writeFrameMapping("{\"frame\":" + frameIndex + ",\"tick\":" + tick + "}");
        }
    }

    public void recordInputEvent(InputEvent event) {
        if (!isRecording.get()) return;
        event.tick = currentTick.get();
        event.offsetMs = System.currentTimeMillis() - tickStartTime;
        writer.writeInput(event.toJson());
    }

    public boolean isRecording() { return isRecording.get(); }
    public long getCurrentTick() { return currentTick.get(); }
    public int getFrameCount() { return segmentedEncoder != null ? segmentedEncoder.getFrameCount() : 0; }
    public String getSessionId() { return sessionId; }

    private long getReplayDurationMs() {
        try {
            PacketListener listener = ReplayModRecording.instance
                    .getConnectionEventHandler().getPacketListener();
            return listener != null ? listener.getCurrentDuration() : 0;
        } catch (Exception e) {
            return 0;
        }
    }

    // ---- Capture helpers (unchanged from original) ----

    private TickData.PlayerState capturePlayerState(ClientPlayerEntity player) {
        MinecraftClient client = MinecraftClient.getInstance();
        TickData.PlayerState state = new TickData.PlayerState();
        state.x = player.getX();
        state.y = player.getY();
        state.z = player.getZ();
        state.pitch = player.getPitch(1.0f);
        state.yaw = player.getYaw(1.0f);

        // Capture the actual rendered camera — position is collision-adjusted in third-person,
        // and pitch/yaw differ from player values in third_person_front (F5 front view).
        net.minecraft.client.render.Camera cam = client.gameRenderer.getCamera();
        if (cam.isReady()) {
            TickData.CameraState cs = new TickData.CameraState();
            net.minecraft.util.math.Vec3d pos = cam.getPos();
            cs.x = pos.x;
            cs.y = pos.y;
            cs.z = pos.z;
            cs.pitch = cam.getPitch();
            cs.yaw = cam.getYaw();
            state.camera = cs;
        }
        // GameRenderer multiplies the settings FOV by getFovMultiplier() before rendering.
        // This accounts for sprinting, speed/slowness effects, etc.
        float fovMultiplier = (client.player != null) ? client.player.getFovMultiplier() : 1.0f;
        state.fov = (float)(client.options.getFov().getValue() * fovMultiplier);
        state.mouseSensitivity = client.options.getMouseSensitivity().getValue();
        state.health = player.getHealth();
        state.hunger = player.getHungerManager().getFoodLevel();
        state.saturation = player.getHungerManager().getSaturationLevel();
        state.armor = player.getArmor();
        state.hotbarIndex = player.getInventory().selectedSlot;
        ItemStack heldStack = player.getMainHandStack();
        if (!heldStack.isEmpty()) state.heldItem = createItemStack(heldStack);
        state.chatOpen = (client.currentScreen != null &&
                client.currentScreen.getClass().getSimpleName().equals("ChatScreen"));
        state.isPaused = client.isPaused();
        state.menuOpen = (client.currentScreen != null);
        state.textInputActive = false;
        if (client.currentScreen != null) {
            String sn = client.currentScreen.getClass().getSimpleName();
            state.textInputActive = sn.equals("ChatScreen") || sn.equals("BookEditScreen") ||
                    sn.equals("SignEditScreen") || sn.equals("AnvilScreen") ||
                    sn.equals("CommandBlockScreen");
        }
        state.hideGui = client.options.hudHidden;
        state.debugScreenVisible = client.options.debugEnabled;
        switch (client.options.getPerspective()) {
            case FIRST_PERSON: state.cameraPerspective = "first_person"; break;
            case THIRD_PERSON_BACK: state.cameraPerspective = "third_person_back"; break;
            case THIRD_PERSON_FRONT: state.cameraPerspective = "third_person_front"; break;
            default: state.cameraPerspective = "first_person";
        }
        state.statusEffects = captureStatusEffects(player);
        state.showSubtitles = client.options.getShowSubtitles().getValue();
        state.autoJumpEnabled = client.options.getAutoJump().getValue();
        return state;
    }

    private TickData.StatusEffect[] captureStatusEffects(ClientPlayerEntity player) {
        java.util.Collection<net.minecraft.entity.effect.StatusEffectInstance> effects = player.getStatusEffects();
        if (effects.isEmpty()) return null;
        java.util.List<TickData.StatusEffect> list = new java.util.ArrayList<>();
        for (net.minecraft.entity.effect.StatusEffectInstance effect : effects) {
            TickData.StatusEffect se = new TickData.StatusEffect();
            Identifier id = Registries.STATUS_EFFECT.getId(effect.getEffectType());
            se.effectId = id != null ? id.toString() : "unknown";
            se.amplifier = effect.getAmplifier();
            se.duration = effect.getDuration();
            se.ambient = effect.isAmbient();
            se.showParticles = effect.shouldShowParticles();
            se.showIcon = effect.shouldShowIcon();
            list.add(se);
        }
        return list.toArray(new TickData.StatusEffect[0]);
    }

    private TickData.WorldState captureWorldState(World world) {
        TickData.WorldState state = new TickData.WorldState();
        state.worldTime = world.getTime();
        state.timeOfDay = world.getTimeOfDay();
        state.dimension = world.getRegistryKey().getValue().toString();
        ClientPlayerEntity player = MinecraftClient.getInstance().player;
        if (player != null) {
            BlockPos pos = player.getBlockPos();
            Identifier biomeId = world.getBiome(pos).getKey().map(k -> k.getValue()).orElse(null);
            state.biome = biomeId != null ? biomeId.toString() : "unknown";
        }
        state.weather = world.isRaining() ? (world.isThundering() ? "thunder" : "rain") : "clear";
        return state;
    }

    private TickData.InventoryState captureInventoryState(ClientPlayerEntity player) {
        TickData.InventoryState state = new TickData.InventoryState();
        java.util.List<TickData.SlotItem> slots = new java.util.ArrayList<>();
        for (int i = 0; i < 36; i++) {
            ItemStack stack = player.getInventory().getStack(i);
            if (!stack.isEmpty()) {
                TickData.SlotItem si = new TickData.SlotItem();
                si.slot = i;
                si.item = createItemStack(stack);
                slots.add(si);
            }
        }
        for (int i = 0; i < 4; i++) {
            ItemStack stack = player.getInventory().armor.get(i);
            if (!stack.isEmpty()) {
                TickData.SlotItem si = new TickData.SlotItem();
                si.slot = 36 + i;
                si.item = createItemStack(stack);
                slots.add(si);
            }
        }
        ItemStack offhand = player.getOffHandStack();
        if (!offhand.isEmpty()) {
            TickData.SlotItem si = new TickData.SlotItem();
            si.slot = 40;
            si.item = createItemStack(offhand);
            slots.add(si);
        }
        state.slots = slots.toArray(new TickData.SlotItem[0]);
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.currentScreen instanceof HandledScreen) {
            ItemStack cursor = ((HandledScreen<?>) client.currentScreen).getScreenHandler().getCursorStack();
            if (!cursor.isEmpty()) state.cursorStack = createItemStack(cursor);
        }
        return state;
    }

    private TickData.GuiState captureGuiState(Screen screen) {
        TickData.GuiState state = new TickData.GuiState();
        if (screen == null) return state;
        state.screenType = screen.getClass().getSimpleName();
        MinecraftClient client = MinecraftClient.getInstance();
        double mx = client.mouse.getX() * client.getWindow().getScaledWidth() / client.getWindow().getWidth();
        double my = client.mouse.getY() * client.getWindow().getScaledHeight() / client.getWindow().getHeight();
        state.cursorX = (int) mx;
        state.cursorY = (int) my;
        if (screen instanceof HandledScreen) {
            HandledScreen<?> hs = (HandledScreen<?>) screen;
            ScreenHandler handler = hs.getScreenHandler();
            if (focusedSlotField != null) {
                try {
                    Slot focused = (Slot) focusedSlotField.get(hs);
                    if (focused != null) state.hoveredSlotIndex = focused.id;
                } catch (Exception ignored) {}
            }
            captureContainerContents(state, handler);
            if (screen instanceof RecipeBookProvider) {
                state.recipeBookOpen = ((RecipeBookProvider) screen).getRecipeBookWidget().isOpen();
            }
        }
        return state;
    }

    private void captureContainerContents(TickData.GuiState state, ScreenHandler handler) {
        int playerInvStart = handler.slots.size() - 36;
        if (playerInvStart <= 0) return;
        TickData.ContainerData container = new TickData.ContainerData();
        container.type = "minecraft:" + handler.getClass().getSimpleName().replace("ScreenHandler", "").toLowerCase();
        container.size = playerInvStart;
        java.util.List<TickData.SlotItem> cSlots = new java.util.ArrayList<>();
        for (int i = 0; i < playerInvStart; i++) {
            ItemStack stack = handler.slots.get(i).getStack();
            if (!stack.isEmpty()) {
                TickData.SlotItem si = new TickData.SlotItem();
                si.slot = i;
                si.item = createItemStack(stack);
                cSlots.add(si);
            }
        }
        container.slots = cSlots.toArray(new TickData.SlotItem[0]);
        state.container = container;
    }

    private TickData.CrosshairTarget captureCrosshairTarget(MinecraftClient client) {
        TickData.CrosshairTarget target = new TickData.CrosshairTarget();
        HitResult hit = client.crosshairTarget;
        if (hit == null) { target.type = "none"; return target; }
        if (hit.getType() == HitResult.Type.BLOCK) {
            BlockPos pos = ((BlockHitResult) hit).getBlockPos();
            target.type = "block";
            target.blockPos = new TickData.BlockPos(pos.getX(), pos.getY(), pos.getZ());
            target.blockId = Registries.BLOCK.getId(client.world.getBlockState(pos).getBlock()).toString();
        } else if (hit.getType() == HitResult.Type.ENTITY) {
            Entity entity = ((EntityHitResult) hit).getEntity();
            target.type = "entity";
            target.entityUuid = PlayerAnonymizer.getInstance().anonymizeUuid(entity.getUuidAsString());
            target.entityType = entity.getType().toString();
        } else {
            target.type = "none";
        }
        return target;
    }

    private TickData.ItemStack createItemStack(ItemStack stack) {
        TickData.ItemStack item = new TickData.ItemStack(
                Registries.ITEM.getId(stack.getItem()).toString(), stack.getCount());
        if (stack.isDamageable()) {
            item.damage = stack.getDamage();
            item.maxDamage = stack.getMaxDamage();
            if (item.maxDamage > 0)
                item.durability = (float)(item.maxDamage - item.damage) / item.maxDamage;
        }
        if (stack.hasEnchantments()) {
            net.minecraft.nbt.NbtList el = stack.getEnchantments();
            java.util.List<TickData.Enchantment> enchants = new java.util.ArrayList<>();
            for (int i = 0; i < el.size(); i++) {
                net.minecraft.nbt.NbtCompound e = el.getCompound(i);
                TickData.Enchantment enc = new TickData.Enchantment();
                enc.id = e.getString("id");
                enc.level = e.getInt("lvl");
                enchants.add(enc);
            }
            item.enchantments = enchants.toArray(new TickData.Enchantment[0]);
        }
        return item;
    }

    // ---- Server communication ----

    private void uploadMetadata() {
        serverPost("/upload-metadata?session_id=" + sessionId,
                "application/json", metadata.toJson().getBytes());
    }

    private File findNewestMcpr(File dir) {
        File result = null;
        File[] entries = dir.listFiles();
        if (entries == null) return null;
        for (File f : entries) {
            if (f.isDirectory()) {
                File sub = findNewestMcpr(f);
                if (sub != null && (result == null || sub.lastModified() > result.lastModified()))
                    result = sub;
            } else if (f.getName().endsWith(".mcpr") && f.lastModified() >= sessionStartTime) {
                if (result == null || f.lastModified() > result.lastModified())
                    result = f;
            }
        }
        return result;
    }

    public void startMcprWatcher() {
        if (mcprUploaded || sessionId == null) return;
        final String watchSessionId = sessionId;
        new Thread(() -> {
            long deadline = System.currentTimeMillis() + 5 * 60 * 1000; // 5 min max
            while (System.currentTimeMillis() < deadline && !mcprUploaded) {
                try { Thread.sleep(2000); } catch (InterruptedException e) { break; }
                if (uploadMcprFile()) break;
            }
            if (!mcprUploaded) {
                System.err.println("[Blockscope] .mcpr watcher timed out for " + watchSessionId);
            }
        }, "blockscope-mcpr-watcher").start();
    }

    /** Returns true if the .mcpr was found and successfully uploaded, false to retry. */
    public boolean uploadMcprFile() {
        if (replayFolder == null || sessionId == null) return false;
        if (mcprUploaded) return true;

        // Try the exact path PacketListener was writing to first (fastest)
        File target = findTarget();
        if (target == null) return false;
        long size = target.length();
        System.out.println("[Blockscope] Uploading replay: " + target.getName() + " (" + size + " bytes)");
        try {
            String url = config.serverUrl + "/upload-replay?session_id=" + sessionId +
                    "&filename=" + java.net.URLEncoder.encode(target.getName(), "UTF-8");
            java.net.HttpURLConnection conn = (java.net.HttpURLConnection) new java.net.URL(url).openConnection();
            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/octet-stream");
            conn.setRequestProperty("Content-Length", String.valueOf(size));
            conn.setChunkedStreamingMode(256 * 1024);
            try (java.io.InputStream in = new java.io.FileInputStream(target);
                 java.io.OutputStream out = conn.getOutputStream()) {
                byte[] buf = new byte[256 * 1024];
                int n;
                while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
            }
            int code = conn.getResponseCode();
            if (code != 200) {
                System.err.println("[Blockscope] .mcpr upload returned " + code);
                return false;
            }
            System.out.println("[Blockscope] .mcpr upload complete");
            mcprUploaded = true;
            conn.disconnect();
            return true;
        } catch (Exception e) {
            System.err.println("[Blockscope] Failed to upload .mcpr: " + e.getMessage());
            return false;
        }
    }

    /**
     * Finds the .mcpr to upload. Checks capturedReplayPath (exact file ReplayMod
     * was writing to) and its parent's siblings first — these appear before post-processing.
     * Falls back to a recursive scan of the whole replay folder.
     */
    private File findTarget() {
        // 1. Check the exact output path captured via reflection
        if (capturedReplayPath != null) {
            File f = capturedReplayPath.toFile();
            if (f.exists() && f.length() > 0 && isStable(f)) return f;
            // Also check the parent folder (ReplayMod may have moved it nearby)
            File parent = f.getParentFile();
            if (parent != null && parent.exists()) {
                File found = newestMcprIn(parent);
                if (found != null) return found;
            }
        }
        // 2. Check raw/ subfolder — file appears here before post-processing completes
        if (replayFolder != null) {
            File raw = new File(replayFolder, "raw");
            if (raw.exists()) {
                File found = newestMcprIn(raw);
                if (found != null) return found;
            }
        }
        // 3. Full recursive scan
        return replayFolder != null ? findNewestMcpr(new File(replayFolder)) : null;
    }

    private File newestMcprIn(File dir) {
        File result = null;
        File[] entries = dir.listFiles();
        if (entries == null) return null;
        for (File f : entries) {
            if (f.isFile() && f.getName().endsWith(".mcpr")
                    && f.lastModified() >= sessionStartTime
                    && f.length() > 0 && isStable(f)) {
                if (result == null || f.lastModified() > result.lastModified()) result = f;
            }
        }
        return result;
    }

    /** Returns true if the file size hasn't changed in the last 1 second (write complete). */
    private boolean isStable(File f) {
        long before = f.length();
        try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
        return f.exists() && f.length() == before && before > 0;
    }

    private void sendChatMessage(String message) {
        if (!config.showChatMessages) return;
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player != null) {
            client.player.sendMessage(Text.literal(message), false);
        }
    }

    private boolean serverPost(String path, String contentType, byte[] body) {
        try {
            java.net.HttpURLConnection conn = (java.net.HttpURLConnection)
                    new java.net.URL(config.serverUrl + path).openConnection();
            conn.setRequestMethod("POST");
            if (body != null) {
                conn.setDoOutput(true);
                conn.setRequestProperty("Content-Type", contentType);
                try (java.io.OutputStream out = conn.getOutputStream()) { out.write(body); }
            }
            int code = conn.getResponseCode();
            if (code != 200) System.err.println("[Blockscope] Server POST " + path + " returned " + code);
            conn.disconnect();
            return code == 200;
        } catch (Exception e) {
            System.err.println("[Blockscope] Server POST " + path + " failed: " + e.getMessage());
            return false;
        }
    }
}
