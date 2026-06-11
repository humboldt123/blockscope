package com.blockscope.agent;

import baritone.api.BaritoneAPI;
import baritone.api.IBaritone;
import baritone.api.pathing.goals.GoalBlock;
import baritone.api.pathing.goals.GoalXZ;
import baritone.api.pathing.goals.GoalYLevel;
import baritone.api.process.IBaritoneProcess;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayNetworking;
import net.minecraft.block.Block;
import net.minecraft.block.Blocks;
import net.minecraft.client.MinecraftClient;
import net.minecraft.item.Items;
import net.minecraft.text.Text;
import net.minecraft.util.ActionResult;
import net.minecraft.util.Hand;
import net.minecraft.util.Identifier;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Direction;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.LightType;

import java.util.Queue;
import java.util.Random;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Autonomous data-collection bot.  Three modes:
 *
 *   CREATIVE_SURVEY  — enables creative flight and uses Baritone's explore
 *                      process to sweep the map at ground level.
 *
 *   SURVIVAL_GATHER  — cycles: chop wood → surface ores → explore → deep ores.
 *                      Interior exploration is driven by a two-tier system:
 *                        1. structureHints queue — positions pushed by Lodestone's
 *                           StructureGuide over the "blockscope:structures" channel.
 *                        2. INTERIOR_BLOCKS scan — detects structure signatures in the
 *                           loaded world even when Lodestone is absent.
 *
 *   VOID_SCATTER     — void world with a connected branching walkway (VoidScatterGenerator).
 *                      Baritone's explore process follows the path naturally; no random
 *                      Y-goals are needed since staircases are baked into the world.
 *
 * Toggle with G.  Cycle modes (while stopped) with H.
 * Requires baritone-api-fabric-1.9.5.jar in mods folder.
 *
 * Call BotModule.init() once from the client mod initialiser to register
 * the optional "blockscope:structures" plugin channel listener.
 */
public class BotModule {

    public enum Mode { CREATIVE_SURVEY, SURVIVAL_GATHER, VOID_SCATTER }

    private static BotModule instance;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private Thread botThread;
    private Mode mode = Mode.CREATIVE_SURVEY;
    private final Random rng = new Random();

    // Smooth look state for creative survey
    private double lookYaw      = 0;
    private double lookPitch    = 0;
    private double targetYaw    = 0;
    private double targetPitch  = 0;
    private int    lookHoldTicks = 0;
    private boolean lookInitialized = false;
    private boolean useFreeLook = false;

    private int torchCheckTicks = 0;

    // Structure positions received from Lodestone's StructureGuide channel.
    // Empty when not connected to a Lodestone server — bot falls back to
    // local interior-block detection.
    private final Queue<BlockPos> structureHints = new ConcurrentLinkedQueue<>();

    private BotModule() {}

    public static BotModule getInstance() {
        if (instance == null) instance = new BotModule();
        return instance;
    }

    /**
     * Register the optional Lodestone structure-hint channel.  Call once from
     * BlockscopeClient.onInitializeClient() inside the baritonePresent block.
     * Safe to call even on servers without Lodestone — the channel simply receives no packets.
     */
    public static void init() {
        ClientPlayNetworking.registerGlobalReceiver(
            new Identifier("blockscope", "structures"),
            (client, handler, buf, responseSender) -> {
                int count = buf.readInt();
                BlockPos[] positions = new BlockPos[count];
                for (int i = 0; i < count; i++)
                    positions[i] = new BlockPos(buf.readInt(), 64, buf.readInt());
                client.execute(() -> {
                    BotModule bot = BotModule.getInstance();
                    for (BlockPos pos : positions) bot.structureHints.add(pos);
                });
            }
        );
    }

    public boolean isRunning() { return running.get(); }
    public Mode    getMode()   { return mode; }
    public void    setMode(Mode m) { if (!running.get()) mode = m; }

    public void cycleMode() {
        if (running.get()) { log("§c[Bot] Stop the bot before changing mode."); return; }
        mode = switch (mode) {
            case CREATIVE_SURVEY -> Mode.SURVIVAL_GATHER;
            case SURVIVAL_GATHER -> Mode.VOID_SCATTER;
            case VOID_SCATTER    -> Mode.CREATIVE_SURVEY;
        };
        log("§e[Bot] Mode → " + mode);
    }

    public void toggle() {
        if (running.get()) stop(); else start();
    }

    public void start() {
        if (running.getAndSet(true)) return;
        applyGlobalSettings();
        log("§a[Bot] Starting — " + mode);
        botThread = new Thread(() -> {
            try {
                if (mode == Mode.CREATIVE_SURVEY) runCreativeSurvey();
                else if (mode == Mode.VOID_SCATTER) runVoidScatter();
                else runSurvivalGather();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } catch (Exception e) {
                System.err.println("[BotModule] Uncaught: " + e);
                e.printStackTrace();
            } finally {
                running.set(false);
                cancelBaritone();
                log("§c[Bot] Stopped.");
            }
        }, "BotModule");
        botThread.setDaemon(true);
        botThread.start();
    }

    public void stop() {
        if (!running.getAndSet(false)) return;
        useFreeLook = false;
        lookInitialized = false;
        torchCheckTicks = 0;
        cancelBaritone();
        if (botThread != null) { botThread.interrupt(); botThread = null; }
    }

    /** Called every game tick from BlockscopeClient. Drives look in creative survey; places torches in survival. */
    public void onTick(MinecraftClient mc) {
        if (!running.get() || mc.player == null) return;
        tickTorchCheck(mc);
        if (!useFreeLook) return;

        if (!lookInitialized) {
            lookYaw   = mc.player.getYaw();
            lookPitch = mc.player.getPitch();
            targetYaw   = lookYaw;
            targetPitch = lookPitch;
            lookInitialized = true;
        }

        if (lookHoldTicks <= 0) {
            Vec3d vel = mc.player.getVelocity();
            double hSpeed = Math.sqrt(vel.x * vel.x + vel.z * vel.z);
            double baseYaw = hSpeed > 0.05
                ? Math.toDegrees(Math.atan2(-vel.x, vel.z))
                : lookYaw;
            boolean bigLook = rng.nextFloat() < 0.15f;
            double yawJitter   = bigLook ? (rng.nextDouble() * 2 - 1) * 75 : (rng.nextDouble() * 2 - 1) * 22;
            double pitchTarget = bigLook ? (rng.nextDouble() * 55 - 20) : (rng.nextDouble() * 18 - 5);
            targetYaw   = baseYaw + yawJitter;
            targetPitch = Math.max(-75, Math.min(60, pitchTarget));
            lookHoldTicks = 30 + rng.nextInt(70);
        }
        lookHoldTicks--;

        lookYaw   += (targetYaw   - lookYaw)   * 0.06;
        lookPitch += (targetPitch - lookPitch)  * 0.06;

        mc.player.setYaw((float) lookYaw);
        mc.player.setPitch((float) lookPitch);
    }

    // ── Torch placement ───────────────────────────────────────────────────────

    private void tickTorchCheck(MinecraftClient mc) {
        if (useFreeLook || mc.player == null || mc.world == null) return;
        if (++torchCheckTicks < 100) return;
        torchCheckTicks = 0;
        maybePlaceTorch(mc);
    }

    private void maybePlaceTorch(MinecraftClient mc) {
        BlockPos pos = mc.player.getBlockPos();
        if (mc.world.getLightLevel(LightType.BLOCK, pos) >= 5) return;
        if (mc.world.getLightLevel(LightType.SKY, pos) >= 5) return;

        int torchSlot = -1;
        for (int i = 0; i < 9; i++) {
            if (mc.player.getInventory().getStack(i).getItem() == Items.TORCH) { torchSlot = i; break; }
        }
        if (torchSlot == -1) return;

        Direction facing = Direction.fromRotation(mc.player.getYaw());
        Direction left   = facing.rotateYCounterclockwise();
        Direction right  = facing.rotateYClockwise();

        for (Direction wallDir : new Direction[]{left, right}) {
            for (int dy : new int[]{1, 0}) {
                BlockPos wallBlock  = pos.up(dy).offset(wallDir);
                BlockPos torchBlock = pos.up(dy);
                if (mc.world.getBlockState(wallBlock).isSolidBlock(mc.world, wallBlock)
                        && mc.world.getBlockState(torchBlock).isAir()) {
                    if (placeTorch(mc, torchSlot, wallBlock, wallDir.getOpposite())) return;
                }
            }
        }
        BlockPos below = pos.down();
        if (mc.world.getBlockState(below).isSolidBlock(mc.world, below) && mc.world.getBlockState(pos).isAir()) {
            placeTorch(mc, torchSlot, below, Direction.UP);
        }
    }

    private boolean placeTorch(MinecraftClient mc, int torchSlot, BlockPos target, Direction face) {
        int prevSlot = mc.player.getInventory().selectedSlot;
        mc.player.getInventory().selectedSlot = torchSlot;
        Vec3d hitVec = Vec3d.ofCenter(target).add(Vec3d.of(face.getVector()).multiply(0.5));
        BlockHitResult hit = new BlockHitResult(hitVec, face, target, false);
        ActionResult result = mc.interactionManager.interactBlock(mc.player, Hand.MAIN_HAND, hit);
        mc.player.swingHand(Hand.MAIN_HAND);
        mc.player.getInventory().selectedSlot = prevSlot;
        return result.isAccepted();
    }

    // ── Global Baritone settings ──────────────────────────────────────────────

    private void applyGlobalSettings() {
        MinecraftClient.getInstance().execute(() -> {
            var s = BaritoneAPI.getSettings();

            s.renderPath.value              = false;
            s.renderGoal.value              = false;
            s.renderGoalAnimated.value      = false;
            s.renderGoalXZBeacon.value      = false;
            s.renderSelection.value         = false;
            s.renderSelectionBoxes.value    = false;
            s.renderCachedChunks.value      = false;
            s.fadePath.value                = false;

            s.smoothLook.value              = true;
            s.smoothLookTicks.value         = 18;
            s.randomLooking.value           = 0.18;
            s.randomLooking113.value        = 0.10;
            s.freeLook.value                = false;

            s.enterPortal.value             = false;
            s.allowSprint.value             = true;
            s.sprintAscends.value           = true;
            s.notificationOnPathComplete.value = false;
        });
    }

    // ── Creative survey ───────────────────────────────────────────────────────

    private void runCreativeSurvey() throws InterruptedException {
        waitForPlayer();
        MinecraftClient mc = MinecraftClient.getInstance();
        mc.execute(() -> {
            if (mc.player != null) {
                mc.player.getAbilities().flying = true;
                mc.player.sendAbilitiesUpdate();
                BaritoneAPI.getSettings().allowPlace.value  = false;
                BaritoneAPI.getSettings().allowBreak.value  = true;
                BaritoneAPI.getSettings().freeLook.value    = true;
            }
        });
        useFreeLook     = true;
        lookInitialized = false;
        Thread.sleep(200);

        final int startX = mc.player.getBlockX();
        final int startZ = mc.player.getBlockZ();
        mc.execute(() -> baritone().getExploreProcess().explore(startX, startZ));
        log("§b[Bot] Creative survey — flying from (" + startX + ", " + startZ + ")");

        while (running.get()) Thread.sleep(5_000);
    }

    // ── Survival gather ───────────────────────────────────────────────────────

    private static final Block[] WOOD = {
        Blocks.OAK_LOG, Blocks.SPRUCE_LOG, Blocks.BIRCH_LOG,
        Blocks.JUNGLE_LOG, Blocks.ACACIA_LOG, Blocks.DARK_OAK_LOG,
    };
    private static final Block[] SURFACE_ORES = {
        Blocks.COAL_ORE, Blocks.IRON_ORE, Blocks.COPPER_ORE,
    };
    private static final Block[] DEEP_ORES = {
        Blocks.IRON_ORE, Blocks.GOLD_ORE, Blocks.REDSTONE_ORE,
        Blocks.LAPIS_ORE, Blocks.DIAMOND_ORE,
    };

    /**
     * Structure-interior signature blocks.  Presence within 24 blocks indicates
     * the bot is near an interior space worth exploring.
     *
     * Covers: village, pillager outpost, woodland mansion, stronghold,
     *         dungeon, mineshaft, jungle/desert temple.
     */
    private static final Block[] INTERIOR_BLOCKS = {
        // Village — beds (all 16 colours)
        Blocks.WHITE_BED, Blocks.ORANGE_BED, Blocks.MAGENTA_BED, Blocks.LIGHT_BLUE_BED,
        Blocks.YELLOW_BED, Blocks.LIME_BED, Blocks.PINK_BED, Blocks.GRAY_BED,
        Blocks.LIGHT_GRAY_BED, Blocks.CYAN_BED, Blocks.PURPLE_BED, Blocks.BLUE_BED,
        Blocks.BROWN_BED, Blocks.GREEN_BED, Blocks.RED_BED, Blocks.BLACK_BED,
        // Village / stronghold
        Blocks.BOOKSHELF,
        Blocks.OAK_PRESSURE_PLATE, Blocks.STONE_PRESSURE_PLATE,
        Blocks.SPRUCE_PRESSURE_PLATE, Blocks.BIRCH_PRESSURE_PLATE,
        // Dungeon / mineshaft
        Blocks.SPAWNER,
        Blocks.RAIL, Blocks.COBWEB,
        // Stronghold
        Blocks.END_PORTAL_FRAME, Blocks.IRON_BARS, Blocks.MOSSY_STONE_BRICKS,
        // Jungle temple
        Blocks.TRIPWIRE_HOOK,
        // Pillager outpost / woodland mansion
        Blocks.DARK_OAK_PLANKS,
    };

    private void runSurvivalGather() throws InterruptedException {
        waitForPlayer();
        MinecraftClient.getInstance().execute(() -> {
            BaritoneAPI.getSettings().allowBreak.value = true;
            BaritoneAPI.getSettings().allowPlace.value = true;
        });

        int cycle = 0;
        while (running.get()) {
            cycle++;
            log("§a[Bot] Survival cycle " + cycle);

            explore(150);
            if (!running.get()) break;

            mine(WOOD, 24, 180);
            if (!running.get()) break;

            mine(SURFACE_ORES, 16, 180);
            if (!running.get()) break;

            log("§a[Bot] Descending underground…");
            descend(12, 150);
            if (!running.get()) break;

            mine(DEEP_ORES, 24, 360);
            if (!running.get()) break;

            explore(120);
            if (!running.get()) break;

            log("§a[Bot] Returning to surface…");
            descend(70, 150);
            if (!running.get()) break;
        }
    }

    private void descend(int targetY, int maxSeconds) throws InterruptedException {
        MinecraftClient.getInstance().execute(() ->
            baritone().getCustomGoalProcess().setGoalAndPath(new GoalYLevel(targetY)));
        awaitProcess(baritone().getCustomGoalProcess(), maxSeconds);
    }

    // ── Void scatter ──────────────────────────────────────────────────────────

    // Mirror of VoidScatterGenerator.SCATTER_MATERIALS and block_vocab.py::SCATTER_BLOCKS.
    private static final Block[] VOID_SCATTER_BLOCKS = {
        Blocks.STONE, Blocks.COBBLESTONE, Blocks.DIRT, Blocks.GRASS_BLOCK,
        Blocks.SAND, Blocks.GRAVEL, Blocks.CLAY, Blocks.SANDSTONE,
        Blocks.MOSSY_COBBLESTONE, Blocks.STONE_BRICKS, Blocks.BRICKS, Blocks.OBSIDIAN,
        Blocks.COARSE_DIRT, Blocks.PACKED_ICE, Blocks.SNOW_BLOCK,
        Blocks.COAL_ORE, Blocks.IRON_ORE, Blocks.GOLD_ORE, Blocks.DIAMOND_ORE,
        Blocks.LAPIS_ORE, Blocks.REDSTONE_ORE, Blocks.EMERALD_ORE,
        Blocks.COAL_BLOCK, Blocks.IRON_BLOCK, Blocks.GOLD_BLOCK,
        Blocks.DIAMOND_BLOCK, Blocks.LAPIS_BLOCK, Blocks.EMERALD_BLOCK,
        Blocks.OAK_LOG, Blocks.OAK_PLANKS, Blocks.OAK_LEAVES, Blocks.BOOKSHELF,
        Blocks.WHITE_WOOL, Blocks.ORANGE_WOOL, Blocks.MAGENTA_WOOL, Blocks.LIGHT_BLUE_WOOL,
        Blocks.YELLOW_WOOL, Blocks.LIME_WOOL, Blocks.PINK_WOOL, Blocks.GRAY_WOOL,
        Blocks.LIGHT_GRAY_WOOL, Blocks.CYAN_WOOL, Blocks.PURPLE_WOOL, Blocks.BLUE_WOOL,
        Blocks.BROWN_WOOL, Blocks.GREEN_WOOL, Blocks.RED_WOOL, Blocks.BLACK_WOOL,
        Blocks.TERRACOTTA,
        Blocks.WHITE_TERRACOTTA, Blocks.ORANGE_TERRACOTTA, Blocks.YELLOW_TERRACOTTA,
        Blocks.RED_TERRACOTTA, Blocks.BROWN_TERRACOTTA, Blocks.GREEN_TERRACOTTA,
        Blocks.GRAY_TERRACOTTA, Blocks.CYAN_TERRACOTTA,
        Blocks.HAY_BLOCK, Blocks.PUMPKIN, Blocks.MELON,
        Blocks.BROWN_MUSHROOM_BLOCK, Blocks.RED_MUSHROOM_BLOCK, Blocks.MUSHROOM_STEM,
        Blocks.GLASS,
        Blocks.WHITE_STAINED_GLASS, Blocks.ORANGE_STAINED_GLASS, Blocks.RED_STAINED_GLASS,
        Blocks.BLUE_STAINED_GLASS, Blocks.GREEN_STAINED_GLASS, Blocks.YELLOW_STAINED_GLASS,
        Blocks.CRAFTING_TABLE, Blocks.CHEST, Blocks.FURNACE,
        Blocks.NETHERRACK,
    };

    private void runVoidScatter() throws InterruptedException {
        waitForPlayer();
        // Baritone resets settings on new server connection — wait for that first
        Thread.sleep(3000);
        MinecraftClient.getInstance().execute(() -> {
            var s = BaritoneAPI.getSettings();
            s.allowBreak.value         = true;
            s.allowPlace.value         = true;
            s.allowParkour.value       = false;
            s.allowParkourPlace.value  = false;
            s.allowParkourAscend.value = false;
            java.util.List<net.minecraft.item.Item> throwaway = new java.util.ArrayList<>();
            for (Block b : VOID_SCATTER_BLOCKS) {
                net.minecraft.item.Item item = b.asItem();
                if (item != Items.AIR) throwaway.add(item);
            }
            s.acceptableThrowawayItems.value = throwaway;
        });

        // The new world has a connected path; just explore it.
        // Staircases and branches are baked into the world gen — no Y-goal stepping needed.
        while (running.get()) {
            exploreVoid(60);
        }
    }

    private void exploreVoid(int seconds) throws InterruptedException {
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc.player == null) { Thread.sleep(1000); return; }
        mc.execute(() -> baritone().getExploreProcess()
            .explore(mc.player.getBlockX(), mc.player.getBlockZ()));
        for (int i = 0; i < seconds && running.get(); i++) {
            Thread.sleep(1_000);
        }
        cancelBaritone();
        Thread.sleep(300);
    }

    // ── Shared helpers ────────────────────────────────────────────────────────

    private void mine(Block[] blocks, int quantity, int maxSeconds) throws InterruptedException {
        log("§a[Bot] Mine: " + label(blocks[0]) + (blocks.length > 1 ? "+…" : "") + " ×" + quantity);
        MinecraftClient.getInstance().execute(() -> baritone().getMineProcess().mine(quantity, blocks));
        awaitProcess(baritone().getMineProcess(), maxSeconds);
    }

    /**
     * Explore for up to {@code seconds}.
     *
     * Priority order:
     *   1. Lodestone structure hints (positions from the "blockscope:structures" channel)
     *   2. Local interior-block detection (INTERIOR_BLOCKS within 24 blocks)
     *   3. Baritone's explore process (default)
     *
     * This means the bot exploits server-provided structure locations when available,
     * falls back to heuristic detection when Lodestone is absent, and always has a
     * default movement strategy.
     */
    private void explore(int seconds) throws InterruptedException {
        // 1. Lodestone structure hint
        BlockPos hint = structureHints.poll();
        if (hint != null) {
            log("§e[Bot] Structure hint → (" + hint.getX() + ", " + hint.getZ() + ")");
            MinecraftClient.getInstance().execute(() ->
                baritone().getCustomGoalProcess().setGoalAndPath(
                    new GoalXZ(hint.getX(), hint.getZ())));
            awaitProcess(baritone().getCustomGoalProcess(), Math.min(seconds, 120));
            return;
        }

        // 2. Default: Baritone explore, scanning every 10 s for local interior blocks
        log("§b[Bot] Exploring for " + seconds + "s…");
        MinecraftClient mc = MinecraftClient.getInstance();
        mc.execute(() -> baritone().getExploreProcess()
            .explore(mc.player.getBlockX(), mc.player.getBlockZ()));

        for (int i = 0; i < seconds && running.get(); i++) {
            Thread.sleep(1_000);
            if (i % 10 == 9 && hasNearbyInteriorBlocks()) {
                log("§e[Bot] Interior detected — navigating in");
                cancelBaritone();
                Thread.sleep(200);
                mine(INTERIOR_BLOCKS, 8, 120);
                return;
            }
        }
        cancelBaritone();
        Thread.sleep(300);
    }

    /**
     * Wait for a Baritone process, scanning for interior blocks every 10 s.
     * Cancels the current task and digs into the interior if signature blocks are found.
     */
    private void awaitProcess(IBaritoneProcess process, int maxSeconds) throws InterruptedException {
        for (int i = 0; i < maxSeconds * 2 && running.get(); i++) {
            Thread.sleep(500);
            if (!process.isActive()) return;
            if (i % 20 == 19 && hasNearbyInteriorBlocks()) {
                log("§e[Bot] Interior detected — navigating in");
                cancelBaritone();
                Thread.sleep(200);
                mine(INTERIOR_BLOCKS, 8, 120);
                return;
            }
        }
    }

    /**
     * Scan a 24-block radius for interior signature blocks (beds, rails, spawner, etc.).
     * Returns true if any INTERIOR_BLOCKS are found nearby — used to trigger
     * a targeted interior-exploration detour regardless of Lodestone availability.
     */
    private boolean hasNearbyInteriorBlocks() {
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc.player == null || mc.world == null) return false;
        int px = mc.player.getBlockX(), py = mc.player.getBlockY(), pz = mc.player.getBlockZ();
        for (int dx = -24; dx <= 24; dx += 3) {
            for (int dy = -4; dy <= 4; dy++) {
                for (int dz = -24; dz <= 24; dz += 3) {
                    Block b = mc.world.getBlockState(new BlockPos(px+dx, py+dy, pz+dz)).getBlock();
                    for (Block sig : INTERIOR_BLOCKS) if (b == sig) return true;
                }
            }
        }
        return false;
    }

    private void waitForPlayer() throws InterruptedException {
        MinecraftClient mc = MinecraftClient.getInstance();
        while (mc.player == null || mc.world == null) Thread.sleep(500);
    }

    private void cancelBaritone() {
        MinecraftClient.getInstance().execute(() -> {
            try { baritone().getPathingBehavior().cancelEverything(); } catch (Exception ignored) {}
        });
    }

    private static IBaritone baritone() {
        return BaritoneAPI.getProvider().getPrimaryBaritone();
    }

    private static String label(Block b) {
        return b.toString().replace("Block{minecraft:", "").replace("}", "");
    }

    private static void log(String msg) {
        MinecraftClient mc = MinecraftClient.getInstance();
        mc.execute(() -> { if (mc.player != null) mc.player.sendMessage(Text.literal(msg), false); });
    }
}
