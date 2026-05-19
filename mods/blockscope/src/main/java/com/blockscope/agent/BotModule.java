package com.blockscope.agent;

import baritone.api.BaritoneAPI;
import baritone.api.IBaritone;
import baritone.api.pathing.goals.GoalBlock;
import baritone.api.pathing.goals.GoalXZ;
import baritone.api.pathing.goals.GoalYLevel;
import baritone.api.process.IBaritoneProcess;
import net.minecraft.block.Block;
import net.minecraft.block.Blocks;
import net.minecraft.client.MinecraftClient;
import net.minecraft.item.Items;
import net.minecraft.text.Text;
import net.minecraft.util.ActionResult;
import net.minecraft.util.Hand;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Direction;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.LightType;

import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Autonomous data-collection bot.  Two modes:
 *
 *   CREATIVE_SURVEY  — enables creative flight and uses Baritone's explore
 *                      process to sweep the map at ground level.
 *
 *   SURVIVAL_GATHER  — cycles: chop wood → surface ores → explore → deep ores.
 *
 *   VOID_SCATTER     — void world seeded with random known blocks; every 30 s
 *                      Baritone pathfinds to a random scatter block, scaffolding
 *                      with the stacks of blocks in inventory.
 *
 * Toggle with G.  Cycle modes (while stopped) with H.
 * Requires baritone-api-fabric-1.9.5.jar in mods folder.
 */
public class BotModule {

    public enum Mode { CREATIVE_SURVEY, SURVIVAL_GATHER, VOID_SCATTER }

    private static BotModule instance;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private Thread botThread;
    private Mode mode = Mode.CREATIVE_SURVEY;
    private final Random rng = new Random();

    // Smooth look state for creative survey (freeLook=true, we drive yaw/pitch ourselves)
    private double lookYaw      = 0;   // current interpolated yaw  (applied to player)
    private double lookPitch    = 0;   // current interpolated pitch
    private double targetYaw    = 0;   // goal we're lerping toward
    private double targetPitch  = 0;
    private int    lookHoldTicks = 0;  // ticks until next target re-roll
    private boolean lookInitialized = false;
    private boolean useFreeLook = false; // true only in creative survey

    private int torchCheckTicks = 0;

    private BotModule() {}

    public static BotModule getInstance() {
        if (instance == null) instance = new BotModule();
        return instance;
    }

    public boolean isRunning() { return running.get(); }
    public Mode    getMode()   { return mode; }
    public void    setMode(Mode m) { if (!running.get()) mode = m; }

    public void cycleMode() {
        if (running.get()) { log("§c[Bot] Stop the bot before changing mode."); return; }
        mode = (mode == Mode.CREATIVE_SURVEY) ? Mode.SURVIVAL_GATHER : Mode.CREATIVE_SURVEY;
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

        // Seed initial position from actual player look so there's no jump on start
        if (!lookInitialized) {
            lookYaw   = mc.player.getYaw();
            lookPitch = mc.player.getPitch();
            targetYaw   = lookYaw;
            targetPitch = lookPitch;
            lookInitialized = true;
        }

        // Pick a new target periodically
        if (lookHoldTicks <= 0) {
            Vec3d vel = mc.player.getVelocity();
            double hSpeed = Math.sqrt(vel.x * vel.x + vel.z * vel.z);

            // Base yaw on movement direction when moving, else stay near current look
            double baseYaw = hSpeed > 0.05
                ? Math.toDegrees(Math.atan2(-vel.x, vel.z))
                : lookYaw;

            // 15% chance of a big look-around (60-90° sweep), otherwise a small glance
            boolean bigLook = rng.nextFloat() < 0.15f;
            double yawJitter   = bigLook ? (rng.nextDouble() * 2 - 1) * 75 : (rng.nextDouble() * 2 - 1) * 22;
            double pitchTarget = bigLook ? (rng.nextDouble() * 55 - 20) : (rng.nextDouble() * 18 - 5);

            targetYaw   = baseYaw + yawJitter;
            targetPitch = Math.max(-75, Math.min(60, pitchTarget));

            // Hold the new target for 1.5–5 s
            lookHoldTicks = 30 + rng.nextInt(70);
        }
        lookHoldTicks--;

        // Lerp at 6%/tick — smooth enough that a 90° turn takes ~2 s
        lookYaw   += (targetYaw   - lookYaw)   * 0.06;
        lookPitch += (targetPitch - lookPitch)  * 0.06;

        mc.player.setYaw((float) lookYaw);
        mc.player.setPitch((float) lookPitch);
    }

    // ── Torch placement ───────────────────────────────────────────────────────

    /** Called every tick; fires the actual check only every 100 ticks (5 s). */
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

        // Prefer wall to left / right (less likely to be in Baritone's path), floor last
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
        // Floor fallback
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

    // ── Global Baritone settings (applied on start) ───────────────────────────

    private void applyGlobalSettings() {
        MinecraftClient.getInstance().execute(() -> {
            var s = BaritoneAPI.getSettings();

            // Rendering — all off so nothing overlays the video
            s.renderPath.value              = false;
            s.renderGoal.value              = false;
            s.renderGoalAnimated.value      = false;
            s.renderGoalXZBeacon.value      = false;
            s.renderSelection.value         = false;
            s.renderSelectionBoxes.value    = false;
            s.renderCachedChunks.value      = false;
            s.fadePath.value                = false;

            // Look — smooth Baritone turns for survival; creative overrides with freeLook+onTick
            s.smoothLook.value              = true;
            s.smoothLookTicks.value         = 18;   // ~0.9 s to complete a turn
            s.randomLooking.value           = 0.18; // small random jitter per tick
            s.randomLooking113.value        = 0.10;
            s.freeLook.value                = false; // creative survey overrides this to true

            // Doors: Baritone naturally opens doors when pathfinding through them.

            // Don't warp into nether portals during automated runs
            s.enterPortal.value             = false;

            // Movement
            s.allowSprint.value             = true;
            s.sprintAscends.value           = true;
            s.notificationOnPathComplete.value = false; // suppress chat spam
        });
    }

    // ── Creative survey ───────────────────────────────────────────────────────

    private void runCreativeSurvey() throws InterruptedException {
        waitForPlayer();

        // Enable creative flight so Baritone uses 3D pathfinding instead of walking
        MinecraftClient mc = MinecraftClient.getInstance();
        mc.execute(() -> {
            if (mc.player != null) {
                mc.player.getAbilities().flying = true;
                mc.player.sendAbilitiesUpdate();
                BaritoneAPI.getSettings().allowPlace.value  = false;
                BaritoneAPI.getSettings().allowBreak.value  = true;  // break anything in flight path
                BaritoneAPI.getSettings().freeLook.value    = true;  // we drive look in onTick
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
    // Village interior proxy — breaking these forces the bot into house interiors
    private static final Block[] VILLAGE_BLOCKS = {
        Blocks.WHITE_BED, Blocks.ORANGE_BED, Blocks.MAGENTA_BED, Blocks.LIGHT_BLUE_BED,
        Blocks.YELLOW_BED, Blocks.LIME_BED, Blocks.PINK_BED, Blocks.GRAY_BED,
        Blocks.LIGHT_GRAY_BED, Blocks.CYAN_BED, Blocks.PURPLE_BED, Blocks.BLUE_BED,
        Blocks.BROWN_BED, Blocks.GREEN_BED, Blocks.RED_BED, Blocks.BLACK_BED,
        Blocks.BOOKSHELF,
        Blocks.OAK_PRESSURE_PLATE, Blocks.STONE_PRESSURE_PLATE,
        Blocks.SPRUCE_PRESSURE_PLATE, Blocks.BIRCH_PRESSURE_PLATE,
    };

    // No deepslate in the beta datapack — standard ore blocks only
    private static final Block[] DEEP_ORES = {
        Blocks.IRON_ORE, Blocks.GOLD_ORE, Blocks.REDSTONE_ORE,
        Blocks.LAPIS_ORE, Blocks.DIAMOND_ORE,
    };

    // ── Void scatter ──────────────────────────────────────────────────────────

    // Mirror of VoidScatterGenerator.SCATTER_MATERIALS (Java server) and
    // block_vocab.py :: SCATTER_BLOCKS (Python pipeline).
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
        // Baritone resets its settings on new server connection — wait for that to fire first
        Thread.sleep(3000);
        MinecraftClient.getInstance().execute(() -> {
            var s = BaritoneAPI.getSettings();
            s.allowBreak.value         = true;
            s.allowPlace.value         = true;
            s.allowParkour.value       = false;
            s.allowParkourPlace.value  = false;
            s.allowParkourAscend.value = false;
            // Let Baritone scaffold with any block in the void scatter inventory
            java.util.List<net.minecraft.item.Item> throwaway = new java.util.ArrayList<>();
            for (Block b : VOID_SCATTER_BLOCKS) {
                net.minecraft.item.Item item = b.asItem();
                if (item != net.minecraft.item.Items.AIR) throwaway.add(item);
            }
            s.acceptableThrowawayItems.value = throwaway;
        });

        int step = 0;
        while (running.get()) {
            if (step % 5 == 4) {
                MinecraftClient mc = MinecraftClient.getInstance();
                if (mc.player != null) {
                    int y = 10 + rng.nextInt(100);
                    log("§d[Bot] Void ↕ y=" + y);
                    mc.execute(() -> baritone().getCustomGoalProcess()
                        .setGoalAndPath(new GoalYLevel(y)));
                    awaitProcess(baritone().getCustomGoalProcess(), 20);
                    // Brief pause after Y goal (success or failure) before next phase
                    Thread.sleep(5_000);
                }
            } else {
                exploreVoid(40);
            }
            step++;
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

            // Long surface sweep — loads new terrain, finds villages
            explore(150);
            if (!running.get()) break;

            mine(WOOD, 24, 180);
            if (!running.get()) break;

            mine(SURFACE_ORES, 16, 180);
            if (!running.get()) break;

            // Descend underground
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

    /** Path to a target Y level (descend or ascend). Baritone will find cave passages or dig. */
    private void descend(int targetY, int maxSeconds) throws InterruptedException {
        MinecraftClient.getInstance().execute(() ->
            baritone().getCustomGoalProcess().setGoalAndPath(new GoalYLevel(targetY)));
        awaitProcess(baritone().getCustomGoalProcess(), maxSeconds);
    }

    // ── Shared helpers ────────────────────────────────────────────────────────

    private void mine(Block[] blocks, int quantity, int maxSeconds) throws InterruptedException {
        log("§a[Bot] Mine: " + label(blocks[0]) + (blocks.length > 1 ? "+…" : "") + " ×" + quantity);
        MinecraftClient.getInstance().execute(() -> baritone().getMineProcess().mine(quantity, blocks));
        awaitProcess(baritone().getMineProcess(), maxSeconds);
    }

    private void explore(int seconds) throws InterruptedException {
        log("§b[Bot] Exploring for " + seconds + "s…");
        MinecraftClient mc = MinecraftClient.getInstance();
        mc.execute(() -> baritone().getExploreProcess()
            .explore(mc.player.getBlockX(), mc.player.getBlockZ()));
        for (int i = 0; i < seconds && running.get(); i++) {
            Thread.sleep(1_000);
            if (i % 10 == 9 && hasNearbyVillageBlocks()) {
                log("§e[Bot] Village interior detected — breaking in");
                cancelBaritone();
                Thread.sleep(200);
                mine(VILLAGE_BLOCKS, 8, 120);
                return;
            }
        }
        cancelBaritone();
        Thread.sleep(300);
    }

    /**
     * Wait for a Baritone process to finish, but every 10 seconds scan for nearby
     * village interior blocks (beds, bookshelves, pressure plates). If found,
     * cancel the current task and mine them — forcing the bot inside buildings —
     * then hand control back to the caller.
     */
    private void awaitProcess(IBaritoneProcess process, int maxSeconds) throws InterruptedException {
        for (int i = 0; i < maxSeconds * 2 && running.get(); i++) {
            Thread.sleep(500);
            if (!process.isActive()) return;
            // Every 10 s check for village interior blocks within 24 blocks
            if (i % 20 == 19 && hasNearbyVillageBlocks()) {
                log("§e[Bot] Village interior detected — breaking in");
                cancelBaritone();
                Thread.sleep(200);
                mine(VILLAGE_BLOCKS, 8, 120);
                return; // caller's phase continues on next iteration
            }
        }
    }

    /** Scan a 24-block radius for any VILLAGE_BLOCKS (beds, bookshelves, pressure plates). */
    private boolean hasNearbyVillageBlocks() {
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc.player == null || mc.world == null) return false;
        int px = mc.player.getBlockX(), py = mc.player.getBlockY(), pz = mc.player.getBlockZ();
        for (int dx = -24; dx <= 24; dx += 3) {
            for (int dy = -4; dy <= 4; dy++) {
                for (int dz = -24; dz <= 24; dz += 3) {
                    Block b = mc.world.getBlockState(new BlockPos(px+dx, py+dy, pz+dz)).getBlock();
                    for (Block vb : VILLAGE_BLOCKS) if (b == vb) return true;
                }
            }
        }
        return false;
    }

    private void sleepSeconds(int seconds) throws InterruptedException {
        for (int i = 0; i < seconds && running.get(); i++) Thread.sleep(1_000);
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
