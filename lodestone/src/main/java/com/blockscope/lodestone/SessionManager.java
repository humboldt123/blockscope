package com.blockscope.lodestone;

import org.bukkit.*;
import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.entity.Player;
import org.bukkit.generator.structure.Structure;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.inventory.ItemStack;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Assigns each joining player a private world with the beta worldgen datapack.
 * Maintains a preload pool so the world is ready instantly on join.
 * On join: teleports player, sets gamemode, gives tools (survival), sends session_start.
 * On quit/timeout: unloads and deletes the world.
 */
public class SessionManager implements Listener {

    private static final String SESSION_CHANNEL = "blockscope:session_start";
    private static final String[] TOOL_TIERS = {"gold", "wood", "stone", "iron", "diamond"};
    private static final String[] TOOL_TYPES = {"pickaxe", "axe", "shovel", "sword"};

    private final LodestonePlugin plugin;
    private final int poolSize;
    private final int sessionDurationSeconds;
    private final boolean toolRandomisation;
    private final int toolCheckIntervalTicks;
    private final boolean survivalEnabled;
    private final boolean voidScatterEnabled;
    private final boolean baritoneControlEnabled;

    // Path to beta_world.zip — resolved relative to plugin data folder
    private File betaDatapackZip;

    private final Queue<World> survivalPool     = new ConcurrentLinkedQueue<>();
    private final Queue<World> voidScatterPool  = new ConcurrentLinkedQueue<>();
    private final Map<UUID, World> playerWorlds = new ConcurrentHashMap<>();
    private final Map<UUID, Integer> toolCheckTasks = new ConcurrentHashMap<>();
    private final Map<UUID, Integer> sessionTimers = new ConcurrentHashMap<>();

    private final AtomicInteger worldCounter = new AtomicInteger(0);
    private final Random rng = new Random();
    private final ExecutorService bgPool = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "lodestone-bg");
        t.setDaemon(true);
        return t;
    });

    private int modeIndex = 0;

    public SessionManager(LodestonePlugin plugin) {
        this.plugin = plugin;
        FileConfiguration cfg = plugin.getConfig();
        this.poolSize               = cfg.getInt("pool_size", 1);
        this.sessionDurationSeconds = cfg.getInt("session_duration_seconds", 1800);
        this.toolRandomisation      = cfg.getBoolean("tool_randomisation_enabled", true);
        this.toolCheckIntervalTicks = cfg.getInt("tool_check_interval_seconds", 30) * 20;
        this.survivalEnabled        = cfg.getBoolean("survival_sessions_enabled", true);
        this.voidScatterEnabled     = cfg.getBoolean("void_scatter_sessions_enabled", true);
        this.baritoneControlEnabled = cfg.getBoolean("baritone_control_enabled", true);

        // Look for beta_world.zip next to the plugin jar (in plugins/ folder)
        betaDatapackZip = new File(plugin.getDataFolder().getParentFile(), "beta_world.zip");
        if (!betaDatapackZip.exists()) {
            // Also check server root
            betaDatapackZip = new File(Bukkit.getWorldContainer().getParentFile(), "beta_world.zip");
        }
        if (!betaDatapackZip.exists()) {
            plugin.getLogger().warning("beta_world.zip not found — worlds will generate without beta datapack!");
            betaDatapackZip = null;
        } else {
            plugin.getLogger().info("Beta datapack found: " + betaDatapackZip.getAbsolutePath());
        }
    }

    // ── World pool ─────────────────────────────────────────────────────────────

    public void startWorldPool() {
        plugin.getServer().getMessenger().registerOutgoingPluginChannel(plugin, SESSION_CHANNEL);
        ensureDatapackInMainWorld();
        refillPool();
    }

    /**
     * Paper uses the main world's datapack repository for all programmatically-created worlds.
     * The per-world datapacks/ folders of custom WorldCreator worlds are NOT read.
     * So we copy beta_world.zip into world/datapacks/ — if it isn't there already this
     * server run started without it and we log a warning (restart required to take effect).
     */
    private void ensureDatapackInMainWorld() {
        if (betaDatapackZip == null) return;
        World mainWorld = Bukkit.getWorlds().get(0);
        File mainDatapacks = new File(mainWorld.getWorldFolder(), "datapacks");
        mainDatapacks.mkdirs();
        File dest = new File(mainDatapacks, "beta_world.zip");
        if (!dest.exists()) {
            try {
                Files.copy(betaDatapackZip.toPath(), dest.toPath());
                plugin.getLogger().warning("beta_world.zip copied to world/datapacks/ — RESTART server for it to take effect!");
            } catch (IOException e) {
                plugin.getLogger().severe("Failed to copy beta datapack to main world: " + e.getMessage());
            }
        } else {
            plugin.getLogger().info("Beta datapack confirmed in world/datapacks/ — active.");
        }
    }

    private void refillPool() {
        bgPool.submit(() -> {
            if (survivalEnabled) {
                while (survivalPool.size() < poolSize) {
                    World w = generateWorld();
                    if (w != null) {
                        survivalPool.add(w);
                        plugin.getLogger().info("World pool: survival world ready (" + w.getName() + "), pool size=" + survivalPool.size());
                    }
                }
            }
            if (voidScatterEnabled) {
                while (voidScatterPool.size() < poolSize) {
                    World w = generateVoidWorldFromBg();
                    if (w != null) {
                        voidScatterPool.add(w);
                        plugin.getLogger().info("World pool: void scatter world ready (" + w.getName() + "), pool size=" + voidScatterPool.size());
                    }
                }
            }
        });
    }

    private World generateWorld() {
        String name = "bs_" + worldCounter.incrementAndGet() + "_" + Long.toHexString(rng.nextLong() & 0xFFFFFFL);
        long seed = rng.nextLong();

        WorldCreator creator = new WorldCreator(name)
            .environment(World.Environment.NORMAL)
            .seed(seed)
            .generateStructures(true);

        CompletableFuture<World> future = new CompletableFuture<>();
        Bukkit.getScheduler().runTask(plugin, () -> {
            try {
                World w = creator.createWorld();
                if (w != null) {
                    w.setDifficulty(Difficulty.PEACEFUL);
                    w.setGameRule(GameRule.DO_MOB_SPAWNING, false);
                    w.setGameRule(GameRule.DO_DAYLIGHT_CYCLE, true);
                    w.setGameRule(GameRule.ANNOUNCE_ADVANCEMENTS, false);
                    w.setGameRule(GameRule.SHOW_DEATH_MESSAGES, false);
                    w.setGameRule(GameRule.DO_IMMEDIATE_RESPAWN, true);
                    // Pre-load a small area around spawn
                    Location spawn = w.getSpawnLocation();
                    for (int dx = -1; dx <= 1; dx++) for (int dz = -1; dz <= 1; dz++) {
                        w.getChunkAt(spawn.getBlockX() / 16 + dx, spawn.getBlockZ() / 16 + dz).load(true);
                    }
                    // 1/5 chance: relocate spawn to a village; 1/5: other structure; 3/5: leave it
                    targetSpawn(w);
                }
                future.complete(w);
            } catch (Exception e) {
                plugin.getLogger().severe("World creation failed: " + e.getMessage());
                future.complete(null);
            }
        });

        try {
            return future.get(120, TimeUnit.SECONDS);
        } catch (Exception e) {
            plugin.getLogger().warning("World generation timed out: " + e.getMessage());
            return null;
        }
    }

    /**
     * Bias spawn point:
     *   2/10 → old village (old_villages:village_plains)
     *   5/10 → special structure, weighted heavily toward herobrine + shrines
     *           (beta_world:shrines randomly picks diamondpillar or bedrock from its pool)
     *   3/10 → default vanilla spawn
     */
    private void targetSpawn(World world) {
        int roll = rng.nextInt(10);
        if (roll < 2) {
            locateAndSetSpawn(world, "old_villages:village_plains");
        } else if (roll < 7) {
            // herobrine×3, shrines×3 (pool picks diamondpillar or bedrock), ruins×1, orewall×1
            String[] weighted = {
                "beta_world:herobrine", "beta_world:herobrine", "beta_world:herobrine",
                "beta_world:shrines",   "beta_world:shrines",   "beta_world:shrines",
                "beta_world:ruins",
                "beta_world:orewall",
            };
            locateAndSetSpawn(world, weighted[rng.nextInt(weighted.length)]);
        }
        // else 3/10: leave vanilla spawn as-is
    }

    private void locateAndSetSpawn(World world, String structureKey) {
        try {
            NamespacedKey key = NamespacedKey.fromString(structureKey);
            if (key == null) { plugin.getLogger().warning("Bad structure key: " + structureKey); return; }

            Structure structure = Registry.STRUCTURE.get(key);
            if (structure == null) {
                StringBuilder sample = new StringBuilder("Structure not in registry: " + structureKey + " — known: ");
                int n = 0;
                for (Structure s : Registry.STRUCTURE) {
                    if (n++ > 0) sample.append(", ");
                    sample.append(s.getKey());
                    if (n >= 5) { sample.append("…"); break; }
                }
                plugin.getLogger().warning(sample.toString());
                return;
            }

            Location origin = world.getSpawnLocation();
            plugin.getLogger().info(world.getName() + ": searching for " + structureKey + " within 300 chunks...");
            var result = world.locateNearestStructure(origin, structure, 300, false);
            if (result != null) {
                Location found = result.getLocation();
                found.setY(world.getHighestBlockYAt(found.getBlockX(), found.getBlockZ()) + 1);
                world.setSpawnLocation(found);
                plugin.getLogger().info(world.getName() + ": spawn → " + structureKey +
                    " @ " + found.getBlockX() + "," + found.getBlockZ());
            } else {
                plugin.getLogger().info(world.getName() + ": " + structureKey + " not found within 300 chunks, keeping vanilla spawn");
            }
        } catch (Exception e) {
            plugin.getLogger().warning("locateAndSetSpawn(" + structureKey + ") threw: " + e);
        }
    }

    // ── Player join / quit ─────────────────────────────────────────────────────

    @EventHandler
    public void onPlayerJoin(PlayerJoinEvent event) {
        Player player = event.getPlayer();
        event.joinMessage(null);

        String modeStr = assignMode();
        boolean isVoidScatter = "void_scatter".equals(modeStr);

        Queue<World> pool = isVoidScatter ? voidScatterPool : survivalPool;
        World world = pool.poll();
        if (world == null) {
            plugin.getLogger().warning("World pool empty — generating on-demand for " + player.getName());
            world = isVoidScatter ? generateVoidWorldSync() : generateWorldSync();
        }
        if (world == null) {
            player.kick(net.kyori.adventure.text.Component.text("§cFailed to create session world. Try again in a moment."));
            refillPool();
            return;
        }

        playerWorlds.put(player.getUniqueId(), world);
        refillPool();

        World finalWorld = world;

        Bukkit.getScheduler().runTask(plugin, () -> {
            player.getInventory().clear();
            player.setGameMode(GameMode.SURVIVAL);
            player.teleport(finalWorld.getSpawnLocation());

            if (isVoidScatter) {
                giveScatterInventory(player);
            } else if (toolRandomisation) {
                giveRandomTools(player);
                scheduleToolCheck(player);
            }

            plugin.getLogger().info("Session: " + player.getName() +
                " → " + finalWorld.getName() + " [" + modeStr + "]");

            // Delay session_start by 40 ticks so the client finishes loading the new
            // world before we send the packet (teleport causes a dimension-change/respawn
            // packet sequence; plugin messages sent mid-switch are silently dropped).
            Bukkit.getScheduler().runTaskLater(plugin, () -> {
                if (player.isOnline()) {
                    sendSessionStart(player, modeStr);
                    plugin.getLogger().info("Sent session_start → " + player.getName() + " [" + modeStr + "]");
                }
            }, 40L);
        });

        // Session timer — kick after duration so client auto-reconnects
        int timerId = Bukkit.getScheduler().scheduleSyncDelayedTask(plugin,
            () -> {
                if (player.isOnline()) {
                    endSession(player, "timeout");
                }
            },
            (long) sessionDurationSeconds * 20L);
        sessionTimers.put(player.getUniqueId(), timerId);
    }

    @EventHandler
    public void onPlayerQuit(PlayerQuitEvent event) {
        event.quitMessage(null);
        endSession(event.getPlayer(), "disconnect");
    }

    private void endSession(Player player, String reason) {
        UUID uuid = player.getUniqueId();
        cancelToolCheck(uuid);
        Integer timerId = sessionTimers.remove(uuid);
        if (timerId != null) Bukkit.getScheduler().cancelTask(timerId);
        // Guard: only handle once (disconnect can fire while timeout kick is processing)
        if (playerWorlds.get(uuid) == null) return;

        World world = playerWorlds.remove(uuid);
        if (world == null) return;

        plugin.getLogger().info("Session ended (" + reason + "): " + player.getName() + " / " + world.getName());

        if ("timeout".equals(reason) && player.isOnline()) {
            player.kick(net.kyori.adventure.text.Component.text("§aSession complete — reconnecting…"));
        }

        // Unload after 2 ticks to ensure the player has left the world,
        // but don't hold the reference any longer — worlds must not pile up.
        Bukkit.getScheduler().runTaskLater(plugin, () -> unloadAndDeleteWorld(world), 2L);
    }

    private void unloadAndDeleteWorld(World world) {
        World fallback = Bukkit.getWorlds().stream()
            .filter(w -> !w.getName().startsWith("bs_"))
            .findFirst().orElse(Bukkit.getWorlds().get(0));
        world.getPlayers().forEach(p -> p.teleport(fallback.getSpawnLocation()));
        Bukkit.unloadWorld(world, false);
        bgPool.submit(() -> deleteDir(world.getWorldFolder()));
    }

    // ── Tool randomisation ─────────────────────────────────────────────────────

    private void giveRandomTools(Player player) {
        String tier = TOOL_TIERS[rng.nextInt(TOOL_TIERS.length)];
        for (String type : TOOL_TYPES) {
            Material mat = Material.matchMaterial("minecraft:" + tier + "_" + type);
            if (mat != null) player.getInventory().addItem(new ItemStack(mat));
        }
        player.getInventory().addItem(new ItemStack(Material.TORCH, 16));
        player.getInventory().addItem(new ItemStack(Material.BREAD, 16));
        plugin.getLogger().fine("Tools: " + tier + " tier → " + player.getName());
    }

    private void scheduleToolCheck(Player player) {
        int taskId = Bukkit.getScheduler().scheduleSyncRepeatingTask(plugin, () -> {
            if (!player.isOnline()) return;
            boolean hasPickaxe = false;
            for (ItemStack item : player.getInventory().getContents()) {
                if (item != null && item.getType().name().endsWith("_PICKAXE")) { hasPickaxe = true; break; }
            }
            if (!hasPickaxe) giveRandomTools(player);
        }, toolCheckIntervalTicks, toolCheckIntervalTicks);
        toolCheckTasks.put(player.getUniqueId(), taskId);
    }

    private void cancelToolCheck(UUID uuid) {
        Integer taskId = toolCheckTasks.remove(uuid);
        if (taskId != null) Bukkit.getScheduler().cancelTask(taskId);
    }

    // ── Session start packet ───────────────────────────────────────────────────

    private void sendSessionStart(Player player, String mode) {
        byte[] modeBytes = mode.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buf = ByteBuffer.allocate(1 + modeBytes.length + 4);
        buf.put((byte) modeBytes.length);
        buf.put(modeBytes);
        buf.putInt(sessionDurationSeconds);
        player.sendPluginMessage(plugin, SESSION_CHANNEL, buf.array());
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private String assignMode() {
        java.util.List<String> modes = new java.util.ArrayList<>();
        if (survivalEnabled)    modes.add("survival");
        if (voidScatterEnabled) modes.add("void_scatter");
        if (modes.isEmpty())    return "survival";
        return modes.get((modeIndex++) % modes.size());
    }

    /** Called from bgPool thread — dispatches createWorld() to main thread then waits. */
    private World generateVoidWorldFromBg() {
        String name = "bs_" + worldCounter.incrementAndGet() + "_" + Long.toHexString(rng.nextLong() & 0xFFFFFFL);
        CompletableFuture<World> future = new CompletableFuture<>();
        Bukkit.getScheduler().runTask(plugin, () -> {
            try {
                World w = new WorldCreator(name)
                    .environment(World.Environment.NORMAL)
                    .generator(new VoidScatterGenerator())
                    .generateStructures(false)
                    .createWorld();
                if (w != null) applyVoidWorldRules(w);
                future.complete(w);
            } catch (Exception e) {
                plugin.getLogger().severe("Void world creation failed: " + e.getMessage());
                future.complete(null);
            }
        });
        try { return future.get(60, TimeUnit.SECONDS); } catch (Exception e) { return null; }
    }

    /** Called from main thread (on-demand fallback during onPlayerJoin). */
    private World generateVoidWorldSync() {
        String name = "bs_" + worldCounter.incrementAndGet() + "_" + Long.toHexString(rng.nextLong() & 0xFFFFFFL);
        try {
            World w = new WorldCreator(name)
                .environment(World.Environment.NORMAL)
                .generator(new VoidScatterGenerator())
                .generateStructures(false)
                .createWorld();
            if (w != null) applyVoidWorldRules(w);
            return w;
        } catch (Exception e) {
            plugin.getLogger().severe("Void world creation failed: " + e.getMessage());
            return null;
        }
    }

    private void applyVoidWorldRules(World w) {
        w.setDifficulty(Difficulty.PEACEFUL);
        w.setGameRule(GameRule.DO_MOB_SPAWNING, false);
        w.setGameRule(GameRule.DO_DAYLIGHT_CYCLE, true);
        w.setGameRule(GameRule.ANNOUNCE_ADVANCEMENTS, false);
        w.setGameRule(GameRule.SHOW_DEATH_MESSAGES, false);
        w.setGameRule(GameRule.DO_IMMEDIATE_RESPAWN, true);
    }

    private void giveScatterInventory(Player player) {
        Material[] pool = VoidScatterGenerator.SCATTER_MATERIALS;
        // Fill hotbar (9 slots) with random distinct block types, 64 each
        java.util.List<Material> chosen = new java.util.ArrayList<>();
        java.util.List<Material> shuffled = new java.util.ArrayList<>(java.util.Arrays.asList(pool));
        java.util.Collections.shuffle(shuffled, rng);
        for (Material m : shuffled) {
            if (chosen.size() >= 9) break;
            chosen.add(m);
        }
        for (int i = 0; i < chosen.size(); i++) {
            player.getInventory().setItem(i, new ItemStack(chosen.get(i), 64));
        }
    }

    private World generateWorldSync() {
        // on-demand fallback for survival — runs generateWorld() on bgPool (it uses runTask internally)
        CompletableFuture<World> f = new CompletableFuture<>();
        bgPool.submit(() -> f.complete(generateWorld()));
        try { return f.get(120, TimeUnit.SECONDS); } catch (Exception e) { return null; }
    }

    private void deleteDir(File dir) {
        if (dir == null || !dir.exists()) return;
        File[] files = dir.listFiles();
        if (files != null) for (File f : files) {
            if (f.isDirectory()) deleteDir(f);
            else f.delete();
        }
        dir.delete();
    }

    public void shutdown() {
        plugin.getServer().getMessenger().unregisterOutgoingPluginChannel(plugin, SESSION_CHANNEL);
        bgPool.shutdownNow();
        for (World w : survivalPool)    unloadAndDeleteWorld(w);
        for (World w : voidScatterPool) unloadAndDeleteWorld(w);
        survivalPool.clear();
        voidScatterPool.clear();
    }
}
