package com.blockscope.lodestone;

import org.bukkit.*;
import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.inventory.ItemStack;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Assigns each joining player a private world and a session mode (CREATIVE or SURVIVAL).
 * Manages a preload pool so world assignment is instant on join.
 * Tears down the world on disconnect.
 * Sends "blockscope:session_start" plugin message to trigger bot + recording on the client.
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
    private final boolean creativeEnabled;
    private final boolean survivalEnabled;
    private final boolean baritoneControlEnabled;

    private final Queue<World> worldPool = new ConcurrentLinkedQueue<>();
    private final Map<UUID, World> playerWorlds = new ConcurrentHashMap<>();
    private final Map<UUID, Integer> toolCheckTasks = new ConcurrentHashMap<>();

    private final AtomicInteger worldCounter = new AtomicInteger(0);
    private final Random rng = new Random();
    private final ExecutorService bgPool = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "lodestone-bg");
        t.setDaemon(true);
        return t;
    });

    private boolean nextCreative = true;

    public SessionManager(LodestonePlugin plugin) {
        this.plugin = plugin;
        FileConfiguration cfg = plugin.getConfig();
        this.poolSize               = cfg.getInt("pool_size", 1);
        this.sessionDurationSeconds = cfg.getInt("session_duration_seconds", 1800);
        this.toolRandomisation      = cfg.getBoolean("tool_randomisation_enabled", true);
        this.toolCheckIntervalTicks = cfg.getInt("tool_check_interval_seconds", 30) * 20;
        this.creativeEnabled        = cfg.getBoolean("creative_sessions_enabled", true);
        this.survivalEnabled        = cfg.getBoolean("survival_sessions_enabled", true);
        this.baritoneControlEnabled = cfg.getBoolean("baritone_control_enabled", true);
    }

    // ── World pool ────────────────────────────────────────────────────────────

    public void startWorldPool() {
        refillPool();
    }

    private void refillPool() {
        bgPool.submit(() -> {
            while (worldPool.size() < poolSize) {
                World w = generateWorld();
                if (w != null) worldPool.add(w);
            }
        });
    }

    private World generateWorld() {
        String name = "bs_world_" + worldCounter.incrementAndGet();
        long seed = rng.nextLong();

        WorldCreator creator = new WorldCreator(name)
            .environment(World.Environment.NORMAL)
            .seed(seed)
            .generateStructures(true);

        CompletableFuture<World> future = new CompletableFuture<>();
        Bukkit.getScheduler().runTask(plugin, () -> {
            World w = creator.createWorld();
            if (w != null) {
                w.setDifficulty(Difficulty.PEACEFUL);
                w.setGameRule(GameRule.DO_MOB_SPAWNING, false);
                w.setGameRule(GameRule.DO_DAYLIGHT_CYCLE, true);
                w.setGameRule(GameRule.ANNOUNCE_ADVANCEMENTS, false);
                w.setGameRule(GameRule.SHOW_DEATH_MESSAGES, false);
                w.getChunkAt(0, 0).load(true);
            }
            future.complete(w);
        });

        try {
            return future.get(60, TimeUnit.SECONDS);
        } catch (Exception e) {
            plugin.getLogger().warning("World generation failed: " + e.getMessage());
            return null;
        }
    }

    // ── Player join / quit ────────────────────────────────────────────────────

    @EventHandler
    public void onPlayerJoin(PlayerJoinEvent event) {
        Player player = event.getPlayer();
        event.joinMessage(null);

        World world = worldPool.poll();
        if (world == null) {
            plugin.getLogger().warning("World pool empty — generating on-demand for " + player.getName());
            world = generateWorldSync();
        }
        if (world == null) {
            player.kick(net.kyori.adventure.text.Component.text("§cFailed to create session world. Try again."));
            return;
        }

        playerWorlds.put(player.getUniqueId(), world);
        refillPool();

        boolean creative = assignCreative();
        String modeStr = creative ? "creative" : "survival";
        World finalWorld = world;

        Bukkit.getScheduler().runTask(plugin, () -> {
            player.teleport(finalWorld.getSpawnLocation());
            player.setGameMode(creative ? GameMode.CREATIVE : GameMode.SURVIVAL);

            if (!creative && toolRandomisation) {
                giveRandomTools(player);
                scheduleToolCheck(player);
            }

            if (baritoneControlEnabled) {
                sendSessionStart(player, modeStr);
            }

            plugin.getLogger().info("Session started: " + player.getName() +
                " → world=" + finalWorld.getName() + " mode=" + modeStr);
        });

        Bukkit.getScheduler().runTaskLater(plugin,
            () -> endSession(player, "timeout"),
            (long) sessionDurationSeconds * 20L);
    }

    @EventHandler
    public void onPlayerQuit(PlayerQuitEvent event) {
        event.quitMessage(null);
        endSession(event.getPlayer(), "disconnect");
    }

    private void endSession(Player player, String reason) {
        UUID uuid = player.getUniqueId();
        cancelToolCheck(uuid);

        World world = playerWorlds.remove(uuid);
        if (world == null) return;

        plugin.getLogger().info("Session ended (" + reason + "): " + player.getName() +
            " world=" + world.getName());

        if ("timeout".equals(reason) && player.isOnline()) {
            player.kick(net.kyori.adventure.text.Component.text("§aSession complete — reconnecting…"));
        }

        Bukkit.getScheduler().runTaskLater(plugin, () -> unloadAndDeleteWorld(world), 20L);
    }

    private void unloadAndDeleteWorld(World world) {
        world.getPlayers().forEach(p -> p.teleport(Bukkit.getWorlds().get(0).getSpawnLocation()));
        Bukkit.unloadWorld(world, false);
        bgPool.submit(() -> deleteDir(world.getWorldFolder()));
    }

    // ── Tool randomisation ────────────────────────────────────────────────────

    private void giveRandomTools(Player player) {
        String tier = TOOL_TIERS[rng.nextInt(TOOL_TIERS.length)];
        player.getInventory().clear();
        for (String type : TOOL_TYPES) {
            Material mat = Material.matchMaterial("minecraft:" + tier + "_" + type);
            if (mat != null) player.getInventory().addItem(new ItemStack(mat));
        }
        player.getInventory().addItem(new ItemStack(Material.TORCH, 16));
        player.getInventory().addItem(new ItemStack(Material.BREAD, 16));
    }

    private void scheduleToolCheck(Player player) {
        int taskId = Bukkit.getScheduler().scheduleSyncRepeatingTask(plugin, () -> {
            if (!player.isOnline()) return;
            ItemStack slot0 = player.getInventory().getItem(0);
            boolean hasPickaxe = slot0 != null && slot0.getType().name().endsWith("_PICKAXE");
            if (!hasPickaxe) giveRandomTools(player);
        }, toolCheckIntervalTicks, toolCheckIntervalTicks);
        toolCheckTasks.put(player.getUniqueId(), taskId);
    }

    private void cancelToolCheck(UUID uuid) {
        Integer taskId = toolCheckTasks.remove(uuid);
        if (taskId != null) Bukkit.getScheduler().cancelTask(taskId);
    }

    // ── Session start packet ──────────────────────────────────────────────────

    private void sendSessionStart(Player player, String mode) {
        byte[] modeBytes = mode.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buf = ByteBuffer.allocate(1 + modeBytes.length + 4);
        buf.put((byte) modeBytes.length);
        buf.put(modeBytes);
        buf.putInt(sessionDurationSeconds);
        player.sendPluginMessage(plugin, SESSION_CHANNEL, buf.array());
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private boolean assignCreative() {
        if (creativeEnabled && !survivalEnabled) return true;
        if (survivalEnabled && !creativeEnabled) return false;
        boolean c = nextCreative;
        nextCreative = !nextCreative;
        return c;
    }

    private World generateWorldSync() {
        CompletableFuture<World> f = new CompletableFuture<>();
        bgPool.submit(() -> f.complete(generateWorld()));
        try { return f.get(90, TimeUnit.SECONDS); } catch (Exception e) { return null; }
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
        bgPool.shutdownNow();
        for (World w : worldPool) unloadAndDeleteWorld(w);
        worldPool.clear();
    }
}
