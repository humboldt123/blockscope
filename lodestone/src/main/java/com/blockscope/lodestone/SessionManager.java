package com.blockscope.lodestone;

import com.blockscope.lodestone.generator.LegacyChunkGenerator;
import org.bukkit.*;
import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.entity.Player;
import org.bukkit.generator.structure.Structure;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.event.world.WorldInitEvent;
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
    private final boolean voidEnabled;
    private final boolean hermitcraftEnabled;
    private final boolean baritoneControlEnabled;
    private final org.bukkit.Material[] voidPalette;
    enum WorldType { MODERN, BETA, LEGACY_2013 }
    private final WorldType worldType;

    // Hermitcraft worlds: persistent shared worlds, never deleted.
    // Maps season number (5-8) -> loaded World (or null if not yet loaded).
    private final java.util.concurrent.ConcurrentHashMap<Integer, World> hermitWorlds
        = new java.util.concurrent.ConcurrentHashMap<>();
    // Spawn points per season, loaded from hermitcraft/spawn_points.json
    private final java.util.Map<Integer, java.util.List<double[]>> hermitSpawnPoints
        = new java.util.HashMap<>();
    private static final int[] HERMIT_SEASONS = {5, 6, 7, 8};

    private final Queue<World> survivalPool = new ConcurrentLinkedQueue<>();
    private final Queue<World> voidPool     = new ConcurrentLinkedQueue<>();
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
        this.survivalEnabled        = cfg.getBoolean("survival_worlds", true);
        this.voidEnabled            = cfg.getBoolean("void_worlds", true);
        this.hermitcraftEnabled     = cfg.getBoolean("hermitcraft_worlds", false);
        this.baritoneControlEnabled = cfg.getBoolean("baritone_control_enabled", true);
        this.voidPalette = VoidScatterGenerator.forConfig(
            cfg.getString("void_scatter_palette", "scatter_classic"));
        this.worldType = switch (cfg.getString("world_type", "modern").toLowerCase()) {
            case "beta"        -> WorldType.BETA;
            case "2013", "legacy_2013" -> WorldType.LEGACY_2013;
            default            -> WorldType.MODERN;
        };

    }

    // ── World pool ─────────────────────────────────────────────────────────────

    public void startWorldPool() {
        plugin.getServer().getMessenger().registerOutgoingPluginChannel(plugin, SESSION_CHANNEL);
        ensureDatapackInMainWorld();
        if (hermitcraftEnabled) {
            loadHermitSpawnPoints();
            bgPool.submit(this::loadHermitcraftWorlds);
        }
        refillPool();
    }

    private void loadHermitSpawnPoints() {
        java.io.File spawnFile = new java.io.File("hermitcraft/spawn_points.json");
        if (!spawnFile.exists()) {
            plugin.getLogger().warning("Hermitcraft spawn_points.json not found at " + spawnFile.getAbsolutePath());
            return;
        }
        try (java.io.FileReader r = new java.io.FileReader(spawnFile)) {
            // Parse JSON manually (no Gson dependency — use Bukkit's bundled approach)
            org.bukkit.configuration.file.YamlConfiguration dummy; // just for context
            String json = new String(java.nio.file.Files.readAllBytes(spawnFile.toPath()));
            // Simple parse: {"5": [[x,y,z],...], "6": [...], ...}
            for (int s : HERMIT_SEASONS) {
                java.util.List<double[]> pts = new java.util.ArrayList<>();
                String marker = "\"" + s + "\":[";
                int start = json.indexOf(marker);
                if (start < 0) continue;
                start += marker.length();
                int depth = 1;
                StringBuilder buf = new StringBuilder("[");
                for (int i = start; i < json.length() && depth > 0; i++) {
                    char c = json.charAt(i);
                    if (c == '[') depth++;
                    else if (c == ']') { depth--; if (depth == 0) break; }
                    buf.append(c);
                }
                buf.append("]");
                // Parse [[x,y,z],...] from buf
                String list = buf.toString().replaceAll("\\s","");
                for (String triple : list.substring(2, list.length()-2).split("\\],\\[")) {
                    String[] parts = triple.split(",");
                    if (parts.length == 3)
                        pts.add(new double[]{Double.parseDouble(parts[0]),
                                             Double.parseDouble(parts[1]),
                                             Double.parseDouble(parts[2])});
                }
                hermitSpawnPoints.put(s, pts);
                plugin.getLogger().info("Hermitcraft S" + s + ": " + pts.size() + " spawn points loaded");
            }
        } catch (Exception e) {
            plugin.getLogger().severe("Failed to load hermitcraft spawn points: " + e.getMessage());
        }
    }

    private void loadHermitcraftWorlds() {
        java.io.File hermitBase = new java.io.File("hermitcraft");
        for (int s : HERMIT_SEASONS) {
            java.io.File worldDir = new java.io.File(hermitBase, "hermitcraft" + s);
            if (!worldDir.exists()) {
                plugin.getLogger().warning("Hermitcraft S" + s + " world not found at " + worldDir);
                continue;
            }
            // Symlink the world into the server's world container so Paper can load it
            // without copying 3+ GB. Remove session.lock first.
            java.io.File dest = new java.io.File("hc" + s);
            if (!dest.exists()) {
                try {
                    java.nio.file.Files.createSymbolicLink(dest.toPath(), worldDir.toPath().toAbsolutePath());
                    plugin.getLogger().info("Symlinked hermitcraft S" + s + " → " + dest);
                } catch (Exception e) {
                    plugin.getLogger().severe("Failed to symlink hc" + s + ": " + e.getMessage());
                    continue;
                }
            }
            new java.io.File(dest, "session.lock").delete();
            CompletableFuture<World> f = new CompletableFuture<>();
            Bukkit.getScheduler().runTask(plugin, () -> {
                try {
                    World w = new WorldCreator("hc" + s)
                        .environment(World.Environment.NORMAL)
                        .generateStructures(false)
                        .createWorld();
                    if (w != null) {
                        w.setGameRule(GameRule.DO_MOB_SPAWNING, false);
                        w.setGameRule(GameRule.DO_DAYLIGHT_CYCLE, true);
                        w.setGameRule(GameRule.ANNOUNCE_ADVANCEMENTS, false);
                        w.setDifficulty(Difficulty.PEACEFUL);
                        hermitWorlds.put(s, w);
                        plugin.getLogger().info("Hermitcraft S" + s + " loaded (" + w.getName() + ")");
                    }
                    f.complete(w);
                } catch (Exception e) {
                    plugin.getLogger().severe("Failed to load hc" + s + ": " + e.getMessage());
                    f.complete(null);
                }
            });
            try { f.get(120, TimeUnit.SECONDS); } catch (Exception ignored) {}
        }
    }

    private void copyDir(java.io.File src, java.io.File dst) throws java.io.IOException {
        dst.mkdirs();
        for (java.io.File f : src.listFiles()) {
            java.io.File d = new java.io.File(dst, f.getName());
            if (f.isDirectory()) copyDir(f, d);
            else java.nio.file.Files.copy(f.toPath(), d.toPath(),
                     java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /**
     * Activate only the period-accurate datapack for the enabled session types.
     * Paper datapacks are server-wide so only one legacy pack should be active at a time.
     *
     * Pack files sit in world/datapacks/:
     *   period_accurate_beta_1_7_3.zip       — active name (Paper loads it)
     *   period_accurate_beta_1_7_3.zip.off   — disabled name (Paper ignores it)
     *   period_accurate_1_5_2.zip            — same pattern
     *
     * Changes take effect on next server restart (datapacks load at startup).
     */
    private void ensureDatapackInMainWorld() {
        World mainWorld = Bukkit.getWorlds().get(0);
        File dp = new File(mainWorld.getWorldFolder(), "datapacks");
        dp.mkdirs();

        File betaOn  = new File(dp, "period_accurate_beta_1_7_3.zip");
        File betaOff = new File(dp, "period_accurate_beta_1_7_3.zip.off");
        File legOn   = new File(dp, "period_accurate_1_5_2.zip");
        File legOff  = new File(dp, "period_accurate_1_5_2.zip.off");

        // Activate exactly the pack matching world_type; disable the other
        if (worldType == WorldType.BETA) {
            if (!betaOn.exists() && betaOff.exists()) betaOff.renameTo(betaOn);
            if (legOn.exists()) legOn.renameTo(legOff);
        } else if (worldType == WorldType.LEGACY_2013) {
            if (!legOn.exists() && legOff.exists()) legOff.renameTo(legOn);
            if (betaOn.exists()) betaOn.renameTo(betaOff);
        } else {
            // MODERN: disable both period packs
            if (betaOn.exists()) betaOn.renameTo(betaOff);
            if (legOn.exists()) legOn.renameTo(legOff);
        }

        // Log current state
        plugin.getLogger().info("Datapacks: beta=" + betaOn.exists() + " legacy2013=" + legOn.exists() + " (world_type=" + worldType + ")");
        if (worldType != WorldType.MODERN) {
            plugin.getLogger().info("Note: datapack changes take effect after a full server restart.");
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
            if (voidEnabled) {
                while (voidPool.size() < poolSize) {
                    World w = generateVoidWorldFromBg();
                    if (w != null) {
                        voidPool.add(w);
                        plugin.getLogger().info("World pool: void world ready (" + w.getName() + ") [" + worldType + "], pool size=" + voidPool.size());
                    }
                }
            }
        });
    }

    private World generateWorld() {
        String name = "bs_" + worldCounter.incrementAndGet() + "_" + Long.toHexString(rng.nextLong() & 0xFFFFFFL);
        long seed = rng.nextLong();

        // Always use vanilla 1.19 terrain. Block palette accuracy is handled by
        // the period-accurate datapack (world_type controls which pack is active).
        WorldCreator creator = new WorldCreator(name)
            .environment(World.Environment.NORMAL)
            .seed(seed)
            .generateStructures(true);

        // Step 1: create world on main thread (fast, ~50ms)
        CompletableFuture<World> createFuture = new CompletableFuture<>();
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
                    w.setTime(6000);
                    if (worldType != WorldType.MODERN) {
                        int surfaceY = w.getHighestBlockYAt(0, 0) + 1;
                        w.setSpawnLocation(0, surfaceY, 0);
                        targetSpawn(w);
                    }
                }
                createFuture.complete(w);
            } catch (Exception e) {
                plugin.getLogger().severe("World creation failed: " + e.getMessage());
                createFuture.complete(null);
            }
        });

        World w;
        try {
            w = createFuture.get(30, TimeUnit.SECONDS);
        } catch (Exception e) {
            plugin.getLogger().warning("World creation timed out: " + e.getMessage());
            return null;
        }
        if (w == null) return null;

        // Step 2: pre-generate spawn region using Paper's async chunk API.
        // getChunkAtAsync runs on Paper's chunk worker threads — main thread stays free.
        // bgPool thread blocks here (fine, its only job is pool management).
        pregenerateSpawnRegion(w);
        return w;
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
            plugin.getLogger().info(world.getName() + ": searching for " + structureKey + " within 50 chunks...");
            var result = world.locateNearestStructure(origin, structure, 50, false);
            if (result != null) {
                Location found = result.getLocation();
                found.setY(world.getHighestBlockYAt(found.getBlockX(), found.getBlockZ()) + 1);
                world.setSpawnLocation(found);
                plugin.getLogger().info(world.getName() + ": spawn → " + structureKey +
                    " @ " + found.getBlockX() + "," + found.getBlockZ());
            } else {
                plugin.getLogger().info(world.getName() + ": " + structureKey + " not found within 50 chunks, keeping vanilla spawn");
            }
        } catch (Exception e) {
            plugin.getLogger().warning("locateAndSetSpawn(" + structureKey + ") threw: " + e);
        }
    }

    // ── Player join / quit ─────────────────────────────────────────────────────

    /**
     * Fires before Paper's setInitialSpawn() runs. Setting spawn here marks it as
     * "already set" so Paper skips the expensive chunk-scan for a valid spawn point.
     */
    @EventHandler
    public void onWorldInit(WorldInitEvent event) {
        // For custom-generator worlds (void, legacy terrain), lock spawn at (0,64,0)
        // so Paper skips its expensive surface-scan — the generator handles spawn itself.
        // For MODERN vanilla worlds, let Paper find a proper non-ocean spawn naturally.
        if (event.getWorld().getName().startsWith("bs_") && worldType != WorldType.MODERN) {
            event.getWorld().setSpawnLocation(0, 64, 0);
        }
    }

    /** In void scatter worlds, cancel void damage and teleport back to spawn instead. */
    @EventHandler
    public void onVoidFall(EntityDamageEvent event) {
        if (event.getCause() != EntityDamageEvent.DamageCause.VOID) return;
        if (!(event.getEntity() instanceof Player player)) return;
        if (!player.getWorld().getName().startsWith("bs_")) return;
        if (!voidEnabled) return;
        event.setCancelled(true);
        player.setFallDistance(0f);
        if (!plugin.isEnabled()) return; // server shutting down — skip scheduling
        // Defer one tick — teleporting inside a damage event corrupts physics state and freezes the player
        Bukkit.getScheduler().runTask(plugin, () -> {
            player.setVelocity(new org.bukkit.util.Vector(0, 0, 0));
            player.teleport(player.getWorld().getSpawnLocation());
        });
    }

    @EventHandler
    public void onPlayerJoin(PlayerJoinEvent event) {
        Player player = event.getPlayer();
        event.joinMessage(null);

        String modeStr = assignMode();
        boolean isVoid        = "void".equals(modeStr);
        boolean isHermitcraft = "hermitcraft".equals(modeStr);

        // Hermitcraft: pick a random loaded season and spawn point — no pool needed
        if (isHermitcraft) {
            java.util.List<Integer> loaded = new java.util.ArrayList<>(hermitWorlds.keySet());
            if (loaded.isEmpty()) {
                plugin.getLogger().warning("No hermitcraft worlds loaded — falling back to survival");
                modeStr = "survival";
                isHermitcraft = false;
            } else {
                int season = loaded.get(rng.nextInt(loaded.size()));
                World hWorld = hermitWorlds.get(season);
                java.util.List<double[]> pts = hermitSpawnPoints.getOrDefault(season, java.util.List.of());
                double[] pt = pts.isEmpty()
                    ? new double[]{hWorld.getSpawnLocation().getX(), 64, hWorld.getSpawnLocation().getZ()}
                    : pts.get(rng.nextInt(pts.size()));

                playerWorlds.put(player.getUniqueId(), hWorld);
                World finalWorld = hWorld;
                String finalMode = "hermitcraft_s" + season;

                Bukkit.getScheduler().runTask(plugin, () -> {
                    player.getInventory().clear();
                    player.setGameMode(GameMode.SURVIVAL);
                    Location spawnLoc = new Location(finalWorld, pt[0], pt[1], pt[2]);
                    player.teleport(spawnLoc);
                    if (toolRandomisation) { giveRandomTools(player); scheduleToolCheck(player); }
                    plugin.getLogger().info("Session: " + player.getName() + " → " + finalWorld.getName() + " [" + finalMode + "] @ " + (int)pt[0] + "," + (int)pt[1] + "," + (int)pt[2]);
                    Bukkit.getScheduler().runTaskLater(plugin, () -> {
                        if (player.isOnline()) sendSessionStart(player, finalMode);
                    }, 40L);
                });

                int timerId = Bukkit.getScheduler().scheduleSyncDelayedTask(plugin,
                    () -> { if (player.isOnline()) endSession(player, "timeout"); },
                    (long) sessionDurationSeconds * 20L);
                sessionTimers.put(player.getUniqueId(), timerId);
                return;
            }
        }

        Queue<World> pool = isVoid ? voidPool : survivalPool;
        World world = pool.poll();
        if (world == null) {
            plugin.getLogger().warning("World pool empty — generating on-demand for " + player.getName());
            world = isVoid ? generateVoidWorldSync() : generateSurvivalWorldSync();
        }
        if (world == null) {
            player.kick(net.kyori.adventure.text.Component.text("§cFailed to create session world. Try again in a moment."));
            refillPool();
            return;
        }

        playerWorlds.put(player.getUniqueId(), world);
        refillPool();

        World finalWorld = world;
        String finalMode = modeStr; // capture before lambda (modeStr may have been reassigned)

        Bukkit.getScheduler().runTask(plugin, () -> {
            player.getInventory().clear();
            player.setGameMode(GameMode.SURVIVAL);
            player.teleport(finalWorld.getSpawnLocation());

            if (isVoid) {
                giveScatterInventory(player);
            } else if (toolRandomisation) {
                giveRandomTools(player);
                scheduleToolCheck(player);
            }

            plugin.getLogger().info("Session: " + player.getName() +
                " → " + finalWorld.getName() + " [" + finalMode + "]");

            Bukkit.getScheduler().runTaskLater(plugin, () -> {
                if (player.isOnline()) {
                    sendSessionStart(player, finalMode);
                    plugin.getLogger().info("Sent session_start → " + player.getName() + " [" + finalMode + "]");
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

        // Hermitcraft worlds are permanent — never unload or delete them.
        if (hermitWorlds.containsValue(world)) {
            World fallback = Bukkit.getWorlds().stream()
                .filter(w -> !w.getName().startsWith("bs_") && !w.getName().startsWith("hc"))
                .findFirst().orElse(Bukkit.getWorlds().get(0));
            world.getPlayers().forEach(p -> p.teleport(fallback.getSpawnLocation()));
            return;
        }
        Bukkit.getScheduler().runTaskLater(plugin, () -> unloadAndDeleteWorld(world), 5L);
    }

    private void unloadAndDeleteWorld(World world) {
        World fallback = Bukkit.getWorlds().stream()
            .filter(w -> !w.getName().startsWith("bs_"))
            .findFirst().orElse(Bukkit.getWorlds().get(0));
        world.getPlayers().forEach(p -> p.teleport(fallback.getSpawnLocation()));
        File worldFolder = world.getWorldFolder();
        // Unload on the next tick to let the teleport complete, then delete async.
        // Do NOT call unloadWorld on the main thread with many loaded chunks — it blocks
        // long enough to trigger the Paper watchdog. Schedule deletion after unload.
        Bukkit.getScheduler().runTaskLater(plugin, () -> {
            Bukkit.unloadWorld(world, false);
            bgPool.submit(() -> deleteDir(worldFolder));
        }, 5L);
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
        if (voidEnabled)        modes.add("void");
        if (hermitcraftEnabled) modes.add("hermitcraft");
        if (modes.isEmpty()) return "survival";
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
                    .generator(new VoidScatterGenerator(voidPalette))
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
                .generator(new VoidScatterGenerator(voidPalette))
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
        w.setTime(6000);
        w.setGameRule(GameRule.ANNOUNCE_ADVANCEMENTS, false);
        w.setGameRule(GameRule.SHOW_DEATH_MESSAGES, false);
        w.setGameRule(GameRule.DO_IMMEDIATE_RESPAWN, true);
    }

    private void giveScatterInventory(Player player) {
        Material[] pool = VoidScatterGenerator.PALETTE_SCATTER_CLASSIC;
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

    /**
     * Pre-generates all chunks within view-distance of spawn using Paper's async
     * chunk API. Chunk gen runs on Paper's worker threads — main thread stays free,
     * so players already in other worlds see no lag. This bgPool thread waits for
     * all chunks to complete before marking the world as pool-ready.
     */
    private void pregenerateSpawnRegion(World w) {
        int spawnCX = w.getSpawnLocation().getBlockX() >> 4;
        int spawnCZ = w.getSpawnLocation().getBlockZ() >> 4;
        int radius = 8; // covers view-distance=6 plus a margin

        java.util.List<CompletableFuture<org.bukkit.Chunk>> futures = new java.util.ArrayList<>();
        for (int cx = spawnCX - radius; cx <= spawnCX + radius; cx++) {
            for (int cz = spawnCZ - radius; cz <= spawnCZ + radius; cz++) {
                futures.add(w.getChunkAtAsync(cx, cz));
            }
        }
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
        // No explicit save needed — chunks are in Paper's cache and served from memory on join.
        plugin.getLogger().info("Pre-generated " + futures.size() + " chunks async for " + w.getName());
    }

    private World generateSurvivalWorldSync() {
        CompletableFuture<World> f = new CompletableFuture<>();
        bgPool.submit(() -> f.complete(generateWorld()));
        try { return f.get(120, TimeUnit.SECONDS); } catch (Exception e) { return null; }
    }

    private World generateLegacyWorldFromBg(LegacyChunkGenerator.Era era) {
        String name = "bs_" + worldCounter.incrementAndGet() + "_" + Long.toHexString(rng.nextLong() & 0xFFFFFFL);
        CompletableFuture<World> future = new CompletableFuture<>();
        Bukkit.getScheduler().runTask(plugin, () -> {
            try {
                World w = new WorldCreator(name)
                    .environment(World.Environment.NORMAL)
                    .generator(new LegacyChunkGenerator(era))
                    .generateStructures(false)
                    .createWorld();
                if (w != null) applyLegacyWorldRules(w);
                future.complete(w);
            } catch (Exception e) {
                plugin.getLogger().severe("Legacy world creation failed (" + era + "): " + e.getMessage());
                future.complete(null);
            }
        });
        try { return future.get(120, TimeUnit.SECONDS); } catch (Exception e) { return null; }
    }

    private World generateLegacyWorldSync(LegacyChunkGenerator.Era era) {
        String name = "bs_" + worldCounter.incrementAndGet() + "_" + Long.toHexString(rng.nextLong() & 0xFFFFFFL);
        try {
            World w = new WorldCreator(name)
                .environment(World.Environment.NORMAL)
                .generator(new LegacyChunkGenerator(era))
                .generateStructures(false)
                .createWorld();
            if (w != null) applyLegacyWorldRules(w);
            return w;
        } catch (Exception e) {
            plugin.getLogger().severe("Legacy world creation failed (" + era + "): " + e.getMessage());
            return null;
        }
    }

    private void applyLegacyWorldRules(World w) {
        w.setDifficulty(Difficulty.PEACEFUL);
        w.setGameRule(GameRule.DO_MOB_SPAWNING, false);
        w.setGameRule(GameRule.DO_DAYLIGHT_CYCLE, true);
        w.setTime(6000);
        w.setGameRule(GameRule.ANNOUNCE_ADVANCEMENTS, false);
        w.setGameRule(GameRule.SHOW_DEATH_MESSAGES, false);
        w.setGameRule(GameRule.DO_IMMEDIATE_RESPAWN, true);
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
        for (World w : survivalPool) unloadAndDeleteWorld(w);
        for (World w : voidPool)     unloadAndDeleteWorld(w);
        survivalPool.clear();
        voidPool.clear();
    }
}
