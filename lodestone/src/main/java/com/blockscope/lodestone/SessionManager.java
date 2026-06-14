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
    private final boolean libraryEnabled;
    private final boolean baritoneControlEnabled;
    private final org.bukkit.Material[] voidPalette;
    // SOURCE world names (case-insensitive) served in ADVENTURE mode — the bot can't
    // break, preserving the most valuable maps. Everything else stays SURVIVAL.
    private final java.util.Set<String> adventureMaps;
    // Source-name substrings whose maps are sky-islands: bot bridges across void (gets
    // cobblestone + allowPlace). Matched as substrings so "skywars"/"bedwars" cover all.
    private final java.util.List<String> bridgeMaps;
    enum WorldType { MODERN, BETA, LEGACY_2013 }
    private final WorldType worldType;

    // Library worlds: pre-built converted maps (build maps + hermitcraft) under
    // map_library/. Served copy-on-join in ADVENTURE mode so the bot walks the
    // build without modifying it, and each session gets a throwaway copy that is
    // deleted on disconnect (the original source is never touched).
    private static final String LIBRARY_DIR = "map_library";
    // Discovered source world dirs (each contains a converted level.dat).
    private final java.util.List<File> librarySources = new java.util.ArrayList<>();
    // Spawn candidates per SOURCE world name, loaded from spawn_candidates.json.
    private final java.util.Map<String, java.util.List<SpawnPoint>> sourceSpawns
        = new java.util.concurrent.ConcurrentHashMap<>();
    // Spawn candidates for each pooled COPY, keyed by the copy's world name.
    // NOTE: a copy shares the SAME list reference as its source (see copyLibraryWorld),
    // so in-game spawn edits via /spawn mutate both consistently.
    private final java.util.Map<String, java.util.List<SpawnPoint>> librarySpawns
        = new java.util.concurrent.ConcurrentHashMap<>();
    // Which source world each pooled COPY was made from (for diagnostics logging).
    private final java.util.Map<String, String> librarySource
        = new java.util.concurrent.ConcurrentHashMap<>();

    private final Queue<World> survivalPool = new ConcurrentLinkedQueue<>();
    private final Queue<World> voidPool     = new ConcurrentLinkedQueue<>();
    private final Queue<World> libraryPool  = new ConcurrentLinkedQueue<>();
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
        this.survivalEnabled        = cfg.getBoolean("survival_worlds", false);
        this.voidEnabled            = cfg.getBoolean("void_worlds", true);
        this.libraryEnabled         = cfg.getBoolean("library_worlds", true);
        this.baritoneControlEnabled = cfg.getBoolean("baritone_control_enabled", true);
        this.voidPalette = VoidScatterGenerator.forConfig(
            cfg.getString("void_scatter_palette", "scatter_classic"));
        // Case-insensitive set of source maps to serve in ADVENTURE mode.
        java.util.Set<String> adv = new java.util.TreeSet<>(String.CASE_INSENSITIVE_ORDER);
        java.util.List<String> advList = cfg.getStringList("adventure_maps");
        if (advList.isEmpty()) advList = java.util.Arrays.asList("hermitcraft2", "hermitcraft3");
        adv.addAll(advList);
        this.adventureMaps = adv;
        // Sky-island maps (bridge across void). Substring match on source name.
        java.util.List<String> brList = cfg.getStringList("bridge_maps");
        if (brList.isEmpty()) brList = java.util.Arrays.asList("skywars", "bedwars");
        java.util.List<String> br = new java.util.ArrayList<>();
        for (String s : brList) br.add(s.toLowerCase());
        this.bridgeMaps = br;
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
        if (libraryEnabled) discoverLibrarySources();
        refillPool();
    }

    // ── Library source discovery ─────────────────────────────────────────────────

    /**
     * Scan map_library/ for converted worlds (a dir containing level.dat). For each,
     * load its spawn_candidates.json if present. Worlds without spawn candidates are
     * still usable — they fall back to the world's own spawn location on join.
     */
    private void discoverLibrarySources() {
        File dir = new File(LIBRARY_DIR);
        File[] entries = dir.listFiles(File::isDirectory);
        if (entries == null || entries.length == 0) {
            plugin.getLogger().warning("library_worlds enabled but no worlds found in " + dir.getAbsolutePath());
            return;
        }
        for (File w : entries) {
            if (!new File(w, "level.dat").exists()) continue;
            librarySources.add(w);
            java.util.List<SpawnPoint> spawns = loadSpawnCandidates(w);
            if (!spawns.isEmpty()) sourceSpawns.put(w.getName(), spawns);
            plugin.getLogger().info("Library source: " + w.getName() + " (" + spawns.size() + " spawn candidates)");
        }
        plugin.getLogger().info("Discovered " + librarySources.size() + " library worlds.");
    }

    /**
     * Parse spawn_candidates.json:
     *   {"spawns": [[x,y,z], ...], "labels": ["xisuma"|"grid"|"manual", ...]}
     * labels[i] pairs with spawns[i]; a missing/short labels array defaults each to "grid".
     * Returns an empty list if absent/bad.
     */
    private java.util.List<SpawnPoint> loadSpawnCandidates(File worldDir) {
        java.util.List<SpawnPoint> out = new java.util.ArrayList<>();
        File f = new File(worldDir, "spawn_candidates.json");
        if (!f.exists()) return out;
        try {
            String raw = new String(java.nio.file.Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8);

            // Parse coordinate triples from the "spawns" array.
            java.util.List<int[]> coords = new java.util.ArrayList<>();
            String compact = raw.replaceAll("\\s", "");
            int arr = compact.indexOf("\"spawns\":[");
            if (arr < 0) return out;
            int start = arr + "\"spawns\":[".length();
            int end = compact.indexOf("]]", start);
            if (end >= 0) {
                String inner = compact.substring(start, end + 1); // [x,y,z],...,[x,y,z]
                for (String triple : inner.split("\\],\\[")) {
                    String[] parts = triple.replace("[", "").replace("]", "").split(",");
                    if (parts.length == 3)
                        coords.add(new int[]{(int) Math.floor(Double.parseDouble(parts[0])),
                                             (int) Math.floor(Double.parseDouble(parts[1])),
                                             (int) Math.floor(Double.parseDouble(parts[2]))});
                }
            }

            // Parse labels (optional) — strings in order.
            java.util.List<String> labels = new java.util.ArrayList<>();
            int lArr = raw.indexOf("\"labels\"");
            if (lArr >= 0) {
                int lStart = raw.indexOf('[', lArr);
                int lEnd = (lStart >= 0) ? raw.indexOf(']', lStart) : -1;
                if (lStart >= 0 && lEnd >= 0) {
                    java.util.regex.Matcher m = java.util.regex.Pattern
                        .compile("\"((?:\\\\.|[^\"\\\\])*)\"")
                        .matcher(raw.substring(lStart, lEnd));
                    while (m.find()) labels.add(m.group(1));
                }
            }

            for (int i = 0; i < coords.size(); i++) {
                int[] c = coords.get(i);
                String label = (i < labels.size()) ? labels.get(i) : "grid";
                out.add(new SpawnPoint(c[0], c[1], c[2], label));
            }
        } catch (Exception e) {
            plugin.getLogger().warning("Failed to parse spawn_candidates.json for " + worldDir.getName() + ": " + e.getMessage());
        }
        return out;
    }

    /** Serialize a spawn list back to the {"spawns":[...],"labels":[...]} format. */
    private String serializeSpawns(java.util.List<SpawnPoint> spawns) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"spawns\": [");
        for (int i = 0; i < spawns.size(); i++) {
            SpawnPoint p = spawns.get(i);
            if (i > 0) sb.append(", ");
            sb.append("[").append(p.x).append(", ").append(p.y).append(", ").append(p.z).append("]");
        }
        sb.append("], \"labels\": [");
        for (int i = 0; i < spawns.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append("\"").append(spawns.get(i).label.replace("\\", "\\\\").replace("\"", "\\\"")).append("\"");
        }
        sb.append("]}");
        return sb.toString();
    }

    /**
     * Write the given spawn list to map_library/&lt;source&gt;/spawn_candidates.json.
     * Synchronizes on the list so concurrent join-path reads see a consistent snapshot.
     */
    void saveSpawnCandidates(String source, java.util.List<SpawnPoint> spawns) throws IOException {
        File f = new File(LIBRARY_DIR + File.separator + source, "spawn_candidates.json");
        String json;
        synchronized (spawns) { json = serializeSpawns(spawns); }
        java.nio.file.Files.write(f.toPath(), json.getBytes(StandardCharsets.UTF_8));
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
            if (libraryEnabled && !librarySources.isEmpty()) {
                while (libraryPool.size() < poolSize) {
                    World w = copyLibraryWorld();
                    if (w != null) {
                        libraryPool.add(w);
                        plugin.getLogger().info("World pool: library world ready (" + w.getName() + "), pool size=" + libraryPool.size());
                    } else {
                        break; // copy failed — avoid a tight retry loop
                    }
                }
            }
        });
    }

    /**
     * Copy a random source world from map_library/ into a fresh bs_lib_* world dir,
     * load it with a void generator (out-of-bounds chunks → void, never new terrain),
     * and remember its spawn candidates. Runs on the bgPool thread; the createWorld
     * call is dispatched to the main thread. Returns the loaded copy, or null on error.
     */
    private World copyLibraryWorld() {
        return copyLibraryWorld(librarySources.get(rng.nextInt(librarySources.size())));
    }

    /**
     * Copy a SPECIFIC source world from map_library/ into a fresh bs_lib_* world dir.
     * Same logic as the random variant but for a caller-chosen source (used by /visit).
     */
    private World copyLibraryWorld(File source) {
        String name = "bs_lib_" + worldCounter.incrementAndGet() + "_" + Long.toHexString(rng.nextLong() & 0xFFFFFFL);
        File dest = new File(name);
        try {
            copyDir(source, dest);
            new File(dest, "session.lock").delete();
            new File(dest, "uid.dat").delete(); // stale world UID from the source copy
        } catch (Exception e) {
            plugin.getLogger().severe("Library copy failed (" + source.getName() + " → " + name + "): " + e.getMessage());
            deleteDir(dest);
            return null;
        }

        CompletableFuture<World> future = new CompletableFuture<>();
        Bukkit.getScheduler().runTask(plugin, () -> {
            try {
                World w = new WorldCreator(name)
                    .environment(World.Environment.NORMAL)
                    .generator(new VoidGenerator())
                    .generateStructures(false)
                    .createWorld();
                if (w != null) {
                    w.setDifficulty(Difficulty.PEACEFUL);
                    w.setGameRule(GameRule.DO_MOB_SPAWNING, false);
                    w.setGameRule(GameRule.DO_DAYLIGHT_CYCLE, true);
                    w.setGameRule(GameRule.ANNOUNCE_ADVANCEMENTS, false);
                    w.setGameRule(GameRule.SHOW_DEATH_MESSAGES, false);
                    w.setGameRule(GameRule.DO_IMMEDIATE_RESPAWN, true);
                    w.setTime(6000);
                    java.util.List<SpawnPoint> spawns = sourceSpawns.get(source.getName());
                    if (spawns != null && !spawns.isEmpty()) librarySpawns.put(name, spawns);
                    librarySource.put(name, source.getName());
                }
                future.complete(w);
            } catch (Exception e) {
                plugin.getLogger().severe("Library world load failed (" + name + "): " + e.getMessage());
                future.complete(null);
            }
        });
        try {
            World w = future.get(120, TimeUnit.SECONDS);
            if (w == null) { deleteDir(dest); return null; }
            return w;
        } catch (Exception e) {
            plugin.getLogger().warning("Library world load timed out (" + name + "): " + e.getMessage());
            return null;
        }
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
        // For custom-generator worlds (void, library copies, legacy terrain), lock spawn
        // at (0,64,0) so Paper skips its expensive surface-scan — the generator handles
        // spawn itself and players are teleported to a chosen point on join anyway.
        // For MODERN vanilla survival worlds, let Paper find a proper non-ocean spawn.
        String name = event.getWorld().getName();
        if (name.startsWith("bs_lib_") || (name.startsWith("bs_") && worldType != WorldType.MODERN)) {
            event.getWorld().setSpawnLocation(0, 64, 0);
        }
    }

    /**
     * Keep the bot alive in any session world. A death would respawn the player in the
     * main 1.19 world (the bug the tester hit). Cancel VOID and FALL damage outright;
     * on a void fall, teleport back to a valid spawn. Peaceful already covers mobs/hunger.
     */
    @EventHandler
    public void onSessionDamage(EntityDamageEvent event) {
        if (!(event.getEntity() instanceof Player player)) return;
        if (!player.getWorld().getName().startsWith("bs_")) return;
        EntityDamageEvent.DamageCause cause = event.getCause();
        boolean isVoidWorld = player.getWorld().getName().startsWith("bs_lib_") || voidEnabled;
        if (cause == EntityDamageEvent.DamageCause.VOID && isVoidWorld) {
            event.setCancelled(true);
            player.setFallDistance(0f);
            if (!plugin.isEnabled()) return;
            boolean isLibrary = player.getWorld().getName().startsWith("bs_lib_");
            // Defer one tick — teleporting inside a damage event freezes the player.
            Bukkit.getScheduler().runTask(plugin, () -> {
                player.setVelocity(new org.bukkit.util.Vector(0, 0, 0));
                player.teleport(pickSpawn(player.getWorld(), isLibrary));
            });
        } else if (cause == EntityDamageEvent.DamageCause.FALL) {
            event.setCancelled(true);
            player.setFallDistance(0f);
        }
    }

    /** Safety net: if the bot somehow dies, respawn it in the SAME session world, never the 1.19 hub. */
    @EventHandler
    public void onPlayerRespawn(org.bukkit.event.player.PlayerRespawnEvent event) {
        World w = playerWorlds.get(event.getPlayer().getUniqueId());
        if (w == null) return;
        event.setRespawnLocation(pickSpawn(w, w.getName().startsWith("bs_lib_")));
    }

    @EventHandler
    public void onPlayerJoin(PlayerJoinEvent event) {
        Player player = event.getPlayer();
        event.joinMessage(null);

        String category = assignMode(); // "void" | "survival" | "library"

        // Resolve a world for the chosen category. Library worlds come only from the
        // pre-staged pool (the copy is too large to do synchronously at join); if the
        // library pool is momentarily empty, fall back to a fast-to-generate void world.
        World world;
        if ("library".equals(category)) {
            world = libraryPool.poll();
            if (world == null) {
                plugin.getLogger().warning("Library pool empty — falling back to void for " + player.getName());
                category = "void";
                world = voidPool.poll();
                if (world == null) world = generateVoidWorldSync();
            }
        } else if ("void".equals(category)) {
            world = voidPool.poll();
            if (world == null) {
                plugin.getLogger().warning("Void pool empty — generating on-demand for " + player.getName());
                world = generateVoidWorldSync();
            }
        } else {
            world = survivalPool.poll();
            if (world == null) {
                plugin.getLogger().warning("Survival pool empty — generating on-demand for " + player.getName());
                world = generateSurvivalWorldSync();
            }
        }

        if (world == null) {
            player.kick(net.kyori.adventure.text.Component.text("§cFailed to create session world. Try again in a moment."));
            refillPool();
            return;
        }

        playerWorlds.put(player.getUniqueId(), world);
        refillPool();

        final World finalWorld = world;
        final String finalCategory = category;
        final boolean isVoid    = "void".equals(category);
        final boolean isLibrary = "library".equals(category);

        // Library worlds are throwaway copies. Most run in SURVIVAL (the bot may break
        // blocks to free itself — the source map is never touched). The most valuable
        // maps, listed in adventure_maps, run in ADVENTURE so the bot can't damage them.
        final String source = isLibrary ? librarySource.getOrDefault(finalWorld.getName(), "?") : null;
        final boolean isAdventureLibrary = isLibrary && source != null && adventureMaps.contains(source);
        // Sky-island library maps (skywars/bedwars): the bot bridges across void.
        final boolean isBridgeLibrary = isLibrary && !isAdventureLibrary && isBridgeSource(source);
        // Client bot mode string: bridge maps use "bridge", other library "adventure".
        final String clientMode = isLibrary ? (isBridgeLibrary ? "bridge" : "adventure")
                                            : clientModeFor(finalCategory);

        Bukkit.getScheduler().runTask(plugin, () -> {
            player.getInventory().clear();
            player.setGameMode(isAdventureLibrary ? GameMode.ADVENTURE : GameMode.SURVIVAL);

            SpawnPoint chosen = isLibrary ? pickSpawnPoint(finalWorld) : null;
            Location spawn = (chosen != null) ? toLocation(finalWorld, chosen)
                                              : finalWorld.getSpawnLocation();
            player.teleport(spawn);

            if (isVoid) {
                giveScatterInventory(player);
            } else if (toolRandomisation && !isAdventureLibrary) {
                // Survival + survival-library get tools (so the bot can break obstacles).
                // Adventure-library maps skip tools — the player can't break anyway.
                giveRandomTools(player);
                scheduleToolCheck(player);
            }
            // Bridge maps also get cobblestone so Baritone can bridge across void.
            if (isBridgeLibrary) giveBridgeBlocks(player);

            String label = isLibrary ? ("library:" + source) : finalCategory;
            String spawnDesc = (chosen != null)
                ? " spawn '" + chosen.label + "' @ " + spawn.getBlockX() + "," + spawn.getBlockY() + "," + spawn.getBlockZ()
                : " @ " + spawn.getBlockX() + "," + spawn.getBlockY() + "," + spawn.getBlockZ();
            String summary = "World: " + finalWorld.getName() + " [" + label + "] mode=" + clientMode + spawnDesc;
            plugin.getLogger().info("Session: " + player.getName() + " → " + summary);
            // In-game diagnostic so the tester can see exactly what they're in.
            player.sendMessage(net.kyori.adventure.text.Component.text("§e[Blockscope] " + summary));

            Bukkit.getScheduler().runTaskLater(plugin, () -> {
                if (player.isOnline()) {
                    sendSessionStart(player, clientMode);
                    plugin.getLogger().info("Sent session_start → " + player.getName() + " [" + clientMode + "]");
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

    /** Pick the spawn location: a random library candidate, else the world's own spawn. */
    private Location pickSpawn(World world, boolean isLibrary) {
        if (isLibrary) {
            SpawnPoint p = pickSpawnPoint(world);
            if (p != null) return toLocation(world, p);
        }
        return world.getSpawnLocation();
    }

    /** Pick a random SpawnPoint for a library world, or null if none are configured. */
    private SpawnPoint pickSpawnPoint(World world) {
        java.util.List<SpawnPoint> spawns = librarySpawns.get(world.getName());
        if (spawns == null || spawns.isEmpty()) return null;
        synchronized (spawns) {
            if (spawns.isEmpty()) return null;
            return spawns.get(rng.nextInt(spawns.size()));
        }
    }

    /** Convert a SpawnPoint to a Location with x/z centered on the block. */
    static Location toLocation(World world, SpawnPoint p) {
        return new Location(world, p.x + 0.5, p.y, p.z + 0.5);
    }

    // ── Accessors for the /maps and /visit commands ─────────────────────────────

    /**
     * Alphabetically-sorted (source name, spawn-candidate count) pairs for every
     * discovered library source. Count is 0 if the source has no spawn_candidates.json.
     */
    java.util.List<Map.Entry<String, Integer>> listLibrarySources() {
        java.util.List<Map.Entry<String, Integer>> out = new java.util.ArrayList<>();
        for (File f : librarySources) {
            java.util.List<SpawnPoint> spawns = sourceSpawns.get(f.getName());
            out.add(new java.util.AbstractMap.SimpleEntry<>(f.getName(), spawns == null ? 0 : spawns.size()));
        }
        out.sort(Map.Entry.comparingByKey(String.CASE_INSENSITIVE_ORDER));
        return out;
    }

    /** Discovered library source world names, in discovery order (for tab-completion). */
    java.util.List<String> librarySourceNames() {
        java.util.List<String> out = new java.util.ArrayList<>();
        for (File f : librarySources) out.add(f.getName());
        return out;
    }

    /**
     * Resolve a user-typed source name to a discovered source File. Tries exact
     * (case-insensitive) match first, then a unique case-insensitive prefix match.
     * Returns null if not found or ambiguous.
     */
    File resolveSource(String query) {
        if (query == null || query.isEmpty()) return null;
        for (File f : librarySources) {
            if (f.getName().equalsIgnoreCase(query)) return f;
        }
        File prefixMatch = null;
        for (File f : librarySources) {
            if (f.getName().toLowerCase().startsWith(query.toLowerCase())) {
                if (prefixMatch != null) return null; // ambiguous
                prefixMatch = f;
            }
        }
        return prefixMatch;
    }

    /**
     * Serve an op a FRESH COPY of a specific source map for manual inspection/curation.
     * Copy + world load run async on the bgPool (never blocks the main thread); when the
     * world is ready, the player is registered like a session (playerWorlds + CREATIVE +
     * teleport to a spawn) but WITHOUT a bot session_start or a session timer — the visit
     * persists until the op disconnects or runs /visit again. Any previous visit/session
     * world the op is in is torn down first.
     */
    void visitSource(Player player, File source) {
        // Tear down whatever session/visit world the op is currently in.
        if (playerWorlds.get(player.getUniqueId()) != null) {
            endSession(player, "visit-switch");
        }
        final UUID uuid = player.getUniqueId();
        final String sourceName = source.getName();
        bgPool.submit(() -> {
            World w = copyLibraryWorld(source);
            Bukkit.getScheduler().runTask(plugin, () -> {
                if (!player.isOnline()) {
                    // Op left before the copy finished — don't leak the world.
                    if (w != null) unloadAndDeleteWorld(w);
                    return;
                }
                if (w == null) {
                    player.sendMessage(net.kyori.adventure.text.Component.text(
                        "§cFailed to copy source '" + sourceName + "' — see server log."));
                    return;
                }
                playerWorlds.put(uuid, w);
                player.setGameMode(GameMode.CREATIVE);
                player.setAllowFlight(true);
                player.setFlying(true);
                Location spawn = pickSpawn(w, true);
                player.teleport(spawn);
                String summary = "World: " + w.getName() + " [visit:" + sourceName + "] mode=creative @ "
                    + spawn.getBlockX() + "," + spawn.getBlockY() + "," + spawn.getBlockZ();
                plugin.getLogger().info("Visit: " + player.getName() + " → " + summary);
                player.sendMessage(net.kyori.adventure.text.Component.text("§e[Blockscope] " + summary));
                player.sendMessage(net.kyori.adventure.text.Component.text(
                    "§7Curate with /spawn add|del|save. Run /visit again or disconnect to clean up."));
            });
        });
    }

    // ── Accessors for the /spawn command ────────────────────────────────────────

    /** The source world name a live copy was made from, or null if not a library copy. */
    String getLibrarySource(String copyWorldName) { return librarySource.get(copyWorldName); }

    /** The live spawn list for a copy world (shared with its source), or null. */
    java.util.List<SpawnPoint> getSpawnsForWorld(String copyWorldName) {
        return librarySpawns.get(copyWorldName);
    }

    /**
     * Reload a source's spawn list from disk and re-point all in-memory references
     * (the source entry and every live copy made from it) at the fresh list, so the
     * current session sees the reloaded data. Returns the new list size.
     */
    int reloadSpawnsForSource(String source) {
        File srcDir = new File(LIBRARY_DIR + File.separator + source);
        java.util.List<SpawnPoint> fresh = loadSpawnCandidates(srcDir);
        // Use a synchronizable, mutable list so future add/del still work in place.
        java.util.List<SpawnPoint> live = java.util.Collections.synchronizedList(
            new java.util.ArrayList<>(fresh));
        sourceSpawns.put(source, live);
        for (java.util.Map.Entry<String, String> e : librarySource.entrySet()) {
            if (source.equals(e.getValue())) librarySpawns.put(e.getKey(), live);
        }
        return live.size();
    }

    /** Map an internal world category to the client-side BotModule mode string. */
    private String clientModeFor(String category) {
        return switch (category) {
            case "void"    -> "void_scatter";
            case "library" -> "adventure";
            default        -> "survival";
        };
    }

    /** True if the source map is a sky-island map (substring match against bridge_maps). */
    private boolean isBridgeSource(String source) {
        if (source == null) return false;
        String s = source.toLowerCase();
        for (String b : bridgeMaps) if (s.contains(b)) return true;
        return false;
    }

    /** Cobblestone for sky-island bridging (Baritone uses it as throwaway scaffolding). */
    private void giveBridgeBlocks(Player player) {
        for (int i = 0; i < 5; i++) player.getInventory().addItem(new ItemStack(Material.COBBLESTONE, 64));
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

        // All session worlds — including library copies — are throwaway and deleted.
        // The library source under map_library/ is never touched (only its copy is).
        librarySpawns.remove(world.getName());
        librarySource.remove(world.getName());
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
        if (survivalEnabled) modes.add("survival");
        if (voidEnabled)     modes.add("void");
        if (libraryEnabled && !librarySources.isEmpty()) modes.add("library");
        if (modes.isEmpty()) return "void";
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
        for (World w : libraryPool)  unloadAndDeleteWorld(w);
        survivalPool.clear();
        voidPool.clear();
        libraryPool.clear();
    }
}
