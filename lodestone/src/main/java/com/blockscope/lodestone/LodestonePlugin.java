package com.blockscope.lodestone;

import org.bukkit.plugin.java.JavaPlugin;

public class LodestonePlugin extends JavaPlugin {

    private static LodestonePlugin instance;
    private SessionManager sessionManager;
    private ConnectionGuard connectionGuard;
    private MirrorPairManager pairManager;

    @Override
    public void onEnable() {
        instance = this;
        saveDefaultConfig();

        sessionManager  = new SessionManager(this);
        connectionGuard = new ConnectionGuard(this);
        pairManager     = new MirrorPairManager(this, sessionManager);
        sessionManager.setPairManager(pairManager);
        // StructureGuide disabled: locateNearestStructure() blocks the main thread for 10+s
        // via NoiseBasedChunkGenerator.getBaseHeight() → vanilla density function tree.
        // TODO: move structure scanning to an async task or remove entirely.

        getServer().getPluginManager().registerEvents(connectionGuard, this);
        getServer().getPluginManager().registerEvents(sessionManager, this);

        // Fill chests in every chunk that loads in a bridge (skywars/bedwars) world.
        // This covers islands the bot bridges to AFTER session start — those chunks
        // weren't loaded at join so fillSkyWarsChests() missed them. Mineflayer's
        // bot.findBlock() will see the block type regardless, but having filled chests
        // means when the bot opens one it finds items rather than an empty inventory.
        getServer().getPluginManager().registerEvents(
            new org.bukkit.event.Listener() {
                @org.bukkit.event.EventHandler(priority = org.bukkit.event.EventPriority.MONITOR)
                public void onChunkLoad(org.bukkit.event.world.ChunkLoadEvent e) {
                    if (!e.isNewChunk() && sessionManager.isBridgeSessionWorld(e.getWorld().getName())) {
                        int n = sessionManager.fillChunkChests(e.getChunk());
                        if (n > 0) getLogger().fine("ChunkLoad: filled " + n + " chests in "
                            + e.getWorld().getName() + " @ " + e.getChunk().getX() + "," + e.getChunk().getZ());
                    }
                }
            }, this);

        SpawnCommand spawnCommand = new SpawnCommand(this, sessionManager);
        getCommand("spawn").setExecutor(spawnCommand);
        getCommand("spawn").setTabCompleter(spawnCommand);
        getCommand("spoofspawn").setExecutor(spawnCommand);

        MapCommand mapCommand = new MapCommand(sessionManager);
        getCommand("maps").setExecutor(mapCommand);
        getCommand("visit").setExecutor(mapCommand);
        getCommand("visit").setTabCompleter(mapCommand);

        // Manual (controller, camera) mirror override for debugging.
        getCommand("mirrorpair").setExecutor((sender, cmd, label, args) ->
            pairManager.handleMirrorPair(sender, args));

        sessionManager.startWorldPool();
        getLogger().info("Lodestone enabled.");
    }

    @Override
    public void onDisable() {
        if (sessionManager != null) sessionManager.shutdown();
        getLogger().info("Lodestone disabled.");
    }

    public static LodestonePlugin get() { return instance; }
    public SessionManager getSessionManager() { return sessionManager; }
}
