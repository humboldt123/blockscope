package com.blockscope.lodestone;

import org.bukkit.plugin.java.JavaPlugin;

public class LodestonePlugin extends JavaPlugin {

    private static LodestonePlugin instance;
    private SessionManager sessionManager;
    private ConnectionGuard connectionGuard;

    @Override
    public void onEnable() {
        instance = this;
        saveDefaultConfig();

        sessionManager  = new SessionManager(this);
        connectionGuard = new ConnectionGuard(this);
        // StructureGuide disabled: locateNearestStructure() blocks the main thread for 10+s
        // via NoiseBasedChunkGenerator.getBaseHeight() → vanilla density function tree.
        // TODO: move structure scanning to an async task or remove entirely.

        getServer().getPluginManager().registerEvents(connectionGuard, this);
        getServer().getPluginManager().registerEvents(sessionManager, this);

        SpawnCommand spawnCommand = new SpawnCommand(this, sessionManager);
        getCommand("spawn").setExecutor(spawnCommand);
        getCommand("spawn").setTabCompleter(spawnCommand);
        getCommand("spoofspawn").setExecutor(spawnCommand);

        MapCommand mapCommand = new MapCommand(sessionManager);
        getCommand("maps").setExecutor(mapCommand);
        getCommand("visit").setExecutor(mapCommand);
        getCommand("visit").setTabCompleter(mapCommand);

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
