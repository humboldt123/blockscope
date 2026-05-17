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

        getServer().getPluginManager().registerEvents(connectionGuard, this);
        getServer().getPluginManager().registerEvents(sessionManager, this);

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
