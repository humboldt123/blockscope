package com.blockscope.lodestone;

import com.github.retrooper.packetevents.PacketEvents;
import com.github.retrooper.packetevents.api.PacketEventsAPI;
import io.github.retrooper.packetevents.factory.spigot.SpigotPacketEventsBuilder;
import org.bukkit.plugin.java.JavaPlugin;

public class LodestonePlugin extends JavaPlugin {

    private static LodestonePlugin instance;
    private SessionManager sessionManager;
    private ConnectionGuard connectionGuard;

    @Override
    public void onLoad() {
        PacketEvents.setAPI(SpigotPacketEventsBuilder.build(this));
        PacketEvents.getAPI().load();
    }

    @Override
    public void onEnable() {
        instance = this;
        saveDefaultConfig();

        sessionManager  = new SessionManager(this);
        connectionGuard = new ConnectionGuard(this, sessionManager);

        PacketEvents.getAPI().getEventManager().registerListeners(connectionGuard);
        PacketEvents.getAPI().init();

        getServer().getPluginManager().registerEvents(sessionManager, this);

        sessionManager.startWorldPool();
        getLogger().info("Lodestone enabled — session manager running.");
    }

    @Override
    public void onDisable() {
        sessionManager.shutdown();
        PacketEvents.getAPI().terminate();
        getLogger().info("Lodestone disabled.");
    }

    public static LodestonePlugin get() { return instance; }
    public SessionManager getSessionManager() { return sessionManager; }
}
