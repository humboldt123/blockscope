package com.blockscope.lodestone;

import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerLoginEvent;

/**
 * Stub connection guard for local testing — allows all connections.
 * Production version will use PacketEvents to intercept login-phase packets
 * and deny clients missing the Blockscope mod or ffmpeg.
 */
public class ConnectionGuard implements Listener {

    private final LodestonePlugin plugin;

    public ConnectionGuard(LodestonePlugin plugin) {
        this.plugin = plugin;
        plugin.getLogger().info("ConnectionGuard: STUB mode — all connections allowed (local dev)");
    }

    @EventHandler
    public void onLogin(PlayerLoginEvent event) {
        // No checks in stub mode — everyone gets in
        event.allow();
    }
}
