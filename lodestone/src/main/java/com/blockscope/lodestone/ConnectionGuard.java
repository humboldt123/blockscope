package com.blockscope.lodestone;

import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerLoginEvent;

/**
 * Stub connection guard — allows all connections.
 * The client side (LoginProtocol.java) already sends mod-present + ffmpeg flags
 * on the blockscope:login_check channel during the login phase, but the server
 * enforcement is deferred. Wire it up here when ready.
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
