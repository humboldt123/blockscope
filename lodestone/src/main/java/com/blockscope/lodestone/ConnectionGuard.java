package com.blockscope.lodestone;

import com.github.retrooper.packetevents.event.PacketListenerAbstract;
import com.github.retrooper.packetevents.event.PacketReceiveEvent;
import com.github.retrooper.packetevents.protocol.packettype.PacketType;
import com.github.retrooper.packetevents.wrapper.login.client.WrapperLoginClientLoginStart;
import net.kyori.adventure.text.Component;
import org.bukkit.Bukkit;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Denies connection at login phase to clients without the Blockscope mod or ffmpeg.
 *
 * Flow:
 *  1. Client sends LOGIN_START → we send a "blockscope:login_check" login-phase custom
 *     payload packet and start a short timeout.
 *  2. Client mod responds with LOGIN_ACKNOWLEDGED carrying [modPresent=1, ffmpegOk=0/1].
 *  3. If no response within timeout, or ffmpegOk=0, we disconnect before the player joins.
 *
 * NOTE: PacketEvents 2.x exposes login-phase custom payload via
 * PacketType.Login.Client.PLUGIN_RESPONSE / Server.PLUGIN_REQUEST.
 */
public class ConnectionGuard extends PacketListenerAbstract {

    private static final String CHANNEL = "blockscope:login_check";
    // Map from connection UUID → whether we sent the check (waiting for response)
    private final Map<Object, PendingCheck> pending = new ConcurrentHashMap<>();

    private final LodestonePlugin plugin;
    private final SessionManager sessions;

    public ConnectionGuard(LodestonePlugin plugin, SessionManager sessions) {
        this.plugin  = plugin;
        this.sessions = sessions;
    }

    @Override
    public void onPacketReceive(PacketReceiveEvent event) {
        if (event.getPacketType() == PacketType.Login.Client.LOGIN_START) {
            handleLoginStart(event);
        } else if (event.getPacketType() == PacketType.Login.Client.PLUGIN_RESPONSE) {
            handlePluginResponse(event);
        }
    }

    private void handleLoginStart(PacketReceiveEvent event) {
        Object channel = event.getChannel();

        // Send our login check payload to the client
        plugin.getServer().getScheduler().runTaskAsynchronously(plugin, () -> {
            try {
                sendLoginCheckPacket(channel);
            } catch (Exception e) {
                plugin.getLogger().warning("Failed to send login check: " + e.getMessage());
            }
        });

        pending.put(channel, new PendingCheck(System.currentTimeMillis()));

        // Schedule timeout: if no response in 3 s, deny
        Bukkit.getScheduler().runTaskLaterAsynchronously(plugin, () -> {
            PendingCheck check = pending.remove(channel);
            if (check != null && !check.responded) {
                disconnect(channel, "§cBlockscope mod required to join this server.\n§7Install the Blockscope Fabric mod.");
            }
        }, 60L); // 3 s at 20 ticks/s
    }

    private void handlePluginResponse(PacketReceiveEvent event) {
        Object channel = event.getChannel();
        PendingCheck check = pending.get(channel);
        if (check == null) return;

        // Parse the two bytes: [modPresent, ffmpegOk]
        byte[] data = event.getByteBuf() != null ? readBytes(event) : new byte[0];
        boolean modPresent = data.length > 0 && data[0] == 1;
        boolean ffmpegOk   = data.length > 1 && data[1] == 1;

        check.responded = true;

        if (!modPresent || !ffmpegOk) {
            pending.remove(channel);
            String reason = !modPresent
                ? "§cBlockscope mod required to join this server.\n§7Install the Blockscope Fabric mod."
                : "§cffmpeg not found on your machine.\n§7Install ffmpeg and add it to PATH, then restart Minecraft.";
            disconnect(channel, reason);
        } else {
            pending.remove(channel);
            // Allow join — SessionManager.onPlayerJoin will handle world assignment
            plugin.getLogger().info("Login check passed for incoming connection.");
        }
    }

    private void sendLoginCheckPacket(Object channel) {
        // Send a login-phase plugin request packet on our channel.
        // PacketEvents 2.x: use WrapperLoginServerPluginRequest
        var packet = new com.github.retrooper.packetevents.wrapper.login.server
            .WrapperLoginServerPluginRequest(0x00, CHANNEL, new byte[0]);
        PacketEvents.getAPI().getProtocolManager().sendPacket(channel, packet);
    }

    private void disconnect(Object channel, String message) {
        plugin.getServer().getScheduler().runTask(plugin, () -> {
            try {
                var packet = new com.github.retrooper.packetevents.wrapper.login.server
                    .WrapperLoginServerDisconnect(
                        net.kyori.adventure.text.serializer.legacy.LegacyComponentSerializer
                            .legacySection().deserialize(message));
                PacketEvents.getAPI().getProtocolManager().sendPacket(channel, packet);
                PacketEvents.getAPI().getProtocolManager().closeChannel(channel);
            } catch (Exception e) {
                plugin.getLogger().warning("Disconnect failed: " + e.getMessage());
            }
        });
    }

    private byte[] readBytes(PacketReceiveEvent event) {
        try {
            var buf = event.getByteBuf();
            // Skip the message ID varint (already consumed by PacketEvents wrapper layer)
            // then read the data payload
            var wrapper = new com.github.retrooper.packetevents.wrapper.login.client
                .WrapperLoginClientPluginResponse(event);
            byte[] data = wrapper.getData();
            return data != null ? data : new byte[0];
        } catch (Exception e) {
            return new byte[0];
        }
    }

    private static class PendingCheck {
        final long sentAt;
        volatile boolean responded = false;
        PendingCheck(long sentAt) { this.sentAt = sentAt; }
    }
}
