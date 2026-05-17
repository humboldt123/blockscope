package com.blockscope.network;

import com.blockscope.util.FfmpegChecker;
import net.fabricmc.fabric.api.client.networking.v1.ClientLoginNetworking;
import net.fabricmc.fabric.api.networking.v1.PacketByteBufs;
import net.minecraft.util.Identifier;

import java.util.concurrent.CompletableFuture;

/**
 * Blockscope login handshake protocol.
 *
 * When the multi-world server sends a query on CHANNEL during the login phase,
 * the client responds with:
 *   byte 1 — always 1 (proves the mod is installed)
 *   byte 2 — 1 if ffmpeg is available, 0 otherwise
 *
 * If the client has no mod it never responds, which the server treats as a
 * missing-mod denial.  The server enforces the check before the player fully
 * joins, so the denial appears as a "Failed to connect" screen rather than a kick.
 */
public class LoginProtocol {

    public static final Identifier CHANNEL = new Identifier("blockscope", "login_check");

    public static void registerClient() {
        ClientLoginNetworking.registerGlobalReceiver(CHANNEL, (client, handler, buf, listenerAdder) -> {
            boolean ffmpegOk = FfmpegChecker.isAvailable();
            var response = PacketByteBufs.create();
            response.writeBoolean(true);       // mod present
            response.writeBoolean(ffmpegOk);   // ffmpeg status
            System.out.println("[Blockscope] Login query from server — responding (ffmpeg=" + ffmpegOk + ")");
            return CompletableFuture.completedFuture(response);
        });
    }
}
