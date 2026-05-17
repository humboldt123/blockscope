package com.blockscope.network;

import com.blockscope.ReplayModIntegration;
import com.blockscope.agent.BotModule;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayNetworking;
import net.minecraft.util.Identifier;

import java.nio.charset.StandardCharsets;

/**
 * Handles the "blockscope:session_start" plugin message from Lodestone.
 *
 * Payload (written by SessionManager.sendSessionStart):
 *   byte          — mode string length
 *   byte[n]       — mode string ("creative" or "survival")
 *   int (4 bytes) — session duration in seconds
 *
 * On receipt: set bot mode, start bot, start recording.
 */
public class SessionProtocol {

    public static final Identifier CHANNEL = new Identifier("blockscope", "session_start");

    private static ReplayModIntegration replayMod;
    private static boolean baritonePresent;

    public static void registerClient(ReplayModIntegration replay) {
        replayMod = replay;
        baritonePresent = net.fabricmc.loader.api.FabricLoader.getInstance().isModLoaded("baritone");

        ClientPlayNetworking.registerGlobalReceiver(CHANNEL, (client, handler, buf, responseSender) -> {
            // Read on network thread, act on client thread
            int modeLen = buf.readByte() & 0xFF;
            byte[] modeBytes = new byte[modeLen];
            buf.readBytes(modeBytes);
            String mode = new String(modeBytes, StandardCharsets.UTF_8);
            int durationSeconds = buf.readInt();

            client.execute(() -> {
                System.out.println("[Blockscope] session_start: mode=" + mode + " duration=" + durationSeconds + "s");

                if (baritonePresent) {
                    BotModule bot = BotModule.getInstance();
                    // Set correct mode before starting
                    if ("creative".equalsIgnoreCase(mode)) {
                        if (bot.getMode() != BotModule.Mode.CREATIVE_SURVEY) bot.cycleMode();
                    } else {
                        if (bot.getMode() != BotModule.Mode.SURVIVAL_GATHER) bot.cycleMode();
                    }
                    if (!bot.isRunning()) bot.start();
                }

                // Start recording
                if (replayMod != null && !replayMod.isRecording()) {
                    replayMod.toggle();
                }
            });
        });
    }
}
