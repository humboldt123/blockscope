package com.blockscope.network;

import com.blockscope.ReplayModIntegration;
import com.blockscope.agent.BotModule;
import com.blockscope.util.Config;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayNetworking;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.TitleScreen;
import net.minecraft.client.network.ServerAddress;
import net.minecraft.client.network.ServerInfo;
import net.minecraft.client.realms.gui.screen.RealmsMainScreen;
import net.minecraft.util.Identifier;

import java.nio.charset.StandardCharsets;

/**
 * Handles the "blockscope:session_start" plugin message from Lodestone.
 *
 * Payload:
 *   byte      — mode string length
 *   byte[n]   — mode string ("creative" or "survival")
 *   int       — session duration in seconds
 *
 * On receipt: set bot mode, start bot, start recording, mark session active.
 * On disconnect while session active: auto-reconnect after configurable delay.
 */
public class SessionProtocol {

    public static final Identifier CHANNEL = new Identifier("blockscope", "session_start");

    private static ReplayModIntegration replayMod;
    private static boolean baritonePresent;

    // Set when session_start is received; cleared on disconnect.
    // Used by BlockscopeClient's DISCONNECT handler to decide whether to reconnect.
    private static volatile boolean inManagedSession = false;
    private static volatile ServerInfo lastServer = null;

    public static boolean isInManagedSession() { return inManagedSession; }
    public static ServerInfo getLastServer()    { return lastServer; }
    public static void clearSession()          { inManagedSession = false; }

    public static void registerClient(ReplayModIntegration replay) {
        replayMod = replay;
        baritonePresent = net.fabricmc.loader.api.FabricLoader.getInstance().isModLoaded("baritone");

        ClientPlayNetworking.registerGlobalReceiver(CHANNEL, (client, handler, buf, responseSender) -> {
            int modeLen = buf.readByte() & 0xFF;
            byte[] modeBytes = new byte[modeLen];
            buf.readBytes(modeBytes);
            String mode = new String(modeBytes, StandardCharsets.UTF_8);
            int durationSeconds = buf.readInt();

            client.execute(() -> {
                System.out.println("[Blockscope] session_start: mode=" + mode + " duration=" + durationSeconds + "s");

                // Mark session active and snapshot the server we're connected to
                inManagedSession = true;
                lastServer = client.getCurrentServerEntry();

                if (baritonePresent) {
                    BotModule bot = BotModule.getInstance();
                    BotModule.Mode botMode = "creative".equalsIgnoreCase(mode)
                        ? BotModule.Mode.CREATIVE_SURVEY
                        : BotModule.Mode.SURVIVAL_GATHER;
                    boolean wasRunning = bot.isRunning();
                    if (wasRunning) bot.stop();
                    bot.setMode(botMode);
                    if (wasRunning) {
                        // Baritone needs ~1.5s to fully tear down before we can restart cleanly
                        Thread restartThread = new Thread(() -> {
                            try { Thread.sleep(1500); } catch (InterruptedException ignored) {}
                            client.execute(bot::start);
                        }, "blockscope-bot-restart");
                        restartThread.setDaemon(true);
                        restartThread.start();
                    } else {
                        bot.start();
                    }
                }

                if (replayMod != null && !replayMod.isRecording()) {
                    replayMod.toggle();
                }
            });
        });
    }

    /** Called by BlockscopeClient's DISCONNECT handler after stopping recording/bot. */
    public static void onDisconnect(MinecraftClient client) {
        if (!inManagedSession) return;
        inManagedSession = false;

        Config cfg = Config.getInstance();
        if (!cfg.autoReconnect || lastServer == null) return;

        ServerInfo server = lastServer;
        int delayMs = cfg.reconnectDelaySeconds * 1000;

        System.out.println("[Blockscope] Session ended — reconnecting to " + server.address + " in " + cfg.reconnectDelaySeconds + "s");

        Thread reconnect = new Thread(() -> {
            try { Thread.sleep(delayMs); } catch (InterruptedException ignored) {}
            client.execute(() -> {
                try {
                    net.minecraft.client.gui.screen.ConnectScreen.connect(
                        new TitleScreen(), client,
                        ServerAddress.parse(server.address), server);
                } catch (Exception e) {
                    System.err.println("[Blockscope] Reconnect failed: " + e.getMessage());
                }
            });
        }, "blockscope-reconnect");
        reconnect.setDaemon(true);
        reconnect.start();
    }
}
