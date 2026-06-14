package com.blockscope;

import com.blockscope.agent.BotModule;
import com.blockscope.agent.InventoryChaosAgent;
import com.blockscope.gui.ConfigScreen;
import com.blockscope.network.LoginProtocol;
import com.blockscope.network.SessionProtocol;
import com.blockscope.recording.RecordingManager;
import com.blockscope.util.FfmpegChecker;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import org.lwjgl.glfw.GLFW;

public class BlockscopeClient implements ClientModInitializer {
    private static KeyBinding recordingToggleKey;
    private static KeyBinding configKey;
    private static KeyBinding chaosAgentKey;
    private static KeyBinding botToggleKey;
    private static KeyBinding botModeKey;
    private static boolean autoJumpDisabled = false;
    private static ReplayModIntegration replayModIntegration;
    private static boolean baritonePresent = false;
    // Headless auto-connect: fire ConnectScreen.connect() once the title screen
    // appears. Guarded so we only initiate one connect attempt; the mod's existing
    // auto-reconnect (SessionProtocol.onDisconnect) handles every subsequent rejoin.
    private static boolean autoConnectAttempted = false;
    private static int autoConnectIdleTicks = 0;

    @Override
    public void onInitializeClient() {
        System.out.println("[Blockscope] Initializing...");

        // Probe ffmpeg eagerly so the result is cached before any server login query
        FfmpegChecker.probe();
        LoginProtocol.registerClient();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (RecordingManager.getInstance().isRecording()) {
                System.out.println("[Blockscope] Game closing - auto-stopping recording...");
                RecordingManager.getInstance().stopRecording();
            }
            // Last-chance .mcpr upload: if the poll thread didn't finish before game closed
            if (RecordingManager.getInstance().getSessionId() != null) {
                System.out.println("[Blockscope] Shutdown hook: attempting .mcpr upload...");
                RecordingManager.getInstance().uploadMcprFile();
            }
        }));

        replayModIntegration = new ReplayModIntegration();
        replayModIntegration.register();
        SessionProtocol.registerClient(replayModIntegration);

        // Stop recording and bot immediately on disconnect so the server select /
        // disconnect screen never leaks into the video. Then auto-reconnect if
        // this was a managed session kicked by Lodestone.
        //
        // IMPORTANT: onDisconnect is called FIRST to spawn the reconnect thread before
        // stopRecording() runs. stopRecording() makes HTTP calls that can block the game
        // thread for several seconds (especially on network-error disconnects). Spawning
        // the reconnect thread early ensures it fires even if cleanup is slow.
        ClientPlayConnectionEvents.DISCONNECT.register((handler, client) -> {
            SessionProtocol.onDisconnect(client);
            if (RecordingManager.getInstance().isRecording()) {
                System.out.println("[Blockscope] Disconnect — stopping recording");
                RecordingManager.getInstance().stopRecording();
            }
            if (baritonePresent && BotModule.getInstance().isRunning()) {
                BotModule.getInstance().stop();
            }
        });

        recordingToggleKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.blockscope.toggle_recording",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_R,
            "category.blockscope"
        ));

        configKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.blockscope.config",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_B,
            "category.blockscope"
        ));

        chaosAgentKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.blockscope.chaos_agent",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_C,
            "category.blockscope"
        ));

        baritonePresent = FabricLoader.getInstance().isModLoaded("baritone");
        if (baritonePresent) {
            BotModule.init();
            botToggleKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.blockscope.bot_toggle",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_G,
                "category.blockscope"
            ));
            botModeKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.blockscope.bot_mode",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_H,
                "category.blockscope"
            ));
            System.out.println("[Blockscope] Baritone detected — bot mode enabled (G=toggle, H=cycle mode)");
        } else {
            System.out.println("[Blockscope] Baritone not found — bot mode disabled");
        }

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            maybeAutoConnect(client);

            if (!autoJumpDisabled && client.options != null) {
                if (client.options.getAutoJump().getValue()) {
                    client.options.getAutoJump().setValue(false);
                    client.options.write();
                    System.out.println("[Blockscope] AutoJump disabled for 1.8.9 compatibility");
                }
                autoJumpDisabled = true;
            }

            if (recordingToggleKey.wasPressed()) {
                replayModIntegration.toggle();
            }

            if (configKey.wasPressed()) {
                client.setScreen(new ConfigScreen(client.currentScreen));
            }

            if (chaosAgentKey.wasPressed()) {
                InventoryChaosAgent.getInstance().toggle();
            }

            if (baritonePresent) {
                if (botToggleKey != null && botToggleKey.wasPressed()) {
                    BotModule.getInstance().toggle();
                }
                if (botModeKey != null && botModeKey.wasPressed()) {
                    BotModule.getInstance().cycleMode();
                }
                BotModule.getInstance().onTick(client);
            }
        });

        System.out.println("[Blockscope] Initialized successfully");
    }

    /**
     * Headless initial auto-connect. 1.19.4 has no --quickPlayMultiplayer launch
     * flag, so we connect from inside the mod: once the TitleScreen is showing and
     * we have no active connection, read the target from config/env and call
     * ConnectScreen.connect(...). Runs at most once; the existing auto-reconnect
     * path takes over for all later rejoins after session-end kicks.
     */
    private static void maybeAutoConnect(net.minecraft.client.MinecraftClient client) {
        if (autoConnectAttempted) return;

        com.blockscope.util.Config cfg = com.blockscope.util.Config.getInstance();
        String host = cfg.autoconnectHost;
        if (host == null || host.isBlank()) {
            autoConnectAttempted = true; // disabled; never check again
            return;
        }

        // Require a clean, disconnected state (no world, no live connection).
        if (client.world != null || client.getNetworkHandler() != null) {
            autoConnectIdleTicks = 0;
            return;
        }
        // Wait until a menu screen is actually showing, then give it a few ticks to
        // settle. We don't gate on TitleScreen specifically: with offline/invalid
        // auth the first menu can be a warning/realms-notice screen, and matching
        // the exact class is brittle across screen wrappers. Any non-null menu with
        // no world is a safe place to initiate the connect.
        if (client.currentScreen == null) {
            autoConnectIdleTicks = 0;
            return;
        }
        if (++autoConnectIdleTicks < 40) { // ~2s at 20 tps
            if (autoConnectIdleTicks == 1) {
                System.out.println("[Blockscope] Menu screen up ("
                    + client.currentScreen.getClass().getName() + ") — auto-connecting shortly");
            }
            return;
        }

        autoConnectAttempted = true;
        String address = host + ":" + cfg.autoconnectPort;
        System.out.println("[Blockscope] Auto-connecting to " + address);
        try {
            net.minecraft.client.network.ServerInfo server =
                new net.minecraft.client.network.ServerInfo("blockscope-autoconnect", address, false);
            net.minecraft.client.gui.screen.ConnectScreen.connect(
                client.currentScreen, client,
                net.minecraft.client.network.ServerAddress.parse(address), server);
        } catch (Exception e) {
            System.err.println("[Blockscope] Auto-connect failed: " + e.getMessage());
        }
    }
}
