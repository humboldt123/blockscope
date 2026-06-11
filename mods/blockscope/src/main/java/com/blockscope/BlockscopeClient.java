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
        ClientPlayConnectionEvents.DISCONNECT.register((handler, client) -> {
            if (RecordingManager.getInstance().isRecording()) {
                System.out.println("[Blockscope] Disconnect — stopping recording");
                RecordingManager.getInstance().stopRecording();
            }
            if (baritonePresent && BotModule.getInstance().isRunning()) {
                BotModule.getInstance().stop();
            }
            SessionProtocol.onDisconnect(client);
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
}
