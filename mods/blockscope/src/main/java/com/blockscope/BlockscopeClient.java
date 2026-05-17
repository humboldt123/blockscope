package com.blockscope;

import com.blockscope.agent.InventoryChaosAgent;
import com.blockscope.gui.ConfigScreen;
import com.blockscope.recording.RecordingManager;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import org.lwjgl.glfw.GLFW;

public class BlockscopeClient implements ClientModInitializer {
    private static KeyBinding recordingToggleKey;
    private static KeyBinding configKey;
    private static KeyBinding chaosAgentKey;
    private static boolean autoJumpDisabled = false;
    private static ReplayModIntegration replayModIntegration;

    @Override
    public void onInitializeClient() {
        System.out.println("[Blockscope] Initializing...");

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
        });

        System.out.println("[Blockscope] Initialized successfully");
    }
}
