package com.blockscope;

import com.blockscope.agent.InventoryChaosAgent;
import com.blockscope.gui.ConfigScreen;
import com.blockscope.gui.RecordingOverlay;
import com.blockscope.recording.RecordingManager;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientChunkEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import org.lwjgl.glfw.GLFW;

public class BlockscopeClient implements ClientModInitializer {
    private static KeyBinding recordingToggleKey;
    private static KeyBinding configKey;
    private static KeyBinding chaosAgentKey;
    private static boolean autoJumpDisabled = false;

    @Override
    public void onInitializeClient() {
        System.out.println("[Blockscope] Initializing...");

        // Add shutdown hook to stop recording on game close
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (RecordingManager.getInstance().isRecording()) {
                System.out.println("[Blockscope] Game closing - auto-stopping recording...");
                RecordingManager.getInstance().stopRecording();
            }
        }));

        // Phase 4.1: Register chunk load/unload events
        // These help track when chunks enter/exit view (useful for chunk transitions)
        ClientChunkEvents.CHUNK_LOAD.register((world, chunk) -> {
            // Chunk loaded - blocks will be captured via ChunkBuilder render on first visibility
            // No action needed here, just logging for debugging
            // System.out.println("[Blockscope] Chunk loaded: " + chunk.getPos());
        });

        ClientChunkEvents.CHUNK_UNLOAD.register((world, chunk) -> {
            // Chunk unloading - blocks remain in master map
            // If player returns, chunk rebuild will verify blocks haven't changed
            // System.out.println("[Blockscope] Chunk unloading: " + chunk.getPos());
        });

        // Register keybinding for recording toggle
        recordingToggleKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.blockscope.toggle_recording",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_R,
            "category.blockscope"
        ));

        // Register keybinding for config screen
        configKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.blockscope.config",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_B,
            "category.blockscope"
        ));

        // Register keybinding for chaos agent
        chaosAgentKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.blockscope.chaos_agent",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_C,
            "category.blockscope"
        ));

        // Register client tick event
        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            // Disable AutoJump once on first tick (for 1.8.9 compatibility)
            if (!autoJumpDisabled && client.options != null) {
                if (client.options.getAutoJump().getValue()) {
                    client.options.getAutoJump().setValue(false);
                    client.options.write();
                    System.out.println("[Blockscope] AutoJump disabled for 1.8.9 compatibility");
                }
                autoJumpDisabled = true;
            }

            // Handle recording toggle keybind
            if (recordingToggleKey.wasPressed()) {
                RecordingManager manager = RecordingManager.getInstance();
                if (manager.isRecording()) {
                    manager.stopRecording();
                } else {
                    manager.startRecording();
                }
            }

            // Handle config screen keybind
            if (configKey.wasPressed()) {
                client.setScreen(new ConfigScreen(client.currentScreen));
            }

            // Handle chaos agent keybind
            if (chaosAgentKey.wasPressed()) {
                System.out.println("[Blockscope] C key pressed - toggling chaos agent");
                InventoryChaosAgent.getInstance().toggle();
            }

            // Record tick data if recording
            RecordingManager.getInstance().onClientTick();
        });

        // Note: Overlay is now rendered in onRenderEnd() AFTER frame capture
        // This ensures the overlay is visible to the player but not included in recordings

        System.out.println("[Blockscope] Initialized successfully");
    }

    public static void onRenderEnd() {
        MinecraftClient client = MinecraftClient.getInstance();

        // Capture frame first (without overlay)
        RecordingManager.getInstance().onRenderTick();

        // Now render overlay AFTER frame capture (visible to player but not recorded)
        if (client.world != null) {
            RecordingOverlay.getInstance().render(new net.minecraft.client.util.math.MatrixStack());
        }
    }
}
