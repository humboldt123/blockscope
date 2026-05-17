package com.blockscope;

import com.blockscope.recording.RecordingManager;
import com.replaymod.core.ReplayMod;
import com.replaymod.recording.ReplayModRecording;
import com.replaymod.recording.packet.PacketListener;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayNetworkHandler;
import net.minecraft.network.ClientConnection;

/**
 * Hooks into ReplayMod's recording lifecycle by polling PacketListener each tick.
 * PacketListener is non-null while ReplayMod is actively recording.
 */
public class ReplayModIntegration {

    private boolean wasRecording = false;

    public void register() {
        // Fallback: if the watcher from stopRecording() didn't run (e.g. recording was
        // never explicitly stopped), start it now on world exit.
        ClientPlayConnectionEvents.DISCONNECT.register((handler, client) ->
            RecordingManager.getInstance().startMcprWatcher()
        );

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            boolean isRecording = isReplayModRecording();

            if (isRecording && !wasRecording) {
                onRecordingStarted(client);
            } else if (!isRecording && wasRecording) {
                onRecordingStopped(client);
            }

            if (isRecording) {
                RecordingManager.getInstance().onTick(client);
            }

            wasRecording = isRecording;
        });
    }

    private boolean isReplayModRecording() {
        try {
            ReplayModRecording mod = ReplayModRecording.instance;
            if (mod == null) return false;
            PacketListener listener = mod.getConnectionEventHandler().getPacketListener();
            return listener != null;
        } catch (Exception e) {
            return false;
        }
    }

    private void onRecordingStarted(MinecraftClient client) {
        try {
            String replayFolder = ReplayMod.instance.folders.getReplayFolder().toAbsolutePath().toString();
            RecordingManager.getInstance().startRecording(replayFolder);
        } catch (Exception e) {
            System.err.println("[Blockscope] Failed to start companion recording: " + e.getMessage());
        }
    }

    /**
     * Called from the keybinding handler. Starts ReplayMod recording if idle,
     * or stops it if already recording. Our tick poll picks up the state change.
     */
    public void toggle() {
        try {
            ReplayModRecording mod = ReplayModRecording.instance;
            if (mod == null) return;
            PacketListener listener = mod.getConnectionEventHandler().getPacketListener();
            if (listener == null) {
                // Not recording — start it
                ClientPlayNetworkHandler networkHandler = MinecraftClient.getInstance().getNetworkHandler();
                if (networkHandler != null) {
                    ClientConnection connection = networkHandler.getConnection();
                    mod.initiateRecording(connection);
                }
            } else {
                // Recording — stop it
                mod.getConnectionEventHandler().reset();
            }
        } catch (Exception e) {
            System.err.println("[Blockscope] Failed to toggle ReplayMod recording: " + e.getMessage());
        }
    }

    private void onRecordingStopped(MinecraftClient client) {
        try {
            // Give ReplayMod a moment to finish writing the .mcpr
            new Thread(() -> {
                try {
                    Thread.sleep(3000);
                } catch (InterruptedException ignored) {}
                RecordingManager.getInstance().stopRecording();
            }, "blockscope-stop").start();
        } catch (Exception e) {
            System.err.println("[Blockscope] Failed to stop companion recording: " + e.getMessage());
        }
    }
}
