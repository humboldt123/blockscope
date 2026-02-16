package com.blockscope.gui;

import com.blockscope.recording.RecordingManager;
import com.blockscope.util.Config;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawableHelper;
import net.minecraft.client.util.math.MatrixStack;

public class RecordingOverlay extends DrawableHelper {
    private static final RecordingOverlay INSTANCE = new RecordingOverlay();

    private RecordingOverlay() {}

    public static RecordingOverlay getInstance() {
        return INSTANCE;
    }

    public void render(MatrixStack matrices) {
        RecordingManager manager = RecordingManager.getInstance();
        Config config = Config.getInstance();

        // Check if overlay is enabled and recording is active
        if (!config.showRecordingOverlay || !manager.isRecording()) {
            return;
        }

        MinecraftClient client = MinecraftClient.getInstance();

        // Draw recording indicator (red dot + text)
        int x = 10;
        int y = 10;

        // Red recording dot (pulsing)
        long time = System.currentTimeMillis();
        int alpha = (int) (128 + 127 * Math.sin(time / 500.0));
        int color = (alpha << 24) | 0xFF0000;

        fill(matrices, x, y, x + 8, y + 8, color);

        // Recording text
        String recordingText = "REC";
        client.textRenderer.drawWithShadow(matrices, recordingText, x + 12, y, 0xFF0000);

        // Session info
        String sessionInfo = String.format("Tick: %d | Frames: %d",
            manager.getCurrentTick(),
            manager.getFrameCount());

        client.textRenderer.drawWithShadow(matrices, sessionInfo, x, y + 12, 0xFFFFFF);
    }
}
