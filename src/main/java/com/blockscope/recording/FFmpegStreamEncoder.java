package com.blockscope.recording;

import com.blockscope.upload.ChunkedVideoUploader;
import com.blockscope.util.Config;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.util.Window;
import org.lwjgl.opengl.GL11;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;

/**
 * Encodes video by piping raw frames to ffmpeg process.
 * FFmpeg encodes to H264 and writes to stdout, which we stream to server.
 */
public class FFmpegStreamEncoder {
    private final Config config;
    private final ChunkedVideoUploader uploader;

    private int captureWidth;
    private int captureHeight;
    private int offsetX;
    private int offsetY;
    private int frameCount;
    private long lastCaptureTime;
    private final long frameIntervalMs;

    private Process ffmpegProcess;
    private OutputStream ffmpegInput;
    private Thread outputReaderThread;

    public FFmpegStreamEncoder(Config config, ChunkedVideoUploader uploader) {
        this.config = config;
        this.uploader = uploader;
        this.frameIntervalMs = 1000L / config.targetFps;

        calculateCaptureDimensions();
    }

    private void calculateCaptureDimensions() {
        MinecraftClient client = MinecraftClient.getInstance();
        Window window = client.getWindow();

        int windowWidth = window.getFramebufferWidth();
        int windowHeight = window.getFramebufferHeight();

        float windowAspect = (float) windowWidth / windowHeight;
        float targetAspect = (float) config.resolutionWidth / config.resolutionHeight;

        if (config.aspectRatioMode.equals("preserve")) {
            if (windowAspect > targetAspect) {
                captureHeight = windowHeight;
                captureWidth = (int) (windowHeight * targetAspect);
                offsetX = (windowWidth - captureWidth) / 2;
                offsetY = 0;
            } else {
                captureWidth = windowWidth;
                captureHeight = (int) (windowWidth / targetAspect);
                offsetX = 0;
                offsetY = (windowHeight - captureHeight) / 2;
            }
        } else {
            captureWidth = windowWidth;
            captureHeight = windowHeight;
            offsetX = 0;
            offsetY = 0;
        }
    }

    public void startRecording() throws IOException {
        this.frameCount = 0;
        this.lastCaptureTime = System.currentTimeMillis();

        // Start ffmpeg process
        ProcessBuilder pb = new ProcessBuilder(
            "ffmpeg",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", captureWidth + "x" + captureHeight,
            "-framerate", String.valueOf(config.targetFps),
            "-i", "pipe:0",  // Read from stdin
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", "2M",
            "-pix_fmt", "yuv420p",
            "-f", "h264",  // Raw H264 output
            "pipe:1"  // Write to stdout
        );

        pb.redirectError(ProcessBuilder.Redirect.INHERIT);
        ffmpegProcess = pb.start();

        ffmpegInput = ffmpegProcess.getOutputStream();

        // Start thread to read ffmpeg output and stream to server
        outputReaderThread = new Thread(() -> {
            try {
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = ffmpegProcess.getInputStream().read(buffer)) != -1) {
                    byte[] data = new byte[bytesRead];
                    System.arraycopy(buffer, 0, data, 0, bytesRead);
                    uploader.write(data);
                }
            } catch (IOException e) {
                System.err.println("[FFmpegEncoder] Output reader error: " + e.getMessage());
            }
        }, "FFmpegOutputReader");
        outputReaderThread.setDaemon(true);
        outputReaderThread.start();

        System.out.println("[FFmpegEncoder] Started: " + config.resolutionWidth + "x" +
            config.resolutionHeight + " @ " + config.targetFps + " FPS");
    }

    public boolean captureAndEncodeFrame() {
        // Frame rate limiting
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastCaptureTime < frameIntervalMs) {
            return false;
        }
        lastCaptureTime = currentTime;

        try {
            calculateCaptureDimensions();

            // Capture frame from OpenGL
            GL11.glPixelStorei(GL11.GL_PACK_ALIGNMENT, 1);
            GL11.glFlush();

            ByteBuffer glBuffer = ByteBuffer.allocateDirect(captureWidth * captureHeight * 3);

            try {
                GL11.glReadPixels(offsetX, offsetY, captureWidth, captureHeight,
                    GL11.GL_RGB, GL11.GL_UNSIGNED_BYTE, glBuffer);
            } catch (Exception e) {
                System.err.println("[FFmpegEncoder] glReadPixels failed: " + e.getMessage());
                return false;
            }

            int error = GL11.glGetError();
            if (error != GL11.GL_NO_ERROR) {
                return false;
            }

            // Flip vertically (OpenGL bottom-left to top-left)
            byte[] flipped = new byte[captureWidth * captureHeight * 3];
            byte[] row = new byte[captureWidth * 3];

            for (int y = 0; y < captureHeight; y++) {
                glBuffer.position((captureHeight - 1 - y) * captureWidth * 3);
                glBuffer.get(row);
                System.arraycopy(row, 0, flipped, y * captureWidth * 3, row.length);
            }

            // Write raw RGB frame to ffmpeg stdin
            ffmpegInput.write(flipped);
            ffmpegInput.flush();

            frameCount++;

            if (frameCount == 1 || frameCount % 100 == 0) {
                System.out.println("[FFmpegEncoder] Frame " + frameCount + " captured");
            }

            return true;

        } catch (Exception e) {
            System.err.println("[FFmpegEncoder] Encoding failed: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    public void stopRecording() {
        try {
            if (ffmpegInput != null) {
                ffmpegInput.close();
            }

            if (ffmpegProcess != null) {
                ffmpegProcess.waitFor();
                ffmpegProcess.destroy();
            }

            if (outputReaderThread != null) {
                outputReaderThread.join(5000);
            }

            System.out.println("[FFmpegEncoder] Finished: " + frameCount + " frames");
        } catch (Exception e) {
            System.err.println("[FFmpegEncoder] Error stopping: " + e.getMessage());
        }
    }

    public int getFrameCount() {
        return frameCount;
    }
}
