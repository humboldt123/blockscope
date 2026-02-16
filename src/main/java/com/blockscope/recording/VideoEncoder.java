package com.blockscope.recording;

import com.blockscope.util.Config;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.util.Window;
import org.jcodec.api.SequenceEncoder;
import org.jcodec.common.io.NIOUtils;
import org.jcodec.common.model.ColorSpace;
import org.jcodec.common.model.Picture;
import org.jcodec.scale.AWTUtil;
import org.lwjgl.opengl.GL11;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

public class VideoEncoder {
    private final Config config;
    private int captureWidth;
    private int captureHeight;
    private int offsetX;
    private int offsetY;

    private SequenceEncoder encoder;
    private File videoFile;
    private int frameCount;
    private long lastCaptureTime;
    private final long frameIntervalMs;

    public VideoEncoder(Config config) {
        this.config = config;
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
            // Preserve aspect ratio by adding black bars
            if (windowAspect > targetAspect) {
                // Window is wider, add vertical bars
                captureHeight = windowHeight;
                captureWidth = (int) (windowHeight * targetAspect);
                offsetX = (windowWidth - captureWidth) / 2;
                offsetY = 0;
            } else {
                // Window is taller, add horizontal bars
                captureWidth = windowWidth;
                captureHeight = (int) (windowWidth / targetAspect);
                offsetX = 0;
                offsetY = (windowHeight - captureHeight) / 2;
            }
        } else if (config.aspectRatioMode.equals("crop")) {
            // Crop to fill target resolution
            captureWidth = windowWidth;
            captureHeight = windowHeight;
            offsetX = 0;
            offsetY = 0;
        } else { // stretch
            captureWidth = windowWidth;
            captureHeight = windowHeight;
            offsetX = 0;
            offsetY = 0;
        }
    }

    /**
     * Initialize the video encoder and start recording to the specified file.
     */
    public void startRecording(File outputFile) throws IOException {
        this.videoFile = outputFile;
        this.frameCount = 0;
        this.lastCaptureTime = System.currentTimeMillis();

        // Create parent directory if needed
        outputFile.getParentFile().mkdirs();

        // Initialize JCODEC SequenceEncoder
        // Rational.R() creates a framerate rational (fps, 1) = fps/1
        this.encoder = SequenceEncoder.createSequenceEncoder(
            outputFile,
            config.targetFps
        );

        System.out.println("[Blockscope] Video encoder started: " + outputFile.getName());
        System.out.println("[Blockscope] Target: " + config.resolutionWidth + "x" + config.resolutionHeight + " @ " + config.targetFps + " FPS");
    }

    /**
     * Capture and encode a single frame. Returns true if frame was captured, false if skipped.
     */
    public boolean captureAndEncodeFrame() {
        if (encoder == null) {
            return false;
        }

        // Frame rate limiting - only capture at target FPS
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastCaptureTime < frameIntervalMs) {
            return false; // Skip this frame
        }
        lastCaptureTime = currentTime;

        try {
            // Recalculate dimensions in case window was resized
            calculateCaptureDimensions();

            // Ensure all GL commands are finished before reading pixels
            GL11.glPixelStorei(GL11.GL_PACK_ALIGNMENT, 1);
            GL11.glFlush();

            // Read pixels from framebuffer
            ByteBuffer buffer = ByteBuffer.allocateDirect(captureWidth * captureHeight * 3);

            // Use try-catch to prevent driver crashes from propagating
            try {
                GL11.glReadPixels(offsetX, offsetY, captureWidth, captureHeight,
                    GL11.GL_RGB, GL11.GL_UNSIGNED_BYTE, buffer);
            } catch (Exception e) {
                System.err.println("[Blockscope] glReadPixels failed: " + e.getMessage());
                return false;
            }

            // Check for GL errors
            int error = GL11.glGetError();
            if (error != GL11.GL_NO_ERROR) {
                System.err.println("[Blockscope] OpenGL error after glReadPixels: " + error);
                return false;
            }

            // Flip vertically (OpenGL origin is bottom-left) and convert to int array
            int[] pixels = new int[captureWidth * captureHeight];
            for (int y = 0; y < captureHeight; y++) {
                for (int x = 0; x < captureWidth; x++) {
                    int srcIndex = ((captureHeight - 1 - y) * captureWidth + x) * 3;
                    int r = buffer.get(srcIndex) & 0xFF;
                    int g = buffer.get(srcIndex + 1) & 0xFF;
                    int b = buffer.get(srcIndex + 2) & 0xFF;
                    pixels[y * captureWidth + x] = (r << 16) | (g << 8) | b;
                }
            }

            // Create BufferedImage with CAPTURED dimensions
            BufferedImage image = new BufferedImage(
                captureWidth,
                captureHeight,
                BufferedImage.TYPE_INT_RGB);

            // Set pixels to image
            image.setRGB(0, 0, captureWidth, captureHeight, pixels, 0, captureWidth);

            // Scale to target resolution if needed
            if (captureWidth != config.resolutionWidth || captureHeight != config.resolutionHeight) {
                BufferedImage scaledImage = new BufferedImage(
                    config.resolutionWidth,
                    config.resolutionHeight,
                    BufferedImage.TYPE_INT_RGB);
                scaledImage.getGraphics().drawImage(image,
                    0, 0, config.resolutionWidth, config.resolutionHeight, null);
                image = scaledImage;
            }

            // Convert BufferedImage to JCODEC Picture format (RGB color space)
            Picture picture = AWTUtil.fromBufferedImage(image, ColorSpace.RGB);

            // Encode frame directly to video
            encoder.encodeNativeFrame(picture);
            frameCount++;

            return true;

        } catch (IOException e) {
            System.err.println("[Blockscope] Frame encoding failed: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Finalize the video file and close the encoder.
     */
    public void stopRecording() {
        if (encoder != null) {
            try {
                encoder.finish();
                System.out.println("[Blockscope] Video encoding finished: " + frameCount + " frames written to " + videoFile.getName());
            } catch (IOException e) {
                System.err.println("[Blockscope] Error finalizing video: " + e.getMessage());
                e.printStackTrace();
            } finally {
                encoder = null;
            }
        }
    }

    /**
     * Get the current frame count.
     */
    public int getFrameCount() {
        return frameCount;
    }

    /**
     * Check if encoder is currently active.
     */
    public boolean isRecording() {
        return encoder != null;
    }
}
