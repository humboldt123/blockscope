package com.blockscope.recording;

import com.blockscope.util.Config;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.util.Window;
import org.lwjgl.opengl.GL11;

import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Iterator;

public class FrameCapture {
    private final Config config;
    private int captureWidth;
    private int captureHeight;
    private int offsetX;
    private int offsetY;

    public FrameCapture(Config config) {
        this.config = config;
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

    public byte[] captureFrame() {
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
                return null;
            }

            // Check for GL errors
            int error = GL11.glGetError();
            if (error != GL11.GL_NO_ERROR) {
                System.err.println("[Blockscope] OpenGL error after glReadPixels: " + error);
                return null;
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

            // Create BufferedImage with CAPTURED dimensions (not target dimensions)
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

            // Encode to PNG with fast compression (optimization for performance)
            ByteArrayOutputStream baos = new ByteArrayOutputStream();

            // Use ImageWriter with compression settings for faster encoding
            Iterator<ImageWriter> writers = ImageIO.getImageWritersByFormatName("PNG");
            if (writers.hasNext()) {
                ImageWriter writer = writers.next();
                ImageWriteParam writeParam = writer.getDefaultWriteParam();

                try (ImageOutputStream ios = ImageIO.createImageOutputStream(baos)) {
                    writer.setOutput(ios);
                    // Compression mode MODE_DEFAULT is fastest
                    if (writeParam.canWriteCompressed()) {
                        writeParam.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
                        writeParam.setCompressionQuality(0.75f); // Slightly lower quality for speed
                    }
                    writer.write(null, new javax.imageio.IIOImage(image, null, null), writeParam);
                } finally {
                    writer.dispose();
                }
            } else {
                // Fallback to default encoding
                ImageIO.write(image, "PNG", baos);
            }

            return baos.toByteArray();

        } catch (IOException e) {
            System.err.println("[Blockscope] Frame capture failed: " + e.getMessage());
            return null;
        }
    }
}
