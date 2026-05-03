package com.blockscope.recording;

import com.blockscope.upload.ChunkedVideoUploader;
import com.blockscope.util.Config;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.util.Window;
import org.jcodec.codecs.h264.H264Encoder;
import org.jcodec.codecs.h264.H264Utils;
import org.jcodec.codecs.h264.encode.H264FixedRateControl;
import org.jcodec.common.model.ColorSpace;
import org.jcodec.common.model.Picture;
import org.lwjgl.opengl.GL11;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Encodes raw H.264 frames and streams them immediately.
 * No container format - just raw NAL units.
 * Server will remux to MP4 after recording completes.
 */
public class RawH264StreamEncoder {
    private final Config config;
    private final ChunkedVideoUploader uploader;
    private final H264Encoder encoder;

    private int captureWidth;
    private int captureHeight;
    private int offsetX;
    private int offsetY;
    private int frameCount;
    private long lastCaptureTime;
    private final long frameIntervalMs;

    private boolean headersSent = false;

    public RawH264StreamEncoder(Config config, ChunkedVideoUploader uploader) {
        this.config = config;
        this.uploader = uploader;
        this.frameIntervalMs = 1000L / config.targetFps;

        // Initialize H264 encoder with fixed bitrate
        this.encoder = new H264Encoder(new H264FixedRateControl(2048)); // 2 Mbps

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
        this.headersSent = false;

        System.out.println("[RawH264] Started: " + config.resolutionWidth + "x" +
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
                System.err.println("[RawH264] glReadPixels failed: " + e.getMessage());
                return false;
            }

            int error = GL11.glGetError();
            if (error != GL11.GL_NO_ERROR) {
                return false;
            }

            // Convert to BufferedImage and scale
            int[] pixels = new int[captureWidth * captureHeight];
            for (int y = 0; y < captureHeight; y++) {
                for (int x = 0; x < captureWidth; x++) {
                    int srcIndex = ((captureHeight - 1 - y) * captureWidth + x) * 3;
                    int r = glBuffer.get(srcIndex) & 0xFF;
                    int g = glBuffer.get(srcIndex + 1) & 0xFF;
                    int b = glBuffer.get(srcIndex + 2) & 0xFF;
                    pixels[y * captureWidth + x] = (r << 16) | (g << 8) | b;
                }
            }

            BufferedImage image = new BufferedImage(captureWidth, captureHeight, BufferedImage.TYPE_INT_RGB);
            image.setRGB(0, 0, captureWidth, captureHeight, pixels, 0, captureWidth);

            // Scale if needed
            if (captureWidth != config.resolutionWidth || captureHeight != config.resolutionHeight) {
                BufferedImage scaled = new BufferedImage(
                    config.resolutionWidth, config.resolutionHeight, BufferedImage.TYPE_INT_RGB);
                scaled.getGraphics().drawImage(image, 0, 0,
                    config.resolutionWidth, config.resolutionHeight, null);
                image = scaled;
            }

            // Convert to YUV420 Picture
            Picture pic = convertToYUV420(image);

            // Encode to H264
            ByteBuffer encodedBuffer = ByteBuffer.allocate(config.resolutionWidth * config.resolutionHeight);
            org.jcodec.common.VideoEncoder.EncodedFrame frame = encoder.encodeFrame(pic, encodedBuffer);

            if (frame != null && frame.getData() != null) {
                ByteBuffer outBuf = frame.getData();

                // H264 frames already contain SPS/PPS in keyframes - just stream everything
                if (outBuf.hasRemaining()) {
                    byte[] frameData = new byte[outBuf.remaining()];
                    outBuf.get(frameData);

                    try {
                        uploader.write(frameData);

                        // Log first frame and every 100th frame
                        if (frameCount == 0 || frameCount % 100 == 0) {
                            System.out.println("[RawH264] Frame " + frameCount + ": " + frameData.length + " bytes encoded");
                        }
                    } catch (Exception e) {
                        System.err.println("[RawH264] Failed to write frame " + frameCount + ": " + e.getMessage());
                        e.printStackTrace();
                        throw e;
                    }
                }

                frameCount++;
                return true;
            }

            return false;

        } catch (Exception e) {
            System.err.println("[RawH264] Encoding failed: " + e.getMessage());
            return false;
        }
    }

    private Picture convertToYUV420(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();

        Picture pic = Picture.create(width, height, ColorSpace.YUV420);

        int[] rgb = img.getRGB(0, 0, width, height, null, 0, width);

        byte[] y = pic.getPlaneData(0);
        byte[] u = pic.getPlaneData(1);
        byte[] v = pic.getPlaneData(2);

        int uvIdx = 0;

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                int pixel = rgb[j * width + i];
                int r = (pixel >> 16) & 0xff;
                int g = (pixel >> 8) & 0xff;
                int b = pixel & 0xff;

                int yVal = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                y[j * width + i] = (byte) Math.max(0, Math.min(255, yVal));

                if ((j % 2 == 0) && (i % 2 == 0)) {
                    int uVal = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                    int vVal = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                    u[uvIdx] = (byte) Math.max(0, Math.min(255, uVal));
                    v[uvIdx] = (byte) Math.max(0, Math.min(255, vVal));
                    uvIdx++;
                }
            }
        }

        return pic;
    }

    public void stopRecording() {
        System.out.println("[RawH264] Finished: " + frameCount + " frames");
    }

    public int getFrameCount() {
        return frameCount;
    }
}
