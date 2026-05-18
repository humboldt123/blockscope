package com.blockscope.recording;

import com.blockscope.util.Config;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.util.Window;
import org.lwjgl.opengl.GL11;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Records video as MPEG-TS segments (10 seconds each).
 * Each segment is uploaded when complete and deleted immediately.
 * Minimal local storage - only current segment exists.
 */
public class SegmentedTSEncoder {
    private final Config config;
    private final String sessionId;
    private final String serverUrl;
    private final Path sessionDir;

    private int captureWidth;
    private int captureHeight;
    private int offsetX;
    private int offsetY;
    private int frameCount;
    private int segmentIndex;
    private long lastCaptureTime;
    private final long frameIntervalMs;
    private final int framesPerSegment; // 10 seconds of frames
    private int lastWindowWidth;
    private int lastWindowHeight;
    private boolean dimensionsChanged;
    private int framesInCurrentSegment;

    private Process ffmpegProcess;
    private OutputStream ffmpegInput;
    private Path currentSegmentPath;
    private Thread outputReaderThread;
    private BlockingQueue<Path> segmentsToUpload;
    private Thread uploaderThread;
    private volatile boolean recording;

    public SegmentedTSEncoder(Config config, String sessionId, String serverUrl, Path sessionDir) {
        this.config = config;
        this.sessionId = sessionId;
        this.serverUrl = serverUrl;
        this.sessionDir = sessionDir;
        this.frameIntervalMs = 1000L / config.targetFps;
        this.framesPerSegment = config.targetFps * 10; // 10-second segments
        this.segmentsToUpload = new LinkedBlockingQueue<>();
        this.segmentIndex = 0;
        this.lastWindowWidth = 0;
        this.lastWindowHeight = 0;
        this.dimensionsChanged = false;
        this.framesInCurrentSegment = 0;

        // Calculate initial dimensions
        calculateCaptureDimensions();
        System.out.println("[SegmentedTS] Initial capture dimensions: " + captureWidth + "x" + captureHeight +
                           " (will scale to " + config.resolutionWidth + "x" + config.resolutionHeight + ")");
    }

    private void calculateCaptureDimensions() {
        MinecraftClient client = MinecraftClient.getInstance();
        Window window = client.getWindow();

        // Always capture full framebuffer - ffmpeg will scale to target resolution
        int windowWidth = window.getFramebufferWidth();
        int windowHeight = window.getFramebufferHeight();

        // H264 requires even dimensions - round down to nearest even number
        captureWidth = (windowWidth / 2) * 2;
        captureHeight = (windowHeight / 2) * 2;
        offsetX = 0;
        offsetY = 0;
    }

    public void startRecording() throws IOException {
        this.frameCount = 0;
        this.segmentIndex = 0;
        this.lastCaptureTime = System.currentTimeMillis();
        this.recording = true;

        // Resolve ffmpeg before touching anything else — fail loud if missing
        config.ffmpegPath = resolveFfmpeg(config.ffmpegPath);

        // Start uploader thread
        uploaderThread = new Thread(this::uploadWorker, "SegmentUploader");
        uploaderThread.setDaemon(true);
        uploaderThread.start();

        // Start first segment
        startNewSegment();

        System.out.println("[SegmentedTS] Started: " + config.resolutionWidth + "x" +
            config.resolutionHeight + " @ " + config.targetFps + " FPS, 10-sec segments");
    }

    /**
     * Resolves the ffmpeg binary. If the configured value is a bare name ("ffmpeg"),
     * probes common install locations so it works out-of-the-box on Mac and Linux
     * without requiring the user to set ffmpeg_path manually.
     */
    private static String resolveFfmpeg(String configured) throws IOException {
        // Absolute path — just verify it exists
        if (configured.contains("/") || configured.contains("\\")) {
            if (new java.io.File(configured).canExecute()) return configured;
            throw new IOException(
                "ffmpeg not found at '" + configured + "'. " +
                "Fix ffmpeg_path in blockscope.properties or install ffmpeg.");
        }
        // Bare name — probe standard locations before giving up
        boolean isWindows = System.getProperty("os.name", "").toLowerCase().contains("win");
        String[] candidates = isWindows ? new String[]{
            "C:\\ffmpeg\\bin\\ffmpeg.exe",                          // common manual install
            "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
            System.getenv("USERPROFILE") + "\\scoop\\shims\\ffmpeg.exe",  // scoop
            "C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe",         // chocolatey
        } : new String[]{
            "/opt/homebrew/bin/ffmpeg",   // Mac ARM (Apple Silicon)
            "/usr/local/bin/ffmpeg",       // Mac Intel / Linux manual install
            "/usr/bin/ffmpeg",             // Linux distro package
            "/snap/bin/ffmpeg",            // Linux snap
        };
        for (String path : candidates) {
            if (path != null && new java.io.File(path).canExecute()) {
                System.out.println("[SegmentedTS] ffmpeg resolved to: " + path);
                return path;
            }
        }
        // Last resort: try running it — works if ffmpeg is anywhere on PATH
        try {
            Process p = new ProcessBuilder(configured, "-version").start();
            p.waitFor();
            return configured;
        } catch (Exception e) {
            String install = isWindows
                ? "winget install ffmpeg  /  choco install ffmpeg  /  scoop install ffmpeg"
                : "brew install ffmpeg  /  apt install ffmpeg";
            throw new IOException(
                "ffmpeg not found. Install it (" + install + ") " +
                "or set ffmpeg_path=/full/path/to/ffmpeg in blockscope.properties.");
        }
    }

    private void startNewSegment() throws IOException {
        // Finalize previous segment if exists
        if (ffmpegProcess != null) {
            finalizeCurrentSegment();
        }

        framesInCurrentSegment = 0;
        // Zero-padded so lexicographic sort == numeric sort (segment_0000000010 > segment_0000000009)
        currentSegmentPath = sessionDir.resolve(String.format("segment_%010d.ts", segmentIndex));
        segmentIndex++;

        // Start ffmpeg for this segment
        // Captures full framebuffer and scales to fixed target resolution (640x360)
        ProcessBuilder pb = new ProcessBuilder(
            config.ffmpegPath,
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", captureWidth + "x" + captureHeight,
            "-framerate", String.valueOf(config.targetFps),
            "-i", "pipe:0",
            "-vf", "scale=" + config.resolutionWidth + ":" + config.resolutionHeight,  // Always scale to target
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", "2M",
            "-pix_fmt", "yuv420p",
            "-f", "mpegts",  // MPEG-TS format
            "-y",
            currentSegmentPath.toString()
        );

        ffmpegProcess = pb.start();
        ffmpegInput = ffmpegProcess.getOutputStream();

        // Read ffmpeg stderr in background to prevent blocking
        outputReaderThread = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(ffmpegProcess.getErrorStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    System.err.println("[FFmpeg] " + line);
                }
            } catch (IOException e) {
                // Expected when process closes
            }
        }, "FFmpeg-stderr-reader");
        outputReaderThread.setDaemon(true);
        outputReaderThread.start();

        System.out.println("[SegmentedTS] Started segment " + (segmentIndex - 1));
    }

    private void finalizeCurrentSegment() {
        try {
            if (ffmpegInput != null) {
                ffmpegInput.close();
            }

            if (ffmpegProcess != null) {
                ffmpegProcess.waitFor();
                int exitCode = ffmpegProcess.exitValue();

                if (exitCode == 0 && Files.exists(currentSegmentPath)) {
                    long sizeMB = Files.size(currentSegmentPath) / (1024 * 1024);
                    System.out.println("[SegmentedTS] Finalized segment: " + currentSegmentPath.getFileName() +
                        " (" + sizeMB + " MB)");

                    // Queue for upload
                    segmentsToUpload.offer(currentSegmentPath);
                } else {
                    System.err.println("[SegmentedTS] FFmpeg failed with code: " + exitCode);
                }
            }
        } catch (Exception e) {
            System.err.println("[SegmentedTS] Error finalizing segment: " + e.getMessage());
        }
    }

    public boolean captureAndEncodeFrame() {
        // Check for window resize
        MinecraftClient client = MinecraftClient.getInstance();
        Window window = client.getWindow();
        int currentWindowWidth = window.getFramebufferWidth();
        int currentWindowHeight = window.getFramebufferHeight();

        if (currentWindowWidth != lastWindowWidth || currentWindowHeight != lastWindowHeight) {
            lastWindowWidth = currentWindowWidth;
            lastWindowHeight = currentWindowHeight;
            dimensionsChanged = true;
            int oldWidth = captureWidth;
            int oldHeight = captureHeight;
            calculateCaptureDimensions();
            System.out.println("[SegmentedTS] Window resized: " + oldWidth + "x" + oldHeight +
                               " -> " + captureWidth + "x" + captureHeight +
                               " (will start new segment after next frame)");
        }

        // Start new segment if dimensions changed (after previous segment has at least 1 frame)
        if (dimensionsChanged && framesInCurrentSegment > 0) {
            try {
                System.out.println("[SegmentedTS] Starting new segment due to resize");
                startNewSegment();
                dimensionsChanged = false;
            } catch (IOException e) {
                System.err.println("[SegmentedTS] Failed to start new segment: " + e.getMessage());
                return false;
            }
        }

        // Check if we need to start a new segment (time-based)
        if (frameCount > 0 && framesInCurrentSegment > 0 && framesInCurrentSegment % framesPerSegment == 0) {
            try {
                startNewSegment();
            } catch (IOException e) {
                System.err.println("[SegmentedTS] Failed to start new segment: " + e.getMessage());
                return false;
            }
        }

        // Frame rate limiting
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastCaptureTime < frameIntervalMs) {
            return false;
        }
        lastCaptureTime = currentTime;

        try {
            // Capture frame from OpenGL
            GL11.glPixelStorei(GL11.GL_PACK_ALIGNMENT, 1);
            GL11.glFlush();

            ByteBuffer glBuffer = ByteBuffer.allocateDirect(captureWidth * captureHeight * 3);

            try {
                GL11.glReadPixels(offsetX, offsetY, captureWidth, captureHeight,
                    GL11.GL_RGB, GL11.GL_UNSIGNED_BYTE, glBuffer);
            } catch (Exception e) {
                System.err.println("[SegmentedTS] glReadPixels failed: " + e.getMessage());
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
            framesInCurrentSegment++;

            if (frameCount == 1 || frameCount % 100 == 0) {
                System.out.println("[SegmentedTS] Frame " + frameCount + " captured");
            }

            return true;

        } catch (Exception e) {
            System.err.println("[SegmentedTS] Encoding failed: " + e.getMessage());
            return false;
        }
    }

    private void uploadWorker() {
        while (recording || !segmentsToUpload.isEmpty()) {
            try {
                Path segment = segmentsToUpload.poll();
                if (segment == null) {
                    Thread.sleep(100);
                    continue;
                }

                // Upload segment
                uploadSegment(segment);

                // Delete after successful upload
                Files.delete(segment);
                System.out.println("[SegmentedTS] Deleted local segment: " + segment.getFileName());

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                System.err.println("[SegmentedTS] Upload worker error: " + e.getMessage());
            }
        }
    }

    private void uploadSegment(Path segment) throws IOException {
        // Extract segment index from filename (segment_0.ts -> 0)
        String filename = segment.getFileName().toString();
        int segIdx = Integer.parseInt(filename.replaceAll("[^0-9]", ""));

        java.net.URL url = new java.net.URL(serverUrl + "/upload-segment?session_id=" + sessionId + "&segment_index=" + segIdx);
        java.net.HttpURLConnection conn = (java.net.HttpURLConnection) url.openConnection();

        try {
            conn.setDoOutput(true);
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "video/mp2t");
            conn.setRequestProperty("X-Segment-Name", segment.getFileName().toString());
            conn.setRequestProperty("User-Agent", "Blockscope/1.0");

            try (OutputStream out = conn.getOutputStream();
                 InputStream in = Files.newInputStream(segment)) {

                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = in.read(buffer)) != -1) {
                    out.write(buffer, 0, bytesRead);
                }
            }

            int responseCode = conn.getResponseCode();
            if (responseCode == 200 || responseCode == 201) {
                System.out.println("[SegmentedTS] Uploaded segment: " + segment.getFileName());
            } else {
                System.err.println("[SegmentedTS] Upload failed: HTTP " + responseCode);
            }
        } finally {
            conn.disconnect();
        }
    }

    public void stopRecording() {
        recording = false;

        try {
            // Finalize current segment
            finalizeCurrentSegment();

            if (ffmpegProcess != null) {
                ffmpegProcess.destroy();
            }

            // Wait for all segments to upload
            if (uploaderThread != null) {
                uploaderThread.join(30000); // 30 second timeout
            }

            System.out.println("[SegmentedTS] Finished: " + frameCount + " frames in " + segmentIndex + " segments");
        } catch (Exception e) {
            System.err.println("[SegmentedTS] Error stopping: " + e.getMessage());
        }
    }

    public int getFrameCount() {
        return frameCount;
    }
}
