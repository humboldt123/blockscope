package com.blockscope.recording;

import java.awt.image.BufferedImage;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Background thread for video encoding to avoid blocking render thread.
 * Captures frames from main thread, encodes in background.
 */
public class VideoStreamingThread extends Thread {
    private final StreamingTSEncoder encoder;
    private final BlockingQueue<BufferedImage> frameQueue;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private int droppedFrames = 0;

    public VideoStreamingThread(StreamingTSEncoder encoder) {
        super("Blockscope-VideoStreaming");
        setDaemon(true);
        this.encoder = encoder;
        // Increased queue size to 15 frames to reduce frame drops
        // At 30fps, this is 0.5 seconds of buffer
        this.frameQueue = new LinkedBlockingQueue<>(15);
    }

    @Override
    public void run() {
        running.set(true);

        while (running.get() || !frameQueue.isEmpty()) {
            try {
                BufferedImage frame = frameQueue.poll();
                if (frame == null) {
                    Thread.sleep(5);
                    continue;
                }

                // Encode frame (blocks until done)
                encoder.encodeFrame(frame);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                System.err.println("[Blockscope] Video streaming error: " + e.getMessage());
            }
        }
    }

    /**
     * Submit frame for encoding (non-blocking).
     * Returns false if queue is full (frame dropped).
     */
    public boolean submitFrame(BufferedImage frame) {
        if (!running.get()) {
            return false;
        }

        boolean accepted = frameQueue.offer(frame);
        if (!accepted) {
            droppedFrames++;
            // Log every 30 dropped frames instead of 10 to reduce spam
            if (droppedFrames % 30 == 0) {
                System.err.println("[Blockscope] Warning: Dropped " + droppedFrames + " frames (encoding can't keep up with capture rate)");
            }
        }
        return accepted;
    }

    public void startEncoding() {
        if (!running.get()) {
            start();
        }
    }

    public void stopEncoding() {
        running.set(false);
        try {
            join(5000); // Wait up to 5 seconds
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        if (droppedFrames > 0) {
            System.out.println("[Blockscope] Total dropped frames: " + droppedFrames);
        }
    }

    public int getQueueSize() {
        return frameQueue.size();
    }
}
