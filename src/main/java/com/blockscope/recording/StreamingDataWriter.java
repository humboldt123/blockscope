package com.blockscope.recording;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Streams ticks.jsonl and inputs.jsonl data to server in real-time.
 * Buffers lines and flushes periodically to avoid excessive HTTP requests.
 */
public class StreamingDataWriter {
    private final String sessionId;
    private final String serverUrl;
    private final String dataType; // "ticks" or "inputs"
    private final BlockingQueue<String> lineQueue;
    private final Thread writerThread;
    private volatile boolean running;

    private static final int FLUSH_INTERVAL_MS = 2000; // Flush every 2 seconds
    private static final int BUFFER_SIZE = 100; // Or flush after 100 lines

    public StreamingDataWriter(String sessionId, String serverUrl, String dataType) {
        this.sessionId = sessionId;
        this.serverUrl = serverUrl;
        this.dataType = dataType;
        this.lineQueue = new LinkedBlockingQueue<>();
        this.running = true;

        // Start background writer thread
        this.writerThread = new Thread(this::writerLoop, "StreamingDataWriter-" + dataType);
        this.writerThread.setDaemon(true);
        this.writerThread.start();
    }

    /**
     * Queue a line to be written (non-blocking)
     */
    public void writeLine(String jsonLine) {
        if (running) {
            lineQueue.offer(jsonLine);
        }
    }

    /**
     * Background thread that batches and sends lines
     */
    private void writerLoop() {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();

        long lastFlush = System.currentTimeMillis();

        while (running || !lineQueue.isEmpty()) {
            try {
                // Poll with timeout to check flush interval
                String line = lineQueue.poll();

                if (line != null) {
                    buffer.write(line.getBytes());
                    buffer.write('\n');
                }

                // Flush if buffer is large enough or enough time passed
                long now = System.currentTimeMillis();
                boolean shouldFlush = buffer.size() > 0 && (
                    lineQueue.size() >= BUFFER_SIZE ||
                    now - lastFlush >= FLUSH_INTERVAL_MS ||
                    !running
                );

                if (shouldFlush) {
                    flush(buffer.toByteArray());
                    buffer.reset();
                    lastFlush = now;
                }

                // Small sleep to batch more efficiently
                if (line == null) {
                    Thread.sleep(100);
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                System.err.println("[StreamingDataWriter] Error: " + e.getMessage());
            }
        }

        // Final flush
        if (buffer.size() > 0) {
            try {
                flush(buffer.toByteArray());
            } catch (IOException e) {
                System.err.println("[StreamingDataWriter] Final flush failed: " + e.getMessage());
            }
        }
    }

    /**
     * Send buffered data to server
     */
    private void flush(byte[] data) throws IOException {
        URL url = new URL(serverUrl + "/stream-data?session_id=" + sessionId + "&data_type=" + dataType);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();

        try {
            conn.setDoOutput(true);
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/jsonl");
            conn.setRequestProperty("User-Agent", "Blockscope/1.0");

            try (OutputStream out = conn.getOutputStream()) {
                out.write(data);
            }

            int responseCode = conn.getResponseCode();
            if (responseCode != 200 && responseCode != 201) {
                System.err.println("[StreamingDataWriter] HTTP " + responseCode + " for " + dataType);
            }
        } finally {
            conn.disconnect();
        }
    }

    /**
     * Stop writer and flush remaining data
     */
    public void stop() {
        running = false;
        try {
            writerThread.join(5000); // Wait up to 5 seconds
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
