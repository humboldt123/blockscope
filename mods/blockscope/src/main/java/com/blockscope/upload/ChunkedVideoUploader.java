package com.blockscope.upload;

import net.minecraft.client.MinecraftClient;
import net.minecraft.text.Text;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Uploads video chunks as they're being encoded (true streaming).
 * Sends chunks to server in real-time, no local file storage.
 */
public class ChunkedVideoUploader {
    private static final int CHUNK_SIZE = 256 * 1024; // 256KB chunks

    private final String sessionId;
    private final String serverUrl;
    private HttpURLConnection connection;
    private OutputStream outputStream;
    private ByteArrayOutputStream chunkBuffer;
    private long totalBytes = 0;
    private int chunkCount = 0;
    private boolean started = false;

    public ChunkedVideoUploader(String sessionId, String serverUrl) {
        this.sessionId = sessionId;
        this.serverUrl = serverUrl;
        this.chunkBuffer = new ByteArrayOutputStream(CHUNK_SIZE);
    }

    /**
     * Start streaming connection
     */
    public void start() throws IOException {
        if (started) return;

        URL url = new URL(serverUrl + "/stream-video?session_id=" + sessionId);
        System.out.println("[ChunkedUpload] Connecting to: " + url);

        connection = (HttpURLConnection) url.openConnection();
        connection.setDoOutput(true);
        connection.setRequestMethod("POST");
        connection.setRequestProperty("Content-Type", "application/octet-stream"); // Raw H264 stream
        connection.setRequestProperty("X-Session-ID", sessionId);
        connection.setChunkedStreamingMode(CHUNK_SIZE);
        connection.setRequestProperty("User-Agent", "Blockscope/1.0");
        connection.setConnectTimeout(30000); // 30 second timeout
        connection.setReadTimeout(300000);   // 5 minute timeout

        outputStream = connection.getOutputStream();
        started = true;

        System.out.println("[ChunkedUpload] Started streaming for session: " + sessionId);
    }

    /**
     * Write video data (buffers and sends in chunks)
     */
    public void write(byte[] data) throws IOException {
        if (!started) {
            throw new IllegalStateException("Uploader not started");
        }

        try {
            chunkBuffer.write(data);
            totalBytes += data.length;

            // Send chunk when buffer reaches size
            if (chunkBuffer.size() >= CHUNK_SIZE) {
                flushChunk();
            }
        } catch (IOException e) {
            System.err.println("[ChunkedUpload] Write error: " + e.getMessage());
            throw e;
        }
    }

    /**
     * Flush current chunk to server
     */
    public void flushChunk() throws IOException {
        if (chunkBuffer.size() == 0) return;

        byte[] chunk = chunkBuffer.toByteArray();
        outputStream.write(chunk);
        outputStream.flush();

        chunkCount++;
        chunkBuffer.reset();

        // Log every 10 chunks
        if (chunkCount % 10 == 0) {
            logProgress();
        }
    }

    /**
     * Finish streaming and get response
     */
    public void finish() throws IOException {
        if (!started) return;

        try {
            // Flush any remaining data
            flushChunk();

            outputStream.close();

            int responseCode = connection.getResponseCode();

            if (responseCode == 200 || responseCode == 201) {
                System.out.println("[ChunkedUpload] Complete: " + chunkCount + " chunks, " +
                    (totalBytes / 1024 / 1024) + " MB, HTTP " + responseCode);
                logToChat("§a[Upload] Video streamed: " + (totalBytes / 1024 / 1024) + " MB");
            } else {
                String error = "HTTP " + responseCode + ": " + connection.getResponseMessage();
                System.err.println("[ChunkedUpload] Failed: " + error);
                logToChat("§c[Upload] Failed: " + error);
            }
        } finally {
            connection.disconnect();
            started = false;
        }
    }

    private void logProgress() {
        System.out.println("[ChunkedUpload] Progress: " + chunkCount + " chunks, " +
            (totalBytes / 1024 / 1024) + " MB uploaded");
    }

    private void logToChat(String message) {
        MinecraftClient client = MinecraftClient.getInstance();
        if (client != null && client.player != null) {
            client.execute(() -> {
                if (client.player != null) {
                    client.player.sendMessage(Text.literal(message), false);
                }
            });
        }
    }

    public long getTotalBytes() {
        return totalBytes;
    }

    public int getChunkCount() {
        return chunkCount;
    }
}
