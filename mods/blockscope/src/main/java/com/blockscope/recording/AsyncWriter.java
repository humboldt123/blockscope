package com.blockscope.recording;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

public class AsyncWriter {
    private static final int FLUSH_INTERVAL_MS = 5000;
    private static final int FLUSH_BATCH_SIZE = 200;

    private String sessionId;
    private String serverUrl;

    private final BlockingQueue<String> tickQueue = new LinkedBlockingQueue<>(100000);
    private final BlockingQueue<String> inputQueue = new LinkedBlockingQueue<>(10000);
    private final BlockingQueue<String> frameQueue = new LinkedBlockingQueue<>(100000);

    private final AtomicBoolean running = new AtomicBoolean(false);
    private Thread workerThread;

    public void start(String sessionId, String serverUrl) {
        this.sessionId = sessionId;
        this.serverUrl = serverUrl;
        running.set(true);
        workerThread = new Thread(this::processQueues, "Blockscope-AsyncWriter");
        workerThread.setDaemon(true);
        workerThread.start();
    }

    public void writeTick(String json) {
        if (!tickQueue.offer(json)) {
            System.err.println("[Blockscope] Tick queue full, dropping tick!");
        }
    }

    public void writeInput(String json) {
        if (!inputQueue.offer(json)) {
            System.err.println("[Blockscope] Input queue full, dropping input!");
        }
    }

    public void writeFrameMapping(String json) {
        if (!frameQueue.offer(json)) {
            System.err.println("[Blockscope] Frame queue full, dropping frame mapping!");
        }
    }

    public void stop() {
        running.set(false);
        if (workerThread != null) {
            workerThread.interrupt();
            try {
                workerThread.join(10000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        flushQueue(tickQueue, "ticks");
        flushQueue(inputQueue, "inputs");
        flushQueue(frameQueue, "frame_mapping");
    }

    private void processQueues() {
        while (running.get()) {
            try {
                if (tickQueue.size() >= FLUSH_BATCH_SIZE) flushQueue(tickQueue, "ticks");
                if (inputQueue.size() >= FLUSH_BATCH_SIZE) flushQueue(inputQueue, "inputs");
                if (frameQueue.size() >= FLUSH_BATCH_SIZE) flushQueue(frameQueue, "frame_mapping");
                Thread.sleep(FLUSH_INTERVAL_MS);
                flushQueue(tickQueue, "ticks");
                flushQueue(inputQueue, "inputs");
                flushQueue(frameQueue, "frame_mapping");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    private void flushQueue(BlockingQueue<String> queue, String dataType) {
        if (queue.isEmpty()) return;
        List<String> batch = new ArrayList<>();
        queue.drainTo(batch);
        if (batch.isEmpty()) return;
        StringBuilder sb = new StringBuilder();
        for (String line : batch) sb.append(line).append('\n');
        sendToServer(dataType, sb.toString().getBytes(StandardCharsets.UTF_8));
    }

    private void sendToServer(String dataType, byte[] data) {
        try {
            String url = serverUrl + "/stream-data?session_id=" + sessionId + "&data_type=" + dataType;
            HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
            conn.setConnectTimeout(2000);
            conn.setReadTimeout(5000);
            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/x-ndjson");
            conn.setRequestProperty("Content-Length", String.valueOf(data.length));
            try (OutputStream out = conn.getOutputStream()) {
                out.write(data);
            }
            int code = conn.getResponseCode();
            if (code != 200) System.err.println("[Blockscope] Stream " + dataType + " returned " + code);
            conn.disconnect();
        } catch (Exception e) {
            System.err.println("[Blockscope] Failed to stream " + dataType + ": " + e.getMessage());
        }
    }
}
