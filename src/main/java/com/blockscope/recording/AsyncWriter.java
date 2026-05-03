package com.blockscope.recording;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

public class AsyncWriter {
    // Increased from 1000 to 10000 to prevent queue overflow during heavy recording
    private final BlockingQueue<WriteTask> queue = new LinkedBlockingQueue<>(10000);
    private final AtomicBoolean running = new AtomicBoolean(false);
    private Thread writerThread;

    public void start() {
        if (running.get()) {
            return;
        }

        running.set(true);
        writerThread = new Thread(this::processQueue, "Blockscope-AsyncWriter");
        writerThread.setDaemon(true);
        writerThread.start();
    }

    public void stop() {
        if (!running.get()) {
            return;
        }

        running.set(false);

        try {
            writerThread.join(5000); // Wait up to 5 seconds for queue to flush
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public void writeJsonLine(Path filePath, String jsonLine) {
        if (!running.get()) {
            return;
        }

        WriteTask task = new WriteTask(filePath, (jsonLine + "\n").getBytes(), false);

        if (!queue.offer(task)) {
            System.err.println("[Blockscope] Write queue full, dropping data!");
        }
    }

    public void writeBytes(Path filePath, byte[] data, boolean overwrite) {
        if (!running.get()) {
            return;
        }

        WriteTask task = new WriteTask(filePath, data, overwrite);

        if (!queue.offer(task)) {
            System.err.println("[Blockscope] Write queue full, dropping frame!");
        }
    }

    /**
     * Write data directly to file, bypassing the queue.
     * Use this for large snapshots to prevent queue overflow.
     * This is a blocking operation.
     */
    public void writeDirectly(Path filePath, String jsonLine) {
        try {
            Files.createDirectories(filePath.getParent());
            Files.write(filePath, (jsonLine + "\n").getBytes(),
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND);
        } catch (IOException e) {
            System.err.println("[Blockscope] Direct write failed: " + e.getMessage());
        }
    }

    private void processQueue() {
        while (running.get() || !queue.isEmpty()) {
            try {
                WriteTask task = queue.poll();
                if (task == null) {
                    Thread.sleep(10);
                    continue;
                }

                Files.createDirectories(task.filePath.getParent());

                if (task.overwrite) {
                    Files.write(task.filePath, task.data);
                } else {
                    Files.write(task.filePath, task.data,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.APPEND);
                }

            } catch (IOException e) {
                System.err.println("[Blockscope] Write failed: " + e.getMessage());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    private static class WriteTask {
        final Path filePath;
        final byte[] data;
        final boolean overwrite;

        WriteTask(Path filePath, byte[] data, boolean overwrite) {
            this.filePath = filePath;
            this.data = data;
            this.overwrite = overwrite;
        }
    }
}
