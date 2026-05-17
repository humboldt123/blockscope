package com.blockscope.upload;

import net.minecraft.client.MinecraftClient;
import net.minecraft.text.Text;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

public class UploaderThread extends Thread {
    private static final int MAX_RETRIES = 3;
    private static final int RETRY_DELAY_MS = 5000;

    private final String sessionId;
    private final String serverUrl;
    private final Path ticksFile;
    private final Path inputsFile;
    private final Path videoFile;

    public UploaderThread(String sessionId, String serverUrl, Path ticksFile, Path inputsFile, Path videoFile) {
        super("UploaderThread-" + sessionId);
        setDaemon(true);
        this.sessionId = sessionId;
        this.serverUrl = serverUrl;
        this.ticksFile = ticksFile;
        this.inputsFile = inputsFile;
        this.videoFile = videoFile;
    }

    @Override
    public void run() {
        MinecraftClient client = MinecraftClient.getInstance();
        logToChat(client, "§e[Upload] Starting upload for session " + sessionId + "...");

        int attempt = 0;
        boolean success = false;

        while (attempt < MAX_RETRIES && !success) {
            attempt++;

            try {
                logToChat(client, "§e[Upload] Attempt " + attempt + "/" + MAX_RETRIES);
                uploadFiles();
                success = true;
                logToChat(client, "§a[Upload] Successfully uploaded session " + sessionId);
            } catch (Exception e) {
                System.err.println("[Upload] Attempt " + attempt + " failed: " + e.getMessage());
                e.printStackTrace();

                if (attempt < MAX_RETRIES) {
                    logToChat(client, "§c[Upload] Failed, retrying in " + (RETRY_DELAY_MS / 1000) + "s...");
                    try {
                        Thread.sleep(RETRY_DELAY_MS);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                } else {
                    logToChat(client, "§c[Upload] Failed after " + MAX_RETRIES + " attempts: " + e.getMessage());
                }
            }
        }
    }

    private void uploadFiles() throws IOException {
        String boundary = "----Blockscope" + UUID.randomUUID().toString().replace("-", "");
        URL url = new URL(serverUrl + "/upload");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();

        try {
            conn.setDoOutput(true);
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
            conn.setRequestProperty("User-Agent", "Blockscope/1.0");

            try (OutputStream out = conn.getOutputStream();
                 PrintWriter writer = new PrintWriter(new OutputStreamWriter(out, StandardCharsets.UTF_8), true)) {

                // Add session_id field
                writer.append("--").append(boundary).append("\r\n");
                writer.append("Content-Disposition: form-data; name=\"session_id\"\r\n");
                writer.append("Content-Type: text/plain; charset=UTF-8\r\n\r\n");
                writer.append(sessionId).append("\r\n");
                writer.flush();

                // Add ticks.jsonl file (field name is form parameter name)
                if (Files.exists(ticksFile)) {
                    addFilePart(writer, out, boundary, "ticks", ticksFile);
                }

                // Add inputs.jsonl file
                if (Files.exists(inputsFile)) {
                    addFilePart(writer, out, boundary, "inputs", inputsFile);
                }

                // Add video.mp4 file (only if not streaming - streaming uploads during recording)
                if (videoFile != null && Files.exists(videoFile)) {
                    addFilePart(writer, out, boundary, "video", videoFile);
                }

                // End multipart
                writer.append("--").append(boundary).append("--\r\n");
                writer.flush();
            }

            // Check response
            int responseCode = conn.getResponseCode();
            if (responseCode == 200 || responseCode == 201) {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        response.append(line);
                    }
                    System.out.println("[Upload] Server response: " + response.toString());
                }

                // Delete local files after successful upload to save disk space
                deleteLocalFiles();
            } else {
                throw new IOException("Server returned HTTP " + responseCode + ": " + conn.getResponseMessage());
            }
        } finally {
            conn.disconnect();
        }
    }

    private void deleteLocalFiles() {
        MinecraftClient client = MinecraftClient.getInstance();

        try {
            // Delete video file (largest file)
            if (videoFile != null && Files.exists(videoFile)) {
                long sizeMB = Files.size(videoFile) / (1024 * 1024);
                Files.delete(videoFile);
                logToChat(client, "§7[Upload] Deleted local video (" + sizeMB + " MB freed)");
            }

            // Optionally delete ticks.jsonl and inputs.jsonl too
            if (ticksFile != null && Files.exists(ticksFile)) {
                Files.delete(ticksFile);
            }
            if (inputsFile != null && Files.exists(inputsFile)) {
                Files.delete(inputsFile);
            }

            // Try to delete parent directory if empty
            try {
                Path sessionDir = videoFile.getParent();
                if (Files.exists(sessionDir) && Files.list(sessionDir).count() == 0) {
                    Files.delete(sessionDir);
                }
            } catch (Exception e) {
                // Ignore - directory not empty or other issue
            }
        } catch (IOException e) {
            System.err.println("[Upload] Failed to delete local files: " + e.getMessage());
        }
    }

    private void addFilePart(PrintWriter writer, OutputStream out, String boundary,
                             String fieldName, Path file) throws IOException {
        String fileName = file.getFileName().toString();
        String contentType = getContentType(fileName);

        writer.append("--").append(boundary).append("\r\n");
        writer.append("Content-Disposition: form-data; name=\"").append(fieldName)
              .append("\"; filename=\"").append(fileName).append("\"\r\n");
        writer.append("Content-Type: ").append(contentType).append("\r\n");
        writer.append("Content-Transfer-Encoding: binary\r\n\r\n");
        writer.flush();

        long fileSize = Files.size(file);
        long bytesWritten = 0;

        try (InputStream fileInput = Files.newInputStream(file)) {
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = fileInput.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
                bytesWritten += bytesRead;

                // Log progress for large files (video)
                if (fileSize > 1_000_000 && bytesWritten % 1_000_000 == 0) {
                    int percent = (int) ((bytesWritten * 100) / fileSize);
                    MinecraftClient client = MinecraftClient.getInstance();
                    logToChat(client, "§e[Upload] " + fileName + ": " + percent + "%");
                }
            }
        }

        out.flush();
        writer.append("\r\n");
        writer.flush();
    }

    private String getContentType(String fileName) {
        if (fileName.endsWith(".jsonl")) {
            return "application/jsonl";
        } else if (fileName.endsWith(".mp4")) {
            return "video/mp4";
        } else if (fileName.endsWith(".json")) {
            return "application/json";
        }
        return "application/octet-stream";
    }

    private void logToChat(MinecraftClient client, String message) {
        if (client != null && client.player != null) {
            client.execute(() -> {
                if (client.player != null) {
                    client.player.sendMessage(Text.literal(message), false);
                }
            });
        }
    }
}
