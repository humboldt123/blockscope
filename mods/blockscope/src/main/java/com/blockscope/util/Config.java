package com.blockscope.util;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;

public class Config {
    private static final String CONFIG_FILE = "blockscope.properties";

    public int resolutionWidth = 640;
    public int resolutionHeight = 360;
    public String aspectRatioMode = "preserve"; // preserve, crop, stretch
    public int targetFps = 20;
    public String recordingDirectory = "./recordings";
    public boolean showChatMessages = true; // Show recording start/stop messages in chat
    public boolean showRecordingOverlay = true; // Show recording indicator overlay

    // Upload server configuration (localhost for debugging)
    public String serverUrl = "http://localhost:9000";

    // Path to ffmpeg binary (GUI apps on Mac don't inherit shell PATH)
    public String ffmpegPath = "ffmpeg";

    // Phase 4: Entity capture settings
    public boolean captureNearbyEntities = true;
    public int entityCaptureRadius = 16;          // blocks (reduced from 32 for performance)
    public boolean captureAllEntities = true;     // If false, only living entities
    public boolean captureItemEntities = true;    // Item drops

    // Phase 4: Block capture settings
    public boolean captureNearbyBlocks = true;
    public int blockCaptureRadius = 128;          // blocks (8 chunks) - matches typical render distance
    public boolean captureBlockEntities = true;   // Chests, furnaces, etc.
    public boolean useRelativeBlockPositions = false; // Relative to player vs absolute

    private static Config instance;

    public static Config getInstance() {
        if (instance == null) {
            instance = new Config();
            instance.load();
        }
        return instance;
    }

    private void load() {
        Path configPath = Paths.get(CONFIG_FILE);

        if (!Files.exists(configPath)) {
            save(); // Create default config
            return;
        }

        try (InputStream input = new FileInputStream(configPath.toFile())) {
            Properties props = new Properties();
            props.load(input);

            resolutionWidth = Integer.parseInt(props.getProperty("resolution_width", "640"));
            resolutionHeight = Integer.parseInt(props.getProperty("resolution_height", "360"));
            aspectRatioMode = props.getProperty("aspect_ratio_mode", "preserve");
            targetFps = Integer.parseInt(props.getProperty("target_fps", "20"));
            recordingDirectory = props.getProperty("recording_directory", "./recordings");
            showChatMessages = Boolean.parseBoolean(props.getProperty("show_chat_messages", "true"));
            showRecordingOverlay = Boolean.parseBoolean(props.getProperty("show_recording_overlay", "true"));

            // Upload server
            serverUrl = props.getProperty("server_url", "https://seattle-ferry-pam-acdbentity.trycloudflare.com");
            ffmpegPath = props.getProperty("ffmpeg_path", "ffmpeg");

            // Phase 4: Entity capture
            captureNearbyEntities = Boolean.parseBoolean(props.getProperty("capture_nearby_entities", "true"));
            entityCaptureRadius = Integer.parseInt(props.getProperty("entity_capture_radius", "16"));
            captureAllEntities = Boolean.parseBoolean(props.getProperty("capture_all_entities", "true"));
            captureItemEntities = Boolean.parseBoolean(props.getProperty("capture_item_entities", "true"));

            // Phase 4: Block capture
            captureNearbyBlocks = Boolean.parseBoolean(props.getProperty("capture_nearby_blocks", "true"));
            blockCaptureRadius = Integer.parseInt(props.getProperty("block_capture_radius", "128"));
            captureBlockEntities = Boolean.parseBoolean(props.getProperty("capture_block_entities", "true"));
            useRelativeBlockPositions = Boolean.parseBoolean(props.getProperty("use_relative_block_positions", "false"));

        } catch (IOException e) {
            System.err.println("[Blockscope] Failed to load config: " + e.getMessage());
        }
    }

    public void save() {
        try (OutputStream output = new FileOutputStream(CONFIG_FILE)) {
            Properties props = new Properties();
            props.setProperty("resolution_width", String.valueOf(resolutionWidth));
            props.setProperty("resolution_height", String.valueOf(resolutionHeight));
            props.setProperty("aspect_ratio_mode", aspectRatioMode);
            props.setProperty("target_fps", String.valueOf(targetFps));
            props.setProperty("recording_directory", recordingDirectory);
            props.setProperty("show_chat_messages", String.valueOf(showChatMessages));
            props.setProperty("show_recording_overlay", String.valueOf(showRecordingOverlay));

            // Upload server
            props.setProperty("server_url", serverUrl);
            props.setProperty("ffmpeg_path", ffmpegPath);

            // Phase 4: Entity capture
            props.setProperty("capture_nearby_entities", String.valueOf(captureNearbyEntities));
            props.setProperty("entity_capture_radius", String.valueOf(entityCaptureRadius));
            props.setProperty("capture_all_entities", String.valueOf(captureAllEntities));
            props.setProperty("capture_item_entities", String.valueOf(captureItemEntities));

            // Phase 4: Block capture
            props.setProperty("capture_nearby_blocks", String.valueOf(captureNearbyBlocks));
            props.setProperty("block_capture_radius", String.valueOf(blockCaptureRadius));
            props.setProperty("capture_block_entities", String.valueOf(captureBlockEntities));
            props.setProperty("use_relative_block_positions", String.valueOf(useRelativeBlockPositions));

            props.store(output, "Blockscope Configuration");
        } catch (IOException e) {
            System.err.println("[Blockscope] Failed to save config: " + e.getMessage());
        }
    }
}
