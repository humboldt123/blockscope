package com.blockscope.model;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class SessionMetadata {
    public String sessionId;
    public long startTimestamp;
    public Long endTimestamp;
    public RecordingConfig config;
    public String minecraftVersion;
    public String modVersion;

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    public static class RecordingConfig {
        public int resolutionWidth;
        public int resolutionHeight;
        public String aspectRatioMode; // "preserve", "crop", "stretch"
        public int targetFps;
        public String recordingDirectory;
    }

    public String toJson() {
        return GSON.toJson(this);
    }
}
