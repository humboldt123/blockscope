package com.blockscope.model;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.util.ArrayList;
import java.util.List;

public class FrameMapping {
    public List<FrameTickPair> frames = new ArrayList<>();

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    public static class FrameTickPair {
        public int frame;
        public long tick;

        public FrameTickPair(int frame, long tick) {
            this.frame = frame;
            this.tick = tick;
        }
    }

    public void addMapping(int frame, long tick) {
        frames.add(new FrameTickPair(frame, tick));
    }

    public String toJson() {
        return GSON.toJson(this);
    }
}
