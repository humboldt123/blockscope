package com.blockscope.model;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class InputEvent {
    public long tick;
    public long offsetMs; // Milliseconds offset within the tick
    public String type;   // "key_press", "key_release", "mouse_button", "mouse_move"

    // For keyboard events
    public Integer keyCode;
    public String keyName;

    // For mouse button events
    public Integer mouseButton;
    public String buttonState; // "pressed", "released"

    // For mouse movement
    public Double deltaX;
    public Double deltaY;
    public Double mouseSensitivity; // Mouse sensitivity setting at time of input

    private static final Gson GSON = new GsonBuilder().create();

    public static InputEvent keyPress(long tick, long offsetMs, int keyCode, String keyName) {
        InputEvent event = new InputEvent();
        event.tick = tick;
        event.offsetMs = offsetMs;
        event.type = "key_press";
        event.keyCode = keyCode;
        event.keyName = keyName;
        return event;
    }

    public static InputEvent keyRelease(long tick, long offsetMs, int keyCode, String keyName) {
        InputEvent event = new InputEvent();
        event.tick = tick;
        event.offsetMs = offsetMs;
        event.type = "key_release";
        event.keyCode = keyCode;
        event.keyName = keyName;
        return event;
    }

    public static InputEvent mouseButton(long tick, long offsetMs, int button, boolean pressed) {
        InputEvent event = new InputEvent();
        event.tick = tick;
        event.offsetMs = offsetMs;
        event.type = "mouse_button";
        event.mouseButton = button;
        event.buttonState = pressed ? "pressed" : "released";
        return event;
    }

    public static InputEvent mouseMove(long tick, long offsetMs, double dx, double dy) {
        InputEvent event = new InputEvent();
        event.tick = tick;
        event.offsetMs = offsetMs;
        event.type = "mouse_move";
        event.deltaX = dx;
        event.deltaY = dy;
        return event;
    }

    public String toJson() {
        return GSON.toJson(this);
    }
}
