package com.blockscope.gui;

import com.blockscope.util.Config;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.gui.widget.ButtonWidget;
import net.minecraft.client.util.math.MatrixStack;
import net.minecraft.text.Text;

public class ConfigScreen extends Screen {
    private final Screen parent;
    private final Config config;

    // Config values (local copies until save)
    private int resolutionWidth;
    private int resolutionHeight;
    private String aspectRatioMode;
    private int targetFps;
    private boolean showChatMessages;
    private boolean showRecordingOverlay;

    public ConfigScreen(Screen parent) {
        super(Text.literal("Blockscope Configuration"));
        this.parent = parent;
        this.config = Config.getInstance();

        // Load current values
        this.resolutionWidth = config.resolutionWidth;
        this.resolutionHeight = config.resolutionHeight;
        this.aspectRatioMode = config.aspectRatioMode;
        this.targetFps = config.targetFps;
        this.showChatMessages = config.showChatMessages;
        this.showRecordingOverlay = config.showRecordingOverlay;
    }

    @Override
    protected void init() {
        int centerX = this.width / 2;
        int startY = 40;
        int spacing = 25;

        // Resolution presets
        this.addDrawableChild(ButtonWidget.builder(
            Text.literal("Resolution: " + resolutionWidth + "×" + resolutionHeight),
            button -> {
                cycleResolution();
                button.setMessage(Text.literal("Resolution: " + resolutionWidth + "×" + resolutionHeight));
            })
            .dimensions(centerX - 100, startY, 200, 20)
            .build()
        );

        // Aspect ratio mode
        this.addDrawableChild(ButtonWidget.builder(
            Text.literal("Aspect Ratio: " + aspectRatioMode),
            button -> {
                cycleAspectRatioMode();
                button.setMessage(Text.literal("Aspect Ratio: " + aspectRatioMode));
            })
            .dimensions(centerX - 100, startY + spacing, 200, 20)
            .build()
        );

        // Target FPS
        this.addDrawableChild(ButtonWidget.builder(
            Text.literal("Target FPS: " + targetFps),
            button -> {
                cycleFps();
                button.setMessage(Text.literal("Target FPS: " + targetFps));
            })
            .dimensions(centerX - 100, startY + spacing * 2, 200, 20)
            .build()
        );

        // Chat messages toggle
        this.addDrawableChild(ButtonWidget.builder(
            Text.literal("Chat Messages: " + (showChatMessages ? "ON" : "OFF")),
            button -> {
                showChatMessages = !showChatMessages;
                button.setMessage(Text.literal("Chat Messages: " + (showChatMessages ? "ON" : "OFF")));
            })
            .dimensions(centerX - 100, startY + spacing * 3, 200, 20)
            .build()
        );

        // Recording overlay toggle
        this.addDrawableChild(ButtonWidget.builder(
            Text.literal("Recording Overlay: " + (showRecordingOverlay ? "ON" : "OFF")),
            button -> {
                showRecordingOverlay = !showRecordingOverlay;
                button.setMessage(Text.literal("Recording Overlay: " + (showRecordingOverlay ? "ON" : "OFF")));
            })
            .dimensions(centerX - 100, startY + spacing * 4, 200, 20)
            .build()
        );

        // Save button
        this.addDrawableChild(ButtonWidget.builder(
            Text.literal("Save"),
            button -> {
                saveConfig();
                this.client.setScreen(parent);
            })
            .dimensions(centerX - 100, this.height - 50, 95, 20)
            .build()
        );

        // Cancel button
        this.addDrawableChild(ButtonWidget.builder(
            Text.literal("Cancel"),
            button -> this.client.setScreen(parent))
            .dimensions(centerX + 5, this.height - 50, 95, 20)
            .build()
        );
    }

    private void cycleResolution() {
        if (resolutionWidth == 640 && resolutionHeight == 360) {
            resolutionWidth = 854;
            resolutionHeight = 480;
        } else if (resolutionWidth == 854 && resolutionHeight == 480) {
            resolutionWidth = 1280;
            resolutionHeight = 720;
        } else if (resolutionWidth == 1280 && resolutionHeight == 720) {
            resolutionWidth = 1920;
            resolutionHeight = 1080;
        } else {
            resolutionWidth = 640;
            resolutionHeight = 360;
        }
    }

    private void cycleAspectRatioMode() {
        switch (aspectRatioMode) {
            case "preserve":
                aspectRatioMode = "crop";
                break;
            case "crop":
                aspectRatioMode = "stretch";
                break;
            default:
                aspectRatioMode = "preserve";
                break;
        }
    }

    private void cycleFps() {
        if (targetFps == 10) {
            targetFps = 15;
        } else if (targetFps == 15) {
            targetFps = 20;
        } else if (targetFps == 20) {
            targetFps = 30;
        } else {
            targetFps = 10;
        }
    }

    private void saveConfig() {
        config.resolutionWidth = this.resolutionWidth;
        config.resolutionHeight = this.resolutionHeight;
        config.aspectRatioMode = this.aspectRatioMode;
        config.targetFps = this.targetFps;
        config.showChatMessages = this.showChatMessages;
        config.showRecordingOverlay = this.showRecordingOverlay;
        config.save();
    }

    @Override
    public void render(MatrixStack matrices, int mouseX, int mouseY, float delta) {
        this.renderBackground(matrices);

        // Title
        drawCenteredTextWithShadow(matrices, this.textRenderer, this.title, this.width / 2, 20, 0xFFFFFF);

        // Descriptions
        drawCenteredTextWithShadow(matrices, this.textRenderer,
            Text.literal("§7Video capture resolution"),
            this.width / 2, 30, 0xFFFFFF);

        super.render(matrices, mouseX, mouseY, delta);
    }

    @Override
    public void close() {
        this.client.setScreen(parent);
    }
}
