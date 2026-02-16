package com.blockscope.gui;

import com.blockscope.util.Config;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.gui.widget.ButtonWidget;
import net.minecraft.client.gui.widget.SliderWidget;
import net.minecraft.client.util.math.MatrixStack;
import net.minecraft.text.LiteralText;
import net.minecraft.text.TranslatableText;

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
        super(new LiteralText("Blockscope Configuration"));
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
        this.addButton(new ButtonWidget(
            centerX - 100, startY, 200, 20,
            new LiteralText("Resolution: " + resolutionWidth + "×" + resolutionHeight),
            button -> {
                cycleResolution();
                button.setMessage(new LiteralText("Resolution: " + resolutionWidth + "×" + resolutionHeight));
            }
        ));

        // Aspect ratio mode
        this.addButton(new ButtonWidget(
            centerX - 100, startY + spacing, 200, 20,
            new LiteralText("Aspect Ratio: " + aspectRatioMode),
            button -> {
                cycleAspectRatioMode();
                button.setMessage(new LiteralText("Aspect Ratio: " + aspectRatioMode));
            }
        ));

        // Target FPS
        this.addButton(new ButtonWidget(
            centerX - 100, startY + spacing * 2, 200, 20,
            new LiteralText("Target FPS: " + targetFps),
            button -> {
                cycleFps();
                button.setMessage(new LiteralText("Target FPS: " + targetFps));
            }
        ));

        // Chat messages toggle
        this.addButton(new ButtonWidget(
            centerX - 100, startY + spacing * 3, 200, 20,
            new LiteralText("Chat Messages: " + (showChatMessages ? "ON" : "OFF")),
            button -> {
                showChatMessages = !showChatMessages;
                button.setMessage(new LiteralText("Chat Messages: " + (showChatMessages ? "ON" : "OFF")));
            }
        ));

        // Recording overlay toggle
        this.addButton(new ButtonWidget(
            centerX - 100, startY + spacing * 4, 200, 20,
            new LiteralText("Recording Overlay: " + (showRecordingOverlay ? "ON" : "OFF")),
            button -> {
                showRecordingOverlay = !showRecordingOverlay;
                button.setMessage(new LiteralText("Recording Overlay: " + (showRecordingOverlay ? "ON" : "OFF")));
            }
        ));

        // Save button
        this.addButton(new ButtonWidget(
            centerX - 100, this.height - 50, 95, 20,
            new LiteralText("Save"),
            button -> {
                saveConfig();
                this.client.openScreen(parent);
            }
        ));

        // Cancel button
        this.addButton(new ButtonWidget(
            centerX + 5, this.height - 50, 95, 20,
            new LiteralText("Cancel"),
            button -> this.client.openScreen(parent)
        ));
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
        drawCenteredText(matrices, this.textRenderer, this.title, this.width / 2, 20, 0xFFFFFF);

        // Descriptions
        drawCenteredText(matrices, this.textRenderer,
            new LiteralText("§7Video capture resolution"),
            this.width / 2, 30, 0xFFFFFF);

        super.render(matrices, mouseX, mouseY, delta);
    }

    @Override
    public void onClose() {
        this.client.openScreen(parent);
    }
}
