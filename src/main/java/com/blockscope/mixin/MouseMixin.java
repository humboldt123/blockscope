package com.blockscope.mixin;

import com.blockscope.model.InputEvent;
import com.blockscope.recording.RecordingManager;
import net.minecraft.client.Mouse;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Mouse.class)
public class MouseMixin {

    @Shadow private double x;
    @Shadow private double y;

    private double lastX = 0;
    private double lastY = 0;

    @Inject(method = "onMouseButton", at = @At("HEAD"))
    private void onMouseButton(long window, int button, int action, int mods, CallbackInfo ci) {
        boolean pressed = (action == 1); // GLFW_PRESS = 1
        InputEvent event = InputEvent.mouseButton(0, 0, button, pressed);
        RecordingManager.getInstance().recordInputEvent(event);
    }

    @Inject(method = "onCursorPos", at = @At("HEAD"))
    private void onCursorPos(long window, double x, double y, CallbackInfo ci) {
        double dx = x - lastX;
        double dy = y - lastY;

        if (lastX != 0 || lastY != 0) { // Skip first event (no delta available)
            InputEvent event = InputEvent.mouseMove(0, 0, dx, dy);
            RecordingManager.getInstance().recordInputEvent(event);
        }

        lastX = x;
        lastY = y;
    }
}
