package com.blockscope.mixin;

import com.blockscope.model.InputEvent;
import com.blockscope.recording.RecordingManager;
import net.minecraft.client.Keyboard;
import org.lwjgl.glfw.GLFW;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Keyboard.class)
public class KeyboardMixin {

    @Inject(method = "onKey", at = @At("HEAD"))
    private void onKeyEvent(long window, int key, int scancode, int action, int modifiers, CallbackInfo ci) {
        if (action == GLFW.GLFW_PRESS) {
            String keyName = getKeyName(key);
            InputEvent event = InputEvent.keyPress(0, 0, key, keyName);
            RecordingManager.getInstance().recordInputEvent(event);
        } else if (action == GLFW.GLFW_RELEASE) {
            String keyName = getKeyName(key);
            InputEvent event = InputEvent.keyRelease(0, 0, key, keyName);
            RecordingManager.getInstance().recordInputEvent(event);
        }
    }

    private String getKeyName(int key) {
        String name = GLFW.glfwGetKeyName(key, 0);
        return name != null ? name : "key_" + key;
    }
}
