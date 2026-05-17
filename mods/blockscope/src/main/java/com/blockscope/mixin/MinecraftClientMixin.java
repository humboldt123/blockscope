package com.blockscope.mixin;

import com.blockscope.gui.RecordingOverlay;
import com.blockscope.recording.RecordingManager;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.util.math.MatrixStack;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(MinecraftClient.class)
public class MinecraftClientMixin {

    /**
     * Capture the frame at the very end of render(), after screens (inventory etc.)
     * have been drawn but before the buffer is presented. Then draw the overlay
     * AFTER capture so it's visible to the player but not in the recorded video.
     */
    @Inject(method = "render", at = @At("TAIL"))
    private void onRenderTail(boolean tick, CallbackInfo ci) {
        MinecraftClient client = (MinecraftClient) (Object) this;
        RecordingManager.getInstance().onRenderFrame();
        if (client.world != null) {
            RecordingOverlay.getInstance().render(new MatrixStack());
        }
    }
}
