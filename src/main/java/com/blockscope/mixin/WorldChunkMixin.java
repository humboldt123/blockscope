package com.blockscope.mixin;

import com.blockscope.recording.RecordingManager;
import net.minecraft.block.BlockState;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.chunk.WorldChunk;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Captures block changes only.
 * Chunks are captured when recording starts (see RecordingManager.captureAllLoadedChunks()).
 */
@Mixin(WorldChunk.class)
public class WorldChunkMixin {

    /**
     * Inject at HEAD to capture the OLD block state before it changes.
     */
    @Inject(
        method = "setBlockState",
        at = @At("HEAD")
    )
    private void onBlockChange(
        BlockPos pos,
        BlockState state,
        boolean moved,
        CallbackInfoReturnable<BlockState> cir
    ) {
        if (!RecordingManager.getInstance().isRecording()) {
            return;
        }

        try {
            WorldChunk chunk = (WorldChunk) (Object) this;

            // Get the old block state (what's currently there)
            BlockState oldState = chunk.getBlockState(pos);

            // Notify RecordingManager of the block change
            RecordingManager.getInstance().onBlockChanged(
                chunk.getWorld(),
                pos,
                oldState,  // Previous state
                state      // New state
            );
        } catch (Exception e) {
            System.err.println("[Blockscope] Error capturing block change: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
