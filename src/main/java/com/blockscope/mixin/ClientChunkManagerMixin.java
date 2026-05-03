package com.blockscope.mixin;

import com.blockscope.recording.RecordingManager;
import net.minecraft.client.world.ClientChunkManager;
import net.minecraft.world.chunk.WorldChunk;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Captures chunks as they load from network packets.
 * ReplayMod approach - passive recording when chunks arrive from server.
 */
@Mixin(ClientChunkManager.class)
public class ClientChunkManagerMixin {

    /**
     * Hook when chunk is loaded from packet data.
     * This fires when server sends chunk data to client.
     */
    @Inject(
        method = "loadChunkFromPacket",
        at = @At("RETURN")
    )
    private void onChunkLoadFromPacket(int x, int z, net.minecraft.network.PacketByteBuf buf, net.minecraft.nbt.NbtCompound nbt, java.util.function.Consumer consumer, CallbackInfoReturnable<WorldChunk> cir) {
        WorldChunk chunk = cir.getReturnValue();
        if (chunk != null) {
            // Pass raw NBT data to RecordingManager (ReplayMod approach)
            RecordingManager.getInstance().onChunkLoad(chunk, nbt);
        }
    }
}
