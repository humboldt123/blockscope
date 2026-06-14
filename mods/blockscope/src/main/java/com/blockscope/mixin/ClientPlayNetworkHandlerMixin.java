package com.blockscope.mixin;

import com.blockscope.network.SessionProtocol;
import net.minecraft.client.network.ClientPlayNetworkHandler;
import net.minecraft.text.Text;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Captures the disconnect reason so SessionProtocol can tell a user-initiated quit
 * ("disconnect.quitting") from a server kick / timeout / our own stuck-disconnect.
 * Only the latter should trigger auto-reconnect.
 *
 * Low priority so this HEAD inject runs before Fabric's DISCONNECT event handler
 * (default priority 1000), guaranteeing the reason is recorded before SessionProtocol
 * .onDisconnect reads it.
 */
@Mixin(value = ClientPlayNetworkHandler.class, priority = 500)
public class ClientPlayNetworkHandlerMixin {

    @Inject(method = "onDisconnected", at = @At("HEAD"))
    private void blockscope$captureDisconnectReason(Text reason, CallbackInfo ci) {
        SessionProtocol.setLastDisconnectReason(reason);
    }
}
