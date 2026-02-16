package com.blockscope.mixin;

import net.minecraft.block.BlockState;
import net.minecraft.client.render.chunk.ChunkRendererRegion;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/**
 * Accessor mixin for ChunkRendererRegion protected fields.
 * Allows us to access the internal data of chunk regions during rendering.
 */
@Mixin(ChunkRendererRegion.class)
public interface ChunkRendererRegionAccessor {

    @Accessor("world")
    World getWorld();

    @Accessor("offset")
    BlockPos getOffset();

    @Accessor("sizeX")
    int getSizeX();

    @Accessor("sizeY")
    int getSizeY();

    @Accessor("sizeZ")
    int getSizeZ();

    @Accessor("blockStates")
    BlockState[] getBlockStates();

    // Note: getIndex() is a method, not a field, so we can't use @Accessor for it
    // We'll calculate the index manually using the same formula Minecraft uses
}
