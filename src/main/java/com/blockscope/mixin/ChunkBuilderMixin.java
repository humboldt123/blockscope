package com.blockscope.mixin;

import com.blockscope.recording.RecordingManager;
import net.minecraft.block.BlockState;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.render.Camera;
import net.minecraft.client.render.chunk.ChunkBuilder;
import net.minecraft.client.render.chunk.ChunkRendererRegion;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.Set;

/**
 * Captures blocks being added to the render mesh (what the player actually sees).
 *
 * INJECTION TARGET: ChunkBuilder.BuiltChunk.RebuildTask.render()
 * This method is called when:
 * - Chunks enter render distance (first time player sees them)
 * - Blocks change in the chunk (player breaks/places, piston moves, redstone)
 * - Lighting changes in the chunk
 *
 * By hooking here, we capture ONLY blocks that are actually rendered (visible to player),
 * not entire chunk columns like frustum culling would.
 */
@Mixin(targets = "net.minecraft.client.render.chunk.ChunkBuilder$BuiltChunk$RebuildTask")
public abstract class ChunkBuilderMixin {

    @Shadow
    private ChunkRendererRegion region;  // The chunk region being rendered (field_20838)

    /**
     * Inject at HEAD of render() to capture blocks before they're added to mesh.
     */
    @Inject(
        method = "render",
        at = @At("HEAD")
    )
    private void onChunkRebuild(CallbackInfoReturnable<Set> cir) {
        if (!RecordingManager.getInstance().isRecording()) {
            return;
        }

        try {
            if (this.region == null) {
                return;
            }

            // Cast to accessor to access protected fields
            ChunkRendererRegionAccessor accessor = (ChunkRendererRegionAccessor) this.region;

            World world = accessor.getWorld();
            BlockPos offset = accessor.getOffset();
            int sizeX = accessor.getSizeX();
            int sizeY = accessor.getSizeY();
            int sizeZ = accessor.getSizeZ();
            BlockState[] blockStates = accessor.getBlockStates();

            if (world == null || offset == null || blockStates == null) {
                return;
            }

            // Get camera for frustum culling
            MinecraftClient client = MinecraftClient.getInstance();
            Camera camera = client.gameRenderer.getCamera();
            if (camera == null) {
                return;
            }
            Vec3d cameraPos = camera.getPos();

            // Iterate through all blocks in the chunk region
            for (int x = 0; x < sizeX; x++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int z = 0; z < sizeZ; z++) {
                        // Calculate the array index manually (same formula as Minecraft's getIndex)
                        // Formula: z * sizeX * sizeY + y * sizeX + x
                        int index = z * sizeX * sizeY + y * sizeX + x;
                        if (index < 0 || index >= blockStates.length) {
                            continue;
                        }

                        BlockState state = blockStates[index];

                        // Skip air blocks (optimization - don't track empty space)
                        if (state == null || state.isAir()) {
                            continue;
                        }

                        // Calculate world position
                        BlockPos pos = offset.add(x, y, z);

                        // CRITICAL: Frustum culling - only save blocks in camera view
                        if (!isInCameraView(pos, cameraPos, camera)) {
                            continue; // Block is behind camera or outside FOV
                        }

                        // Only save blocks with at least one exposed face (eliminates underground)
                        if (!hasExposedFace(world, pos, x, y, z, sizeX, sizeY, sizeZ, blockStates, offset)) {
                            continue; // Block is completely surrounded, not visible
                        }

                        // Notify RecordingManager that this block was rendered (seen by player)
                        RecordingManager.getInstance().onBlockRendered(world, pos, state);
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("[Blockscope] Error capturing rendered blocks: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Check if a block is in the camera's view frustum.
     * This eliminates blocks behind the player and outside FOV.
     */
    private boolean isInCameraView(BlockPos blockPos, Vec3d cameraPos, Camera camera) {
        // Vector from camera to block center
        double dx = blockPos.getX() + 0.5 - cameraPos.x;
        double dy = blockPos.getY() + 0.5 - cameraPos.y;
        double dz = blockPos.getZ() + 0.5 - cameraPos.z;

        // Distance check (only render blocks within 64 blocks for performance)
        double distanceSq = dx * dx + dy * dy + dz * dz;
        if (distanceSq > 64 * 64) {
            return false;
        }

        // Get camera direction using pitch and yaw
        float pitch = camera.getPitch();
        float yaw = camera.getYaw();

        // Convert pitch/yaw to direction vector
        double yawRad = Math.toRadians(yaw + 90); // Add 90 to align with Minecraft's coordinate system
        double pitchRad = Math.toRadians(pitch);

        double lookX = Math.cos(pitchRad) * Math.cos(yawRad);
        double lookY = -Math.sin(pitchRad);
        double lookZ = Math.cos(pitchRad) * Math.sin(yawRad);

        // Normalize the vector to block
        double length = Math.sqrt(distanceSq);
        if (length < 0.01) {
            return true; // Block is at camera position
        }

        double ndx = dx / length;
        double ndy = dy / length;
        double ndz = dz / length;

        // Dot product with camera direction
        double dot = lookX * ndx + lookY * ndy + lookZ * ndz;

        // If dot product > 0, block is in front of camera
        // We use a threshold slightly negative to include blocks just at edge of view
        return dot > -0.2;
    }

    /**
     * Check if a block has at least one face exposed (adjacent to air or non-opaque block).
     * This filters out completely buried underground blocks.
     * NOTE: Blocks behind glass/water ARE considered exposed (they're visible).
     */
    private boolean hasExposedFace(World world, BlockPos worldPos, int x, int y, int z,
                                    int sizeX, int sizeY, int sizeZ,
                                    BlockState[] blockStates, BlockPos offset) {
        // Check all 6 adjacent positions (up, down, north, south, east, west)
        int[][] neighbors = {
            {0, 1, 0},   // Up
            {0, -1, 0},  // Down
            {1, 0, 0},   // East
            {-1, 0, 0},  // West
            {0, 0, 1},   // South
            {0, 0, -1}   // North
        };

        for (int[] dir : neighbors) {
            int nx = x + dir[0];
            int ny = y + dir[1];
            int nz = z + dir[2];

            // Check if neighbor is outside chunk region (exposed to adjacent chunk)
            if (nx < 0 || nx >= sizeX || ny < 0 || ny >= sizeY || nz < 0 || nz >= sizeZ) {
                // Need to check world for blocks outside this chunk region
                BlockPos neighborWorldPos = worldPos.add(dir[0], dir[1], dir[2]);
                BlockState neighborState = world.getBlockState(neighborWorldPos);

                // If neighbor is air OR non-opaque (glass, water, etc.), this face is exposed
                if (neighborState.isAir() || !neighborState.isOpaque()) {
                    return true;
                }
            } else {
                // Neighbor is within chunk region, check blockStates array
                int neighborIndex = nz * sizeX * sizeY + ny * sizeX + nx;
                if (neighborIndex >= 0 && neighborIndex < blockStates.length) {
                    BlockState neighborState = blockStates[neighborIndex];

                    // If neighbor is air OR non-opaque (glass, water, etc.), this face is exposed
                    if (neighborState == null || neighborState.isAir() || !neighborState.isOpaque()) {
                        return true;
                    }
                }
            }
        }

        return false; // All faces are blocked by opaque blocks, not visible
    }
}
