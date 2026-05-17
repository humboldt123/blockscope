"""
Replay-driven world that progressively reveals blocks as ticks advance.
Uses spatial chunking: the world is divided into 16x16x16 chunks, each with
its own mesh. Only dirty chunks rebuild when blocks change.
"""

import numpy as np

from block_registry import BlockRegistry, RENDER_CUBE, RENDER_CROSS, RENDER_ELEMENTS, TINT_NONE
from meshes.chunk_mesh_builder import build_chunk_mesh
from replay_loader import ReplayLoader
from settings import *

CHUNK_SIZE = 16


def _chunk_key(x, y, z):
    return (x >> 4, y >> 4, z >> 4)


class Chunk:
    __slots__ = ('blocks', 'vao', 'vbo', 'vao_trans', 'vbo_trans', 'dirty')

    def __init__(self):
        self.blocks: dict[tuple[int, int, int], str] = {}
        self.vao = None
        self.vbo = None
        self.vao_trans = None   # transparent geometry (water)
        self.vbo_trans = None
        self.dirty = True


class ReplayWorld:
    def __init__(self, app, replay: ReplayLoader, block_registry: BlockRegistry):
        self.app = app
        self.replay = replay
        self.block_registry = block_registry

        # Global block lookup for AO across chunk boundaries
        self.blocks: dict[tuple[int, int, int], str] = {}
        # Solid positions for face culling / AO (only full opaque cubes)
        self._solid: set[tuple[int, int, int]] = set()
        # Liquid positions for internal face culling (water, lava)
        self._liquid: set[tuple[int, int, int]] = set()

        # Spatial chunks
        self._chunks: dict[tuple[int, int, int], Chunk] = {}
        self._any_dirty = False

        # Track which tick we've processed up to
        self.current_tick = 0

        # Current biome (from replay ticks)
        self.biome = 'minecraft:plains'

        # Pre-cache per-block data
        self._face_tex_cache: dict[str, tuple] = {}
        self._render_type_cache: dict[str, int] = {}
        self._tint_color_cache: dict[str, tuple] = {}
        self._tint_faces_cache: dict[str, tuple] = {}
        self._is_full_cache: dict[str, bool] = {}
        self._elements_cache: dict[str, list] = {}
        self._alpha_cache: dict[str, float] = {}

        # Get biome from tick 0 before registering blocks (needed for grass_block side baking)
        state = replay.get_player_state(0)
        if state:
            self.biome = state.get('world', {}).get('biome', 'minecraft:plains')

        import time
        t0 = time.time()
        for bid in replay.get_all_unique_block_ids():
            self._register_block(bid)
        print(f'[TIME] World: register blocks: {time.time() - t0:.2f}s')

        # Load all chunks at initialization
        t0 = time.time()
        self._load_all_chunks()
        print(f'[TIME] World: load chunks: {time.time() - t0:.2f}s')

        # Mark current tick as 0 (chunks already loaded)
        self.current_tick = 0

    def _cache_block(self, key: str):
        """Cache render data for a block key (plain block_id or variant key)."""
        self._face_tex_cache[key] = self.block_registry.block_face_textures.get(key, (0,)*6)
        self._render_type_cache[key] = self.block_registry.block_render_type.get(key, RENDER_CUBE)
        self._tint_faces_cache[key] = self.block_registry.block_tint_faces.get(key, (False,)*6)
        self._is_full_cache[key] = self.block_registry.block_is_full.get(key, True)
        # For variant keys, get tint color from base block_id
        base_id = key.split('[')[0] if '[' in key else key
        self._tint_color_cache[key] = self.block_registry.get_tint_color(base_id, self.biome)
        if key in self.block_registry.block_elements:
            self._elements_cache[key] = self.block_registry.block_elements[key]
        # Water is semi-transparent
        if base_id == 'minecraft:water':
            self._alpha_cache[key] = 0.5

    def _register_block(self, block_id: str):
        """Register a block and cache all its render data."""
        self.block_registry.register_block(block_id, biome=self.biome)
        self._cache_block(block_id)

    def _recompute_tint_colors(self):
        """Recompute tint colors for all registered blocks (when biome changes)."""
        for block_id in list(self._tint_color_cache.keys()):
            self._tint_color_cache[block_id] = self.block_registry.get_tint_color(block_id, self.biome)

    def _mark_dirty(self, x, y, z):
        """Mark the chunk containing (x,y,z) as dirty, plus neighbors if on a boundary."""
        ck = _chunk_key(x, y, z)
        chunk = self._chunks.get(ck)
        if chunk:
            chunk.dirty = True
        self._any_dirty = True

        # If block is on chunk boundary, also dirty the neighbor chunk
        # (needed for correct AO and face culling across boundaries)
        lx, ly, lz = x & 15, y & 15, z & 15
        if lx == 0:
            nb = self._chunks.get((ck[0]-1, ck[1], ck[2]))
            if nb: nb.dirty = True
        elif lx == 15:
            nb = self._chunks.get((ck[0]+1, ck[1], ck[2]))
            if nb: nb.dirty = True
        if ly == 0:
            nb = self._chunks.get((ck[0], ck[1]-1, ck[2]))
            if nb: nb.dirty = True
        elif ly == 15:
            nb = self._chunks.get((ck[0], ck[1]+1, ck[2]))
            if nb: nb.dirty = True
        if lz == 0:
            nb = self._chunks.get((ck[0], ck[1], ck[2]-1))
            if nb: nb.dirty = True
        elif lz == 15:
            nb = self._chunks.get((ck[0], ck[1], ck[2]+1))
            if nb: nb.dirty = True

    def _set_block(self, x, y, z, block_id):
        """Add or update a block."""
        pos = (x, y, z)
        if block_id not in self._face_tex_cache:
            self._register_block(block_id)

        self.blocks[pos] = block_id

        # Update solid set
        if self._is_full_cache.get(block_id, True):
            self._solid.add(pos)
        else:
            self._solid.discard(pos)

        # Track liquid positions for internal face culling
        base_id = block_id.split('[')[0] if '[' in block_id else block_id
        if base_id in ('minecraft:water', 'minecraft:lava'):
            self._liquid.add(pos)
        else:
            self._liquid.discard(pos)

        # Add to chunk
        ck = _chunk_key(x, y, z)
        if ck not in self._chunks:
            self._chunks[ck] = Chunk()
        self._chunks[ck].blocks[pos] = block_id

        self._mark_dirty(x, y, z)

    def _remove_block(self, x, y, z):
        """Remove a block."""
        pos = (x, y, z)
        if pos not in self.blocks:
            return
        del self.blocks[pos]
        self._solid.discard(pos)
        self._liquid.discard(pos)

        ck = _chunk_key(x, y, z)
        chunk = self._chunks.get(ck)
        if chunk:
            chunk.blocks.pop(pos, None)
            if not chunk.blocks:
                self._release_chunk_gpu(chunk)
                del self._chunks[ck]
            else:
                chunk.dirty = True

        self._mark_dirty(x, y, z)

    def _resolve_block_key(self, block_id: str, properties: str) -> str:
        """Resolve a block_id + properties into the right render key.
        For element blocks with state properties, creates a variant with correct rotation."""
        if not properties:
            return block_id

        # Try to register variant (returns block_id if not an element block)
        variant_key = self.block_registry.register_block_variant(
            block_id, properties, self.biome)

        # Cache the variant if it's new
        if variant_key != block_id and variant_key not in self._face_tex_cache:
            self._cache_block(variant_key)

        return variant_key

    def _process_tick(self, tick: int):
        """
        Process world state for a given tick.
        Applies block changes for this specific tick.
        """
        changes = self.replay.block_changes_by_tick.get(tick, [])
        if changes:
            print(f"[World] Tick {tick}: Applying {len(changes)} block changes")
        for change in changes:
            x = change['x']
            y = change['y']
            z = change['z']
            block_id = change['blockId']
            properties = change.get('blockStateProperties', '')

            if block_id == 'minecraft:air':
                self._remove_block(x, y, z)
            else:
                key = self._resolve_block_key(block_id, properties)
                self._set_block(x, y, z, key)

    def _load_all_chunks(self):
        """Load all chunk blocks into the world (called once at init)."""
        total_chunks = len(self.replay.chunks)
        print(f"[World] Loading {total_chunks} chunks...")

        # Batch load without marking dirty - MUCH faster
        for (chunkX, chunkZ), chunk_blocks in self.replay.chunks.items():
            if not chunk_blocks:
                continue

            # IMPORTANT: Must calculate chunk key from actual block positions, not from (chunkX, chunkZ)
            # because blocks may span multiple 16x16x16 chunks in Y direction (different Y sections)
            chunk_keys_seen = set()
            for (x, y, z) in chunk_blocks.keys():
                ck = _chunk_key(x, y, z)
                if ck not in chunk_keys_seen:
                    chunk_keys_seen.add(ck)
                    if ck not in self._chunks:
                        self._chunks[ck] = Chunk()

            for (x, y, z), block_id in chunk_blocks.items():
                if block_id and block_id != 'minecraft:air':
                    pos = (x, y, z)
                    ck = _chunk_key(x, y, z)

                    # Register block if needed
                    if block_id not in self._face_tex_cache:
                        self._register_block(block_id)

                    # Direct add without dirty marking
                    self.blocks[pos] = block_id
                    if self._is_full_cache.get(block_id, True):
                        self._solid.add(pos)

                    base_id = block_id.split('[')[0] if '[' in block_id else block_id
                    if base_id in ('minecraft:water', 'minecraft:lava'):
                        self._liquid.add(pos)

                    self._chunks[ck].blocks[pos] = block_id

        # Mark all chunks dirty once at end
        for chunk in self._chunks.values():
            chunk.dirty = True
        self._any_dirty = True

        print(f"[World] Loaded {len(self.blocks)} blocks from {total_chunks} chunks")

    def reset(self):
        """Clear all state and release GPU resources. Used for restart/seek-backward."""
        for chunk in self._chunks.values():
            self._release_chunk_gpu(chunk)
        self._chunks.clear()
        self.blocks.clear()
        self._solid.clear()
        self._liquid.clear()
        self.current_tick = 0
        self._any_dirty = False

        # Reload all chunks from NBT
        self._load_all_chunks()

    def advance_to_tick(self, target_tick: int):
        """Advance world state to the given tick."""
        target_tick = min(target_tick, self.replay.max_tick)

        # # Biome-change tint recompute disabled — tints are hardcoded now
        # state = self.replay.get_player_state(target_tick)
        # if state:
        #     new_biome = state.get('world', {}).get('biome', self.biome)
        #     if new_biome != self.biome:
        #         self.biome = new_biome
        #         self._recompute_tint_colors()
        #         for chunk in self._chunks.values():
        #             chunk.dirty = True
        #         self._any_dirty = True

        # Skip to next tick with changes if target is far ahead
        ticks_with_changes = sorted(self.replay.block_changes_by_tick.keys())
        for change_tick in ticks_with_changes:
            if self.current_tick < change_tick <= target_tick:
                # Jump directly to this tick and process it
                self.current_tick = change_tick
                self._process_tick(change_tick)

        # Update current_tick to target even if no changes
        if self.current_tick < target_tick:
            self.current_tick = target_tick

    def _release_chunk_gpu(self, chunk: Chunk):
        """Release all GPU resources for a chunk."""
        if chunk.vao:
            chunk.vao.release()
            chunk.vao = None
        if chunk.vbo:
            chunk.vbo.release()
            chunk.vbo = None
        if chunk.vao_trans:
            chunk.vao_trans.release()
            chunk.vao_trans = None
        if chunk.vbo_trans:
            chunk.vbo_trans.release()
            chunk.vbo_trans = None

    def _rebuild_chunk(self, chunk: Chunk):
        """Rebuild mesh for a single chunk."""
        if not chunk.blocks:
            self._release_chunk_gpu(chunk)
            chunk.dirty = False
            return

        opaque_data, trans_data = build_chunk_mesh(
            blocks=chunk.blocks,
            solid_occupied=self._solid,
            liquid_occupied=self._liquid,
            block_face_tex_map=self._face_tex_cache,
            block_render_types=self._render_type_cache,
            block_tint_colors=self._tint_color_cache,
            block_tint_faces=self._tint_faces_cache,
            block_elements=self._elements_cache,
            block_alpha=self._alpha_cache,
        )

        self._release_chunk_gpu(chunk)

        fmt = '3f 1f 1f 3f 1f'
        attrs = ('in_position', 'in_tex_id', 'in_packed_face_ao', 'in_tint_color', 'in_alpha')

        if len(opaque_data) > 0:
            chunk.vbo = self.app.ctx.buffer(opaque_data.tobytes())
            chunk.vao = self.app.ctx.vertex_array(
                self.app.shader_program.chunk,
                [(chunk.vbo, fmt, *attrs)],
                skip_errors=True
            )

        if len(trans_data) > 0:
            chunk.vbo_trans = self.app.ctx.buffer(trans_data.tobytes())
            chunk.vao_trans = self.app.ctx.vertex_array(
                self.app.shader_program.chunk,
                [(chunk.vbo_trans, fmt, *attrs)],
                skip_errors=True
            )

        chunk.dirty = False

    def rebuild_mesh(self):
        """Rebuild only dirty chunks."""
        for chunk in self._chunks.values():
            if chunk.dirty:
                self._rebuild_chunk(chunk)
        self._any_dirty = False

    def render(self):
        if self._any_dirty:
            self.rebuild_mesh()

        # Pass 1: opaque geometry
        for chunk in self._chunks.values():
            if chunk.vao:
                chunk.vao.render()

        # Pass 2: transparent geometry (depth write off so blocks behind show through)
        has_trans = any(c.vao_trans for c in self._chunks.values())
        if has_trans:
            fbo = self.app.ctx.detect_framebuffer()
            fbo.depth_mask = False
            for chunk in self._chunks.values():
                if chunk.vao_trans:
                    chunk.vao_trans.render()
            fbo.depth_mask = True
