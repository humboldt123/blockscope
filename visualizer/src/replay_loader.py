"""
Loads a Blockscope recording session and provides tick-by-tick access to:
  - Player position/rotation per tick (from ticks.jsonl)
  - Chunks (binary NBT files in chunks/)
  - Block changes per tick (from block_changes.jsonl)
"""

import json
import os
import nbtlib
import numpy as np
from collections import defaultdict


class ReplayLoader:
    def __init__(self, session_dir: str):
        self.session_dir = session_dir

        # Load metadata
        with open(os.path.join(session_dir, 'metadata.json')) as f:
            self.metadata = json.load(f)

        # Load ticks (player state per tick - includes camera x,y,z,pitch,yaw,fov,cameraPerspective)
        self.ticks = []
        with open(os.path.join(session_dir, 'ticks.jsonl')) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.ticks.append(json.loads(line))

        # Load chunks (binary NBT files) - ONLY LOAD NEARBY CHUNKS
        self.chunks = {}  # (chunkX, chunkZ) -> blocks dict
        self.all_chunk_files = {}  # (chunkX, chunkZ) -> filepath (for lazy loading)
        chunks_dir = os.path.join(session_dir, 'chunks')

        # Get initial player position to determine which chunks to load
        initial_pos = self.get_initial_player_pos()
        player_chunk_x = int(initial_pos[0]) >> 4
        player_chunk_z = int(initial_pos[2]) >> 4
        load_radius = 8  # Only load chunks within 8 chunks of spawn

        if os.path.exists(chunks_dir):
            import time
            print(f"[Replay] Loading chunks within {load_radius} chunks of player spawn...")
            t_start = time.time()
            t_nbt_load = 0
            t_nbt_parse = 0
            for filename in os.listdir(chunks_dir):
                if filename.endswith('.nbt'):
                    # Parse filename: chunk_overworld_X_Z_tickN.nbt
                    parts = filename.replace('.nbt', '').split('_')
                    if len(parts) >= 4:
                        chunkX = int(parts[2])
                        chunkZ = int(parts[3])
                        chunk_path = os.path.join(chunks_dir, filename)

                        # Store filepath for potential lazy loading
                        self.all_chunk_files[(chunkX, chunkZ)] = chunk_path

                        # Only load chunks near spawn
                        if abs(chunkX - player_chunk_x) <= load_radius and abs(chunkZ - player_chunk_z) <= load_radius:
                            try:
                                t0 = time.time()
                                nbt = nbtlib.load(chunk_path, gzipped=True)
                                t_nbt_load += time.time() - t0

                                t0 = time.time()
                                blocks = self._parse_chunk_nbt(nbt, chunkX, chunkZ)
                                t_nbt_parse += time.time() - t0

                                self.chunks[(chunkX, chunkZ)] = blocks
                            except Exception as e:
                                print(f"[Replay] Failed to load {filename}: {e}")

            print(f"[Replay] Loaded {len(self.chunks)}/{len(self.all_chunk_files)} chunks")
            print(f"[TIME] NBT file loading: {t_nbt_load:.2f}s")
            print(f"[TIME] NBT parsing: {t_nbt_parse:.2f}s")
            print(f"[TIME] Total chunk loading: {time.time() - t_start:.2f}s")

        # Load block changes (overrides chunk data)
        self.block_changes_by_tick: dict[int, list] = defaultdict(list)
        block_changes_path = os.path.join(session_dir, 'block_changes.jsonl')
        if os.path.exists(block_changes_path):
            with open(block_changes_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    change = json.loads(line)
                    tick = change.get('tick', 0)
                    self.block_changes_by_tick[tick].append(change)
            total_changes = sum(len(v) for v in self.block_changes_by_tick.values())
            print(f"[Replay] Loaded {total_changes} block changes across {len(self.block_changes_by_tick)} ticks")

        self.max_tick = max(t['tick'] for t in self.ticks) if self.ticks else 0

        # Load frame mapping (frame index -> tick number) if available
        self.frame_to_tick: list[int] = []
        frame_mapping_path = os.path.join(session_dir, 'frame_mapping.jsonl')
        if os.path.exists(frame_mapping_path):
            with open(frame_mapping_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self.frame_to_tick.append(entry['tick'])
        self.has_frame_mapping = len(self.frame_to_tick) > 0

        print(f"[Replay] {len(self.ticks)} ticks, {len(self.chunks)} chunks, max_tick={self.max_tick}")

    def _parse_chunk_nbt(self, nbt, chunkX: int, chunkZ: int) -> dict:
        """Parse chunk NBT and return dict of (x,y,z) -> blockId"""
        blocks = {}

        # Minecraft 1.19.4 chunk NBT structure varies
        # Try different NBT layouts
        try:
            # Try sections array
            if 'sections' in nbt:
                for section in nbt['sections']:
                    section_y = int(section.get('Y', 0))

                    if 'block_states' in section and 'palette' in section['block_states']:
                        palette = section['block_states']['palette']

                        # Single block type - fill entire section
                        if len(palette) == 1:
                            block_name = palette[0].get('Name', 'minecraft:air')
                            if block_name != 'minecraft:air':
                                for dx in range(16):
                                    for dy in range(16):
                                        for dz in range(16):
                                            x = chunkX * 16 + dx
                                            y = section_y * 16 + dy
                                            z = chunkZ * 16 + dz
                                            blocks[(x, y, z)] = block_name

                        # Multiple block types - unpack data array (vectorized)
                        elif 'data' in section['block_states']:
                            data = section['block_states']['data']
                            bits_per_block = max(4, (len(palette) - 1).bit_length())
                            blocks_per_long = 64 // bits_per_block
                            mask = (1 << bits_per_block) - 1

                            # Vectorized unpacking using numpy
                            data_array = np.array(data, dtype=np.int64)

                            # Pre-compute indices for all 4096 blocks
                            indices = np.arange(4096, dtype=np.int32)
                            long_indices = indices // blocks_per_long
                            bit_indices = (indices % blocks_per_long) * bits_per_block

                            # Extract all palette indices at once
                            valid_mask = long_indices < len(data_array)
                            palette_indices = np.zeros(4096, dtype=np.int32)
                            palette_indices[valid_mask] = (data_array[long_indices[valid_mask]] >> bit_indices[valid_mask]) & mask

                            # Filter valid non-air blocks
                            valid_blocks = (palette_indices >= 0) & (palette_indices < len(palette))

                            # Compute dx, dy, dz for all blocks (index = dy*256 + dz*16 + dx)
                            dx_coords = indices % 16
                            dz_coords = (indices // 16) % 16
                            dy_coords = indices // 256

                            # Set blocks
                            for i in np.where(valid_blocks)[0]:
                                palette_index = int(palette_indices[i])
                                block_name = palette[palette_index].get('Name', 'minecraft:air')
                                if block_name != 'minecraft:air':
                                    x = chunkX * 16 + int(dx_coords[i])
                                    y = section_y * 16 + int(dy_coords[i])
                                    z = chunkZ * 16 + int(dz_coords[i])
                                    blocks[(x, y, z)] = block_name
        except Exception as e:
            print(f"[Replay] Error parsing chunk NBT: {e}")

        return blocks

    def get_player_state(self, tick: int) -> dict | None:
        """Get player state at a given tick."""
        if 0 <= tick < len(self.ticks):
            return self.ticks[tick]
        return None

    def get_initial_player_pos(self) -> tuple[float, float, float]:
        """Get player position at tick 0."""
        if self.ticks:
            p = self.ticks[0]['player']
            return (p['x'], p['y'], p['z'])
        return (0.0, 64.0, 0.0)

    def get_blocks_at_tick(self, tick: int) -> dict:
        """
        Get all blocks visible at a given tick.
        Start with chunks, apply block changes up to this tick.
        Returns dict: (x,y,z) -> blockId
        """
        # Start with all chunk blocks
        blocks = {}
        for chunk_blocks in self.chunks.values():
            blocks.update(chunk_blocks)

        # Apply block changes up to this tick
        for t in range(tick + 1):
            for change in self.block_changes_by_tick.get(t, []):
                x = change['x']
                y = change['y']
                z = change['z']
                block_id = change['blockId']

                if block_id == 'minecraft:air':
                    blocks.pop((x, y, z), None)  # Remove block
                else:
                    blocks[(x, y, z)] = block_id  # Add/update block

        return blocks

    def get_all_unique_block_ids(self) -> set[str]:
        """Scan all chunks and block changes to find unique block IDs."""
        block_ids = set()

        # From chunks
        for chunk_blocks in self.chunks.values():
            for block_id in chunk_blocks.values():
                if block_id != 'minecraft:air':
                    block_ids.add(block_id)

        # From block changes
        for changes in self.block_changes_by_tick.values():
            for change in changes:
                bid = change.get('blockId', '')
                if bid and bid != 'minecraft:air':
                    block_ids.add(bid)

        return block_ids

    def get_all_unique_block_variants(self) -> set[str]:
        """Alias for get_all_unique_block_ids (for compatibility)."""
        return self.get_all_unique_block_ids()

    def get_tick_for_frame(self, frame_index: int) -> int:
        """Convert video frame index to tick number."""
        if self.has_frame_mapping and 0 <= frame_index < len(self.frame_to_tick):
            return self.frame_to_tick[frame_index]
        # Fallback: assume frame rate from metadata
        target_fps = self.metadata.get('config', {}).get('targetFps', 20)
        return int(frame_index * 20.0 / target_fps)
