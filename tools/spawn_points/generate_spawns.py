#!/usr/bin/env python3
"""
Spawn-point generator for the Blockscope Minecraft data-collection world library.

Produces, for a converted world directory, a file `spawn_candidates.json` of the form:
    {"spawns": [[x, y, z], [x, y, z], ...]}
where each entry is an integer FEET block position (the first air block above solid
ground) at which a player can validly stand to begin an exploration session.

Two strategies, auto-selected per world:

  A. Hermitcraft worlds -- if `playerdata/*.dat` exists, parse real hermit logout
     positions (overworld only), floor to int block coords, dedupe within ~8 blocks.
     If that yields < MIN_SPAWNS points, top up with strategy B.

  B. Build maps (or top-up) -- grid-scan the populated region/*.mca chunks, find the
     highest non-air block per sampled column, and emit (x, top_y+1, z) after filtering
     out void margins, non-standable spots, and fluid feet.

Usage:
    python3 generate_spawns.py --world /path/to/world [options]

The script writes `spawn_candidates.json` into the world dir. It is re-runnable.

Dependencies: nbtlib (for level.dat / playerdata). Region (.mca) parsing is hand-rolled
(post-1.13 Anvil container, 1.16+ paletted sections with padded longs, no spanning).
"""

import argparse
import glob
import io
import json
import math
import os
import struct
import sys
import zlib

import nbtlib

# ---- tunables -------------------------------------------------------------
MIN_SPAWNS = 10            # hard minimum candidates per world
MAX_SPAWNS = 200           # cap for large build maps
DEDUPE_DIST = 8            # hermits within this many blocks are merged
GRID_STEP = 32             # default (x,z) sampling step for build maps (blocks)
MIN_COLUMN_BLOCKS = 5      # a column with fewer non-air blocks is void/edge
MIN_CHUNK_BLOCKS = 256     # a chunk with fewer non-air blocks is void/edge

AIR_BLOCKS = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}
FLUID_BLOCKS = {"minecraft:water", "minecraft:lava",
                "minecraft:flowing_water", "minecraft:flowing_lava"}


# ---- NBT helpers ----------------------------------------------------------
def _root(nbt):
    """nbtlib wraps the unnamed root compound under key '' -- unwrap it."""
    return nbt[""] if "" in nbt else nbt


def world_dataversion(world_dir):
    lv = nbtlib.load(os.path.join(world_dir, "level.dat"))
    data = _root(lv).get("Data", _root(lv))
    dv = data.get("DataVersion")
    return int(dv) if dv is not None else None


# ---- strategy A: hermit playerdata ---------------------------------------
def _is_overworld(dim):
    if dim is None:
        return True  # absent => overworld in this era
    # int form: 0 overworld, -1 nether, 1 end
    try:
        return int(dim) == 0
    except (TypeError, ValueError):
        return str(dim) == "minecraft:overworld"


_name_cache = {}


def resolve_name(uuid_with_dashes):
    """UUID -> player name via Mojang sessionserver; falls back to a short uuid tag."""
    if uuid_with_dashes in _name_cache:
        return _name_cache[uuid_with_dashes]
    u = uuid_with_dashes.replace("-", "")
    name = "hermit_" + u[:8]
    try:
        import urllib.request
        url = f"https://sessionserver.mojang.com/session/minecraft/profile/{u}"
        with urllib.request.urlopen(url, timeout=5) as r:
            name = json.load(r).get("name", name)
    except Exception:
        pass
    _name_cache[uuid_with_dashes] = name
    return name


def collect_player_spawns(world_dir):
    """Return labeled hermit logout positions: (x, y, z, hermit_name) tuples."""
    pts = []
    for pf in sorted(glob.glob(os.path.join(world_dir, "playerdata", "*.dat"))):
        try:
            root = _root(nbtlib.load(pf))
        except Exception as e:
            print(f"  WARN: could not read {os.path.basename(pf)}: {e}", file=sys.stderr)
            continue
        if not _is_overworld(root.get("Dimension")):
            continue
        pos = root.get("Pos")
        if pos is None or len(pos) != 3:
            continue
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        uuid = os.path.splitext(os.path.basename(pf))[0]
        name = resolve_name(uuid)
        pts.append((int(math.floor(x)), int(math.floor(y)), int(math.floor(z)), name))
    return dedupe(pts, DEDUPE_DIST)


def dedupe(points, dist):
    kept = []
    for p in points:
        if all(abs(p[0] - q[0]) > dist or abs(p[1] - q[1]) > dist or
               abs(p[2] - q[2]) > dist for q in kept):
            kept.append(p)
    return kept


# ---- Anvil region parsing -------------------------------------------------
def iter_region_chunks(region_path):
    """Yield decoded chunk root compounds from one .mca file."""
    with open(region_path, "rb") as f:
        header = f.read(4096)
        if len(header) < 4096:
            return
        for i in range(1024):
            off = struct.unpack(">I", b"\x00" + header[i * 4:i * 4 + 3])[0]
            if off == 0:
                continue
            f.seek(off * 4096)
            sz = f.read(4)
            if len(sz) < 4:
                continue
            length = struct.unpack(">I", sz)[0]
            if length == 0:
                continue
            comp = f.read(1)[0]
            raw = f.read(length - 1)
            try:
                if comp == 1:
                    data = zlib.decompress(raw, 16 + zlib.MAX_WBITS)  # gzip
                elif comp == 2:
                    data = zlib.decompress(raw)  # zlib
                elif comp == 3:
                    data = raw  # uncompressed
                else:
                    continue
                nbt = nbtlib.File.parse(io.BytesIO(data))
            except Exception:
                continue
            yield _root(nbt)


def _unpack_section(block_states):
    """Return (palette_names, indices[4096]) for a section's block_states, or None.

    1.16+ layout: data is an array of longs, each holds floor(64/bits) entries with
    NO spanning across longs (padded). bits = max(4, ceil(log2(palette_len))).
    Index order is YZX (y*256 + z*16 + x).
    """
    pal = block_states.get("palette")
    if pal is None:
        return None
    names = [str(b.get("Name")) for b in pal]
    if len(names) == 1:
        return names, None  # single-block section; indices all 0
    data = block_states.get("data")
    if data is None:
        return names, None
    bits = max(4, (len(names) - 1).bit_length())
    per_long = 64 // bits
    mask = (1 << bits) - 1
    indices = [0] * 4096
    idx = 0
    for word in data:
        w = int(word) & 0xFFFFFFFFFFFFFFFF
        for _ in range(per_long):
            if idx >= 4096:
                break
            indices[idx] = w & mask
            w >>= bits
            idx += 1
        if idx >= 4096:
            break
    return names, indices


def chunk_columns(root):
    """Build a per-column top-surface summary for one chunk.

    Returns dict keyed by (world_x, world_z) -> (top_y, feet_block, head_block,
    nonair_count) for the highest non-air block in that column, or None if the
    chunk is too empty (void margin).
    """
    cx = int(root.get("xPos"))
    cz = int(root.get("zPos"))
    sections = root.get("sections")
    if not sections:
        return None

    # decode sections, lowest Y first
    decoded = []  # (section_y, names, indices)
    total_nonair = 0
    for s in sections:
        bs = s.get("block_states")
        if bs is None:
            continue
        sy = int(s.get("Y"))
        res = _unpack_section(bs)
        if res is None:
            continue
        names, indices = res
        if indices is None and names[0] in AIR_BLOCKS:
            continue  # all-air section, skip
        decoded.append((sy, names, indices))

    if not decoded:
        return None
    decoded.sort(key=lambda t: t[0])

    # per local column (lx,lz): walk from top section downwards to find highest non-air
    cols = {}
    for lx in range(16):
        for lz in range(16):
            top_y = None
            feet = head = None
            nonair = 0
            # iterate sections high->low
            for sy, names, indices in reversed(decoded):
                base_y = sy * 16
                for ly in range(15, -1, -1):
                    if indices is None:
                        name = names[0]
                    else:
                        name = names[indices[ly * 256 + lz * 16 + lx]]
                    if name in AIR_BLOCKS:
                        continue
                    nonair += 1
                    if top_y is None:
                        top_y = base_y + ly
                        feet = name  # block AT surface (the standable solid)
            if top_y is None or nonair < MIN_COLUMN_BLOCKS:
                continue
            wx = cx * 16 + lx
            wz = cz * 16 + lz
            cols[(wx, wz)] = (top_y, feet, nonair)
    # require chunk-level density too
    chunk_nonair = sum(v[2] for v in cols.values())
    if chunk_nonair < MIN_CHUNK_BLOCKS:
        return None
    return cols


def collect_grid_spawns(world_dir, step=GRID_STEP, limit=MAX_SPAWNS, exclude=None):
    """Scan region files, sampling one column per `step` blocks, return feet positions."""
    region_dir = os.path.join(world_dir, "region")
    region_files = sorted(glob.glob(os.path.join(region_dir, "*.mca")))
    exclude = exclude or []

    # First pass: gather surface columns for sampled (x,z) positions only.
    samples = {}  # (wx,wz) -> top_y
    for rf in region_files:
        for root in iter_region_chunks(rf):
            try:
                cols = chunk_columns(root)
            except Exception:
                continue
            if not cols:
                continue
            for (wx, wz), (top_y, feet, nonair) in cols.items():
                if wx % step != 0 or wz % step != 0:
                    continue
                if feet in FLUID_BLOCKS:
                    continue  # standing on fluid surface -> skip
                # feet position is top_y+1; require it (and head) plausibly air:
                # since top_y is the HIGHEST non-air, top_y+1 and +2 are air by definition.
                samples[(wx, wz)] = top_y + 1

    spawns = [(wx, fy, wz) for (wx, wz), fy in samples.items()]

    # drop ones near excluded (already-have) points
    if exclude:
        spawns = [s for s in spawns
                  if all(abs(s[0] - e[0]) > DEDUPE_DIST or abs(s[2] - e[2]) > DEDUPE_DIST
                         for e in exclude)]

    # spread + cap: sort for determinism, then thin to limit
    spawns.sort()
    if len(spawns) > limit:
        stride = len(spawns) / limit
        spawns = [spawns[int(i * stride)] for i in range(limit)]
    return spawns


# ---- main -----------------------------------------------------------------
def generate(world_dir, step=GRID_STEP):
    dv = world_dataversion(world_dir)
    if dv is None:
        raise SystemExit(f"{world_dir}: level.dat has no DataVersion -- not a converted world")
    print(f"World: {world_dir}  (DataVersion {dv})")

    has_playerdata = bool(glob.glob(os.path.join(world_dir, "playerdata", "*.dat")))
    hermits = []  # (x,y,z,name) tuples
    if has_playerdata:
        hermits = collect_player_spawns(world_dir)
        print(f"  strategy A (playerdata): {len(hermits)} hermit points after dedupe")

    # Always add grid spawns for broader coverage. For hermitcraft this gives many
    # more spawns than just the hermit logout spots; for build maps it's the main source.
    grid_cap = MAX_SPAWNS if not hermits else max(MIN_SPAWNS * 4, MAX_SPAWNS - len(hermits))
    print(f"  strategy B (grid scan): scanning regions (have {len(hermits)} hermit pts)...")
    grid = collect_grid_spawns(world_dir, step=step, limit=grid_cap, exclude=hermits)
    grid = [(x, y, z, "grid") for (x, y, z) in grid]
    print(f"  strategy B produced {len(grid)} grid points")

    spawns = dedupe(hermits + grid, DEDUPE_DIST)

    # Cap, but keep ALL hermit points (they're the highest-value spawns) and thin grid.
    if len(spawns) > MAX_SPAWNS:
        hermit_pts = [s for s in spawns if s[3] != "grid"]
        grid_pts   = [s for s in spawns if s[3] == "grid"]
        room = max(0, MAX_SPAWNS - len(hermit_pts))
        if room and len(grid_pts) > room:
            stride = len(grid_pts) / room
            grid_pts = [grid_pts[int(i * stride)] for i in range(room)]
        elif room == 0:
            grid_pts = []
        spawns = hermit_pts + grid_pts

    out = {
        "spawns": [[int(x), int(y), int(z)] for (x, y, z, _) in spawns],
        "labels": [lbl for (_, _, _, lbl) in spawns],
    }
    out_path = os.path.join(world_dir, "spawn_candidates.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    n_hermit = sum(1 for s in spawns if s[3] != "grid")
    print(f"  wrote {len(out['spawns'])} candidates ({n_hermit} hermit, {len(spawns)-n_hermit} grid) -> {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate spawn_candidates.json for a converted world")
    ap.add_argument("--world", required=True, help="path to converted world dir (contains level.dat)")
    ap.add_argument("--step", type=int, default=GRID_STEP, help="grid sampling step in blocks (build maps)")
    args = ap.parse_args()
    if not os.path.isfile(os.path.join(args.world, "level.dat")):
        raise SystemExit(f"{args.world}: no level.dat found")
    generate(args.world, step=args.step)


if __name__ == "__main__":
    main()
