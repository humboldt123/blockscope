# mc_converter

Converts Minecraft 1.8 world saves to 1.19.4 format without triggering DFU's terrain generation (no deepslate, no new caves, no blending). Chunks outside the original world generate as void (handled by VoidGenerator plugin in Paper).

## Build

```bash
./gradlew jar   # downloads WorldEdit legacy.json + builds fat jar
```

Output: `build/libs/mc-converter.jar`

## Usage

```bash
java -jar build/libs/mc-converter.jar --input <world_dir> --output <out_dir>
```

Converts `region/`, `DIM-1/region/`, `DIM1/region/`, `entities/`, and `level.dat`. Copies `data/`, `datapacks/`, `playerdata/`, `stats/`, etc. as-is.

## Known warnings on load (all benign for training use)

- **`Skipping BlockEntity with id minecraft:control`** — `"Control"` is the 1.8 NBT ID for command blocks. The block itself is preserved; only the command block's stored command is lost. Not mapped in `BlockEntityMapper` because command blocks are irrelevant to training data.
- **`Skipping BlockEntity with id minecraft:record_player`** — `"RecordPlayer"` is the 1.8 ID for jukeboxes. Block preserved, disc data lost.
- **`Skipping Entity with id minecraft:itemframe`** — Entity ID mapping drops some item frames. Visual only.
- **`Ignoring unknown attribute generic.maxHealth` etc.** — Pre-1.16 mob attribute names. Mobs load fine, custom attribute values are lost.
- **Lighting appears dim on first load** — Chunks are written with `Status: "features"` (matching DFU behaviour), so Minecraft relights them on first visit. Brightens up as you explore.

## How it works

- Block IDs: WorldEdit's `legacy.json` maps `(numericID:meta)` → namespaced 1.19.4 block state
- 24 sections per chunk (Y=-4 to Y=19); sections not in the original 1.8 world are written as air
- `DataVersion=3337` on every chunk and `level.dat` — Paper skips DFU entirely
- `blending_data` intentionally omitted — prevents deepslate/cave generation at y<0
- `level.dat` gets a full `WorldGenSettings` compound built from the old `RandomSeed` value
