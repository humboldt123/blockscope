/**
 * build_state_table.js
 *
 * One-time script: reads minecraft-data for 1.19.4 and emits
 *   pipeline/data/blockstate_table_1.19.4.json
 *
 * Schema:
 *   {
 *     "version": "1.19.4",
 *     "max_state_id": <int>,
 *     "states": [                       // indexed by block-state ID
 *       { "name": "minecraft:stone", "opaque": true, "aabbs": [] },
 *       ...
 *     ]
 *   }
 *
 * "opaque" means: a ray that hits this block's AABB terminates (does not pass through).
 * Transparent blocks (glass, leaves, water, ice, etc.) are NOT opaque — rays continue.
 * Blocks with no AABBs (air, fluids) are never hit by rays.
 *
 * Run: node pipeline/src/js/build_state_table.js
 */

'use strict';

const fs   = require('fs');
const path = require('path');

const MC_VERSION = '1.19.4';

// Resolve paths relative to the pipeline/ directory (two levels up from src/js/).
const pipelineRoot = path.resolve(__dirname, '..', '..');
const outPath = path.join(pipelineRoot, 'data', 'blockstate_table_1.19.4.json');
const overridePath = path.join(pipelineRoot, 'data', 'opacity_overrides.json');

const mcData = require('minecraft-data')(MC_VERSION);

// ---------- helpers ----------------------------------------------------------

/**
 * Read the opacity-overrides file (if it exists).
 * Returns a Map<blockName, bool> of manual opaque settings.
 */
function loadOverrides() {
  const result = new Map();
  if (!fs.existsSync(overridePath)) return result;
  const obj = JSON.parse(fs.readFileSync(overridePath, 'utf8'));
  for (const [name, opaque] of Object.entries(obj)) {
    result.set(name, opaque);
  }
  return result;
}

/**
 * Determine whether a block is "opaque" for raycasting purposes.
 *
 * Rule: opaque = NOT visually transparent.
 * We use blocks.json `transparent` field as the primary signal, since it
 * aligns well with blocks that let vision pass (glass, leaves, water, ice…).
 * Blocks with boundingBox=="empty" have no geometry; they're never hit so
 * opacity doesn't matter, but we set opaque=false for clarity.
 *
 * Manual overrides in opacity_overrides.json take precedence.
 */
function isOpaque(block, overrides) {
  if (overrides.has(block.name)) return overrides.get(block.name);
  if (block.boundingBox === 'empty') return false;   // no geometry
  // minecraft-data `transparent` flag: true = light passes through.
  // That correlates well with "vision passes through" for our purposes.
  return !block.transparent;
}

// ---------- main -------------------------------------------------------------

function buildTable() {
  const overrides  = loadOverrides();
  const bcs        = mcData.blockCollisionShapes;   // { blocks, shapes }
  const allBlocks  = mcData.blocks;                  // indexed by block ID (0..N)
  const shapes     = bcs.shapes;                     // shapeIndex -> [[x0,y0,z0,x1,y1,z1],...]

  // Find the maximum state ID across all blocks
  let maxStateId = 0;
  for (const block of Object.values(allBlocks)) {
    if (block.maxStateId > maxStateId) maxStateId = block.maxStateId;
  }

  console.log(`Building table for ${MC_VERSION}: ${Object.keys(allBlocks).length} blocks, max state ID ${maxStateId}`);

  // Pre-allocate output array (one entry per state ID)
  const states = new Array(maxStateId + 1).fill(null);

  for (const block of Object.values(allBlocks)) {
    const { name, minStateId, maxStateId: blockMax } = block;
    const fullName = `minecraft:${name}`;
    const opaque   = isOpaque(block, overrides);

    // Collision shapes for this block
    const shapeEntry = bcs.blocks[name];   // single index OR array of indices

    for (let stateId = minStateId; stateId <= blockMax; stateId++) {
      const stateIndex = stateId - minStateId;

      let shapeIdx;
      if (Array.isArray(shapeEntry)) {
        // One shape per state; clamp to avoid out-of-bounds on malformed data
        shapeIdx = shapeEntry[Math.min(stateIndex, shapeEntry.length - 1)];
      } else if (shapeEntry !== undefined) {
        shapeIdx = shapeEntry;
      } else {
        // No collision data for this block → treat as empty
        shapeIdx = null;
      }

      const aabbs = (shapeIdx !== null && shapeIdx !== undefined)
        ? (shapes[shapeIdx] || [])
        : [];

      states[stateId] = { name: fullName, opaque, aabbs };
    }
  }

  // Fill any gaps (shouldn't happen but be safe)
  for (let i = 0; i <= maxStateId; i++) {
    if (!states[i]) {
      states[i] = { name: 'minecraft:unknown', opaque: false, aabbs: [] };
    }
  }

  const table = { version: MC_VERSION, max_state_id: maxStateId, states };

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(table));
  const sizeKB = (fs.statSync(outPath).size / 1024).toFixed(1);
  console.log(`Wrote ${outPath}  (${sizeKB} KB)`);

  return table;
}

function verify(table) {
  const check = (name, stateOffset, label) => {
    const block = mcData.blocksByName[name];
    if (!block) { console.warn(`  [WARN] block not found: ${name}`); return; }
    const stateId = block.minStateId + stateOffset;
    const entry   = table.states[stateId];
    console.log(`  ${label} (stateId ${stateId}): opaque=${entry.opaque}  aabbs=${JSON.stringify(entry.aabbs)}`);
  };

  console.log('\n--- Verification ---');
  // oak_stairs: 80 states, shape varies by facing/half/shape
  check('oak_stairs',  0, 'oak_stairs[0]');     // first state
  check('oak_stairs', 40, 'oak_stairs[40]');    // mid state
  // oak_slab: 6 states (bottom / top / double × 2 waterlogged variants)
  check('oak_slab', 0, 'oak_slab[0] (bottom)');
  check('oak_slab', 2, 'oak_slab[2] (top)');
  check('oak_slab', 4, 'oak_slab[4] (double)');
  // oak_fence: 32 states
  check('oak_fence', 0, 'oak_fence[0]');
  // stone and glass
  check('stone',  0, 'stone');
  check('glass',  0, 'glass');
  check('water',  0, 'water');
  check('air',    0, 'air');
  check('oak_leaves', 0, 'oak_leaves');
}

const table = buildTable();
verify(table);
