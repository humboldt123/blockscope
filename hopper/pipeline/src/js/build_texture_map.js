'use strict';
/**
 * build_texture_map.js
 *
 * Resolves per-face texture names for every Minecraft 1.19.4 block state.
 * Output: pipeline/data/texture_map_1.19.4.json
 *   { "version": "1.19.4", "states": [ {up,down,north,south,east,west}, ... ] }
 *   Indexed by block-state ID (same IDs as blockstate_table_1.19.4.json).
 *   Each face value is a bare texture name like "stone" or "oak_planks" —
 *   load from visualizer/assets/minecraft/textures/block/<name>.png.
 *
 * Run:
 *   node pipeline/src/js/build_texture_map.js [assets_dir]
 *   assets_dir defaults to $MC_ASSETS_DIR or <repo_root>/visualizer/assets/minecraft
 */

const fs   = require('fs');
const path = require('path');

const MC_VERSION    = '1.19.4';
const PIPELINE_ROOT = path.resolve(__dirname, '..', '..');
const BST_PATH      = path.join(PIPELINE_ROOT, 'data', 'blockstate_table_1.19.4.json');
const OUT_PATH      = path.join(PIPELINE_ROOT, 'data', 'texture_map_1.19.4.json');

const ASSETS_DIR = process.argv[2] ||
    process.env.MC_ASSETS_DIR ||
    path.resolve(PIPELINE_ROOT, '..', '..', 'visualizer', 'assets', 'minecraft');

const mcData = require('minecraft-data')(MC_VERSION);

// ── Model resolution (with cache) ────────────────────────────────────────────

const modelCache = new Map();   // filePath → {textures, faceTexRefs}

function resolveModel(modelRef, depth = 0) {
    if (!modelRef || depth > 12) return { textures: {}, faceTexRefs: {} };

    const parts = modelRef.split(':');
    const rel   = (parts.length > 1 ? parts[1] : parts[0])
                    .replace(/^block\//, '');
    const filePath = path.join(ASSETS_DIR, 'models', 'block', rel + '.json');

    if (modelCache.has(filePath)) return modelCache.get(filePath);

    // Placeholder to break cycles
    const placeholder = { textures: {}, faceTexRefs: {}, faceTintIdx: {}, faceRotation: {}, faceUV: {} };
    modelCache.set(filePath, placeholder);

    if (!fs.existsSync(filePath)) return placeholder;

    let model;
    try { model = JSON.parse(fs.readFileSync(filePath, 'utf8')); }
    catch (e) { return placeholder; }

    // Resolve parent
    const parent = model.parent ? resolveModel(model.parent, depth + 1)
                                : { textures: {}, faceTexRefs: {}, faceTintIdx: {}, faceRotation: {}, faceUV: {} };

    // Merge textures: parent first, child overrides
    const textures    = { ...parent.textures, ...(model.textures || {}) };
    const faceTexRefs = { ...parent.faceTexRefs };
    const faceTintIdx = { ...parent.faceTintIdx };
    const faceRotation = { ...parent.faceRotation };
    const faceUV       = { ...parent.faceUV };

    // Read elements: first element wins per direction.
    // Overlay elements (e.g. grass_block_side_overlay) come second and would
    // replace the base texture with a partially-transparent PNG — causing the
    // dirt/base to disappear.  First-wins keeps the base texture visible.
    if (model.elements) {
        for (const elem of model.elements) {
            for (const [dir, face] of Object.entries(elem.faces || {})) {
                if (face.texture !== undefined && faceTexRefs[dir] === undefined)
                    faceTexRefs[dir] = face.texture;
                if (face.tintindex !== undefined && faceTintIdx[dir] === undefined)
                    faceTintIdx[dir] = face.tintindex;
                if (face.rotation !== undefined && faceRotation[dir] === undefined)
                    faceRotation[dir] = face.rotation;
                if (face.uv !== undefined && faceUV[dir] === undefined) {
                    const [u1,v1,u2,v2] = face.uv;
                    if (u1 !== 0 || v1 !== 0 || u2 !== 16 || v2 !== 16)
                        faceUV[dir] = [u1/16, v1/16, u2/16, v2/16];
                }
            }
        }
    }

    const result = { textures, faceTexRefs, faceTintIdx, faceRotation, faceUV };
    modelCache.set(filePath, result);
    return result;
}

// ── Texture ref resolver ──────────────────────────────────────────────────────

function resolveRef(ref, textures, visited) {
    if (!ref) return null;
    visited = visited || new Set();
    if (visited.has(ref)) return null;
    visited.add(ref);
    if (ref.startsWith('#')) {
        return resolveRef(textures[ref.slice(1)], textures, visited);
    }
    // Strip namespace, e.g. "minecraft:block/stone" → "stone"
    return ref.replace(/^minecraft:block\//, '').replace(/^block\//, '');
}

// ── State property computation ────────────────────────────────────────────────

function computeStateProps(statesDefs, offset) {
    let stride = 1;
    const props = {};
    for (let i = statesDefs.length - 1; i >= 0; i--) {
        const def = statesDefs[i];
        const values = def.values ||
            (def.type === 'bool'
                ? ['true', 'false']
                : Array.from({ length: def.num_values }, (_, j) => String(j)));
        const idx = Math.floor(offset / stride) % def.num_values;
        props[def.name] = String(values[idx]);
        stride *= def.num_values;
    }
    return props;
}

// ── Variant matching ──────────────────────────────────────────────────────────

function variantMatches(variantKey, props) {
    if (!variantKey) return true;
    for (const pair of variantKey.split(',')) {
        const eq  = pair.indexOf('=');
        const k   = pair.slice(0, eq);
        const v   = pair.slice(eq + 1);
        if (props[k] !== v) return false;
    }
    return true;
}

// ── Blockstates file cache ────────────────────────────────────────────────────

const bsCache = new Map();

function loadBlockstates(blockName) {
    if (bsCache.has(blockName)) return bsCache.get(blockName);
    const p = path.join(ASSETS_DIR, 'blockstates', blockName + '.json');
    if (!fs.existsSync(p)) { bsCache.set(blockName, null); return null; }
    try {
        const v = JSON.parse(fs.readFileSync(p, 'utf8'));
        bsCache.set(blockName, v);
        return v;
    } catch (e) { bsCache.set(blockName, null); return null; }
}

// ── Face rotation ─────────────────────────────────────────────────────────────

// Y rotation (CW from above, degrees): world_dir = model face that rotated to that world dir.
// After CW rotation by Y, a model face pointing in `from` ends up pointing in `to`:
//   y=90:  N→W  E→N  S→E  W→S
//   y=180: N→S  E→W  S→N  W→E
//   y=270: N→E  E→S  S→W  W→N
// So world_north = model face that now points north = model_east (for y=90).
const Y_ROT = {
    90:  { north:'east', east:'south', south:'west', west:'north' },
    180: { north:'south', east:'west', south:'north', west:'east' },
    270: { north:'west', east:'north', south:'east', west:'south' },
};

// X rotation (CW from east, degrees):
//   x=90:  up→N  N→down  down→S  S→up   (tilting top toward north)
//   x=180: up→down  down→up  N→S  S→N
//   x=270: up→S  S→down  down→N  N→up
const X_ROT = {
    90:  { up:'north', north:'down', down:'south', south:'up' },
    180: { up:'down',  down:'up',    north:'south', south:'north' },
    270: { up:'south', south:'down', down:'north', north:'up' },
};

function applyRotation(faces, x, y) {
    let f = { ...faces };
    if (y && Y_ROT[y]) {
        const r = Y_ROT[y];
        f = { ...f, north:f[r.north], east:f[r.east], south:f[r.south], west:f[r.west] };
    }
    if (x && X_ROT[x]) {
        const r = X_ROT[x];
        const next = { ...f };
        for (const [to, from] of Object.entries(r)) next[to] = f[from];
        // east/west unchanged by X rotation, up/down/north/south may change
        f = next;
    }
    return f;
}

// ── Main per-state lookup ─────────────────────────────────────────────────────

const DIRS = ['down', 'up', 'north', 'south', 'east', 'west'];

// Resolve face textures from a single model entry.
// fill=true: fill missing cardinal/vertical faces from neighbouring sides (used for full-cube variants).
// fill=false: only return faces explicitly defined in model elements (used for individual multipart parts).
function resolveFacesFromEntry(modelEntry, fill) {
    if (!modelEntry || !modelEntry.model) return null;

    const { textures, faceTexRefs, faceTintIdx, faceRotation, faceUV } = resolveModel(modelEntry.model);

    let result = {};
    let tints  = {};
    let rots   = {};
    let uvs    = {};

    if (Object.keys(faceTexRefs).length > 0) {
        for (const dir of DIRS) {
            const ref = faceTexRefs[dir];
            result[dir] = ref ? resolveRef(ref, textures) : null;
            tints[dir]  = faceTintIdx[dir] ?? -1;
            rots[dir]   = faceRotation[dir] ?? 0;
            uvs[dir]    = faceUV[dir] ?? null;
        }
        if (fill) {
            const side     = result.north || result.south || result.east || result.west;
            const sideTint = tints.north ?? tints.south ?? tints.east ?? tints.west ?? -1;
            const sideRot  = rots.north  ?? rots.south  ?? rots.east  ?? rots.west  ?? 0;
            for (const dir of ['north', 'south', 'east', 'west']) {
                if (!result[dir]) { result[dir] = side; tints[dir] = sideTint; rots[dir] = sideRot; }
            }
            if (!result.up)   { result.up   = result.down  || side; tints.up   = tints.down   ?? sideTint; rots.up   = rots.down  ?? 0; }
            if (!result.down) { result.down = result.up    || side; tints.down = tints.up     ?? sideTint; rots.down = rots.up    ?? 0; }
        }
    } else if (fill) {
        const any  = Object.values(textures)[0];
        const side = textures.side || textures.all || any;
        result.up    = resolveRef(textures.up   || textures.top    || textures.end || side, textures);
        result.down  = resolveRef(textures.down || textures.bottom || textures.end || side, textures);
        for (const dir of ['north', 'south', 'east', 'west']) {
            result[dir] = resolveRef(textures[dir] || side, textures);
        }
        for (const dir of DIRS) { tints[dir] = -1; rots[dir] = 0; uvs[dir] = null; }
    }

    // Apply blockstate model rotation (Y then X)
    const rotY = (modelEntry.y || 0) % 360;
    const rotX = (modelEntry.x || 0) % 360;
    if (rotY || rotX) {
        result = applyRotation(result, rotX, rotY);
        tints  = applyRotation(tints,  rotX, rotY);
        rots   = applyRotation(rots,   rotX, rotY);
        uvs    = applyRotation(uvs,    rotX, rotY);
    }

    const out = {};
    for (const dir of DIRS) {
        if (result[dir] !== undefined) {
            out[dir] = result[dir] || null;
            if (tints[dir] >= 0) out[dir + '_t'] = 1;
            if (rots[dir])       out[dir + '_r'] = rots[dir];
            if (uvs[dir])        out[dir + '_uv'] = uvs[dir];
        }
    }
    return Object.keys(out).length > 0 ? out : null;
}

function getFaceTextures(blockName, props) {
    const bs = loadBlockstates(blockName);
    if (!bs) return null;

    if (bs.variants) {
        let bestLen = -1, modelEntry = null;
        for (const [key, val] of Object.entries(bs.variants)) {
            if (variantMatches(key, props)) {
                const len = key ? key.split(',').length : 0;
                if (len > bestLen) {
                    bestLen    = len;
                    modelEntry = Array.isArray(val) ? val[0] : val;
                }
            }
        }
        return modelEntry ? resolveFacesFromEntry(modelEntry, true) : null;

    } else if (bs.multipart) {
        // Collect all matching parts; merge their face data (first non-null per direction wins).
        // Individual parts use fill=false so arms/posts don't bleed into unrelated faces.
        const merged = {};
        for (const part of bs.multipart) {
            const cond = part.when;
            if (!cond || conditionMatches(cond, props)) {
                const apply = part.apply;
                const entry = Array.isArray(apply) ? apply[0] : apply;
                const partial = resolveFacesFromEntry(entry, false);
                if (!partial) continue;
                for (const dir of DIRS) {
                    if (merged[dir] == null && partial[dir] != null) {
                        merged[dir] = partial[dir];
                        if (partial[dir + '_t'])  merged[dir + '_t']  = 1;
                        if (partial[dir + '_r'])  merged[dir + '_r']  = partial[dir + '_r'];
                        if (partial[dir + '_uv']) merged[dir + '_uv'] = partial[dir + '_uv'];
                    }
                }
            }
        }
        // Ensure all DIRS present
        for (const dir of DIRS) { if (!(dir in merged)) merged[dir] = null; }
        return Object.values(merged).some(v => v) ? merged : null;
    }

    return null;
}

function conditionMatches(cond, props) {
    // Handles simple {key: val} and {OR: [...]}
    if (cond.OR) return cond.OR.some(c => conditionMatches(c, props));
    if (cond.AND) return cond.AND.every(c => conditionMatches(c, props));
    for (const [k, v] of Object.entries(cond)) {
        // v can be "true|false" or a single value
        const vals = String(v).split('|');
        if (!vals.includes(props[k])) return false;
    }
    return true;
}

// ── Main ──────────────────────────────────────────────────────────────────────

function main() {
    console.log(`Assets dir: ${ASSETS_DIR}`);
    if (!fs.existsSync(path.join(ASSETS_DIR, 'blockstates'))) {
        console.error('ERROR: blockstates dir not found under assets dir');
        process.exit(1);
    }

    const bst    = JSON.parse(fs.readFileSync(BST_PATH, 'utf8'));
    const output = new Array(bst.states.length).fill(null);

    let resolved = 0, fallback = 0;

    for (const block of Object.values(mcData.blocks)) {
        const statesDefs = block.states || [];

        for (let sid = block.minStateId; sid <= block.maxStateId; sid++) {
            const props  = computeStateProps(statesDefs, sid - block.minStateId);
            const result = getFaceTextures(block.name, props);

            if (result && Object.values(result).some(v => v)) {
                output[sid] = result;
                resolved++;
            } else {
                // Fallback: guess from block name (usually works for simple blocks)
                const guess = block.name;
                output[sid] = { down: guess, up: guess, north: guess, south: guess, east: guess, west: guess };
                fallback++;
            }
        }
    }

    console.log(`Resolved: ${resolved}  Fallback: ${fallback}`);

    fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
    fs.writeFileSync(OUT_PATH, JSON.stringify({ version: MC_VERSION, states: output }));
    const kb = (fs.statSync(OUT_PATH).size / 1024).toFixed(0);
    console.log(`Wrote ${OUT_PATH}  (${kb} KB)`);
}

main();
