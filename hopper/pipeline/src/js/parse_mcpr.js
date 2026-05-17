/**
 * parse_mcpr.js  —  Stage 1: .mcpr → world_states.bin
 *
 * Usage:
 *   node src/js/parse_mcpr.js <session_dir> <out_dir>
 *
 * Reads the .mcpr file inside <session_dir>, replays the Minecraft 1.19.4
 * protocol packet stream, reconstructs world state incrementally, then for
 * each tick in ticks.jsonl snapshots the 32³ block window centred on the
 * player and writes it to <out_dir>/world_states.bin.
 *
 * world_states.bin binary layout (little-endian throughout):
 *   [8]  magic "WSBIN001"
 *   [4]  tick_count uint32
 *   [tick_count × 4]  player_block_x int32[]
 *   [tick_count × 4]  player_block_y int32[]
 *   [tick_count × 4]  player_block_z int32[]
 *   [tick_count × 65536]  block_data uint16[]  — shape [tick, 32, 32, 32]
 *                          world_coord = (px-15+i, py-15+j, pz-15+k)
 *
 * 1.19.4 play-state packet IDs (from minecraft-data protocol.json):
 *   0x24 map_chunk           (Chunk Data and Update Light)
 *   0x0a block_change        (Block Update — single block)
 *   0x43 multi_block_change  (Update Section Blocks — batch)
 *   0x41 respawn             (dimension change → clear world state)
 *   0x28 login               (Join Game — first chunk load, keep world state)
 *   0x1e unload_chunk        (chunk leaves render distance)
 */

'use strict';

const fs     = require('fs');
const path   = require('path');
const yauzl  = require('yauzl');
const nbt    = require('prismarine-nbt');
const { Vec3 } = require('vec3');

const MCVersion = '1.19.4';
// Overworld minY / worldHeight for 1.19.4
const WORLD_MIN_Y    = -64;
const WORLD_HEIGHT   = 384;
const SECTION_HEIGHT = 16;
const NUM_SECTIONS   = WORLD_HEIGHT / SECTION_HEIGHT; // 24

// ---------- VarInt / primitive readers ---------------------------------------

function readVarInt(buf, offset) {
  let result = 0, shift = 0, pos = offset;
  while (pos < buf.length) {
    const byte = buf[pos++];
    result |= (byte & 0x7f) << shift;
    shift += 7;
    if ((byte & 0x80) === 0) break;
  }
  return { value: result, size: pos - offset };
}

function readInt32BE(buf, offset) {
  return buf.readInt32BE(offset);
}

function readInt32LE(buf, offset) {
  return buf.readInt32LE(offset);
}

// Skip over network-NBT (big-endian, no gzip compression) and return byte count.
// Uses nbt.proto (big-endian ProtoDef) which reports metadata.size.
function skipNBT(buf, offset) {
  const slice = buf.slice(offset);
  const { metadata } = nbt.proto.parsePacketBuffer('nbt', slice);
  return metadata.size;
}

// Read a VarInt-length prefixed byte array; return { data: Buffer, size: total bytes consumed }
function readByteArray(buf, offset) {
  const { value: len, size: varintSize } = readVarInt(buf, offset);
  const data = buf.slice(offset + varintSize, offset + varintSize + len);
  return { data, size: varintSize + len };
}

// Read a VarLong (used in multi_block_change section position and block states)
function readVarLong(buf, offset) {
  let result = BigInt(0);
  let shift = BigInt(0);
  let pos = offset;
  while (pos < buf.length) {
    const byte = BigInt(buf[pos++]);
    result |= (byte & BigInt(0x7f)) << shift;
    shift += BigInt(7);
    if ((byte & BigInt(0x80)) === BigInt(0)) break;
  }
  return { value: result, size: pos - offset };
}

// ---------- Chunk handling with prismarine-chunk ----------------------------
// We use prismarine-chunk but call it directly with the section data buffer.

let PCChunk;
try {
  PCChunk = require('prismarine-chunk')(MCVersion);
} catch (e) {
  throw new Error(`prismarine-chunk failed to load for ${MCVersion}: ${e.message}`);
}

/**
 * Parse a map_chunk packet body (after the VarInt packet ID has been stripped).
 * Returns { chunkX, chunkZ, chunk } where chunk is a PCChunk instance.
 */
function parseMapChunkPacket(data) {
  let off = 0;

  const chunkX = data.readInt32BE(off); off += 4;
  const chunkZ = data.readInt32BE(off); off += 4;

  // HeightMaps NBT — skip it
  const nbtBytes = skipNBT(data, off);
  off += nbtBytes;

  // Data ByteArray (chunk sections)
  const { data: sectionData, size: arrBytes } = readByteArray(data, off);
  off += arrBytes;
  // (we don't need block entities or light data)

  // Create and load the chunk
  const chunk = new PCChunk({ minY: WORLD_MIN_Y, worldHeight: WORLD_HEIGHT });
  chunk.load(sectionData);

  return { chunkX, chunkZ, chunk };
}

/**
 * Parse a block_change packet (0x0a).
 * Returns { worldX, worldY, worldZ, stateId }
 *
 * Packet layout:
 *   position: 8-byte packed (x 26 bits, z 26 bits, y 12 bits)
 *   block_id: VarInt (global block state ID)
 */
function parseBlockChangePacket(data) {
  const raw = data.readBigInt64BE(0);
  // Upper 26 bits = X, next 26 = Z, lower 12 = Y (signed)
  let x = Number(raw >> BigInt(38));
  let z = Number((raw >> BigInt(12)) & BigInt(0x3ffffff));
  let y = Number(raw & BigInt(0xfff));
  // Sign extend
  if (x >= (1 << 25))  x -= (1 << 26);
  if (z >= (1 << 25))  z -= (1 << 26);
  if (y >= (1 << 11))  y -= (1 << 12);

  const { value: stateId } = readVarInt(data, 8);
  return { worldX: x, worldY: y, worldZ: z, stateId };
}

/**
 * Parse a multi_block_change packet (0x43).
 * Returns { chunkSectionX, chunkSectionY, chunkSectionZ, changes: [{localX,localY,localZ,stateId}] }
 *
 * Packet layout:
 *   chunk_section_pos: Long (packed: x 22 bits, z 22 bits, y 20 bits)
 *   no_trust_edges: Boolean (1 byte, ignored)
 *   changed_count: VarInt
 *   changes: VarLong[] — each: stateId in upper bits, (x<<8 | z<<4 | y) in lower 12 bits
 */
function parseMultiBlockChangePacket(data) {
  const packed = data.readBigInt64BE(0);
  // Bits 63..42 = sectionX (22 bits signed), 41..20 = sectionZ (22 bits signed), 19..0 = sectionY (20 bits signed)
  let sX = Number(packed >> BigInt(42));
  let sZ = Number((packed >> BigInt(20)) & BigInt(0x3fffff));
  let sY = Number(packed & BigInt(0xfffff));
  if (sX >= (1 << 21)) sX -= (1 << 22);
  if (sZ >= (1 << 21)) sZ -= (1 << 22);
  if (sY >= (1 << 19)) sY -= (1 << 20);

  let off = 8;
  off += 1; // no_trust_edges boolean

  const { value: changeCount, size: vcSize } = readVarInt(data, off);
  off += vcSize;

  const changes = [];
  for (let i = 0; i < changeCount; i++) {
    const { value: packed2, size: vlSize } = readVarLong(data, off);
    off += vlSize;
    const stateId = Number(packed2 >> BigInt(12));
    const local   = Number(packed2 & BigInt(0xfff));
    const localX  = (local >> 8) & 0xf;
    const localZ  = (local >> 4) & 0xf;
    const localY  = local & 0xf;
    changes.push({ localX, localY, localZ, stateId });
  }

  return { sectionX: sX, sectionY: sY, sectionZ: sZ, changes };
}

/**
 * Parse an unload_chunk packet (0x1e).
 * Returns { chunkX, chunkZ }
 */
function parseUnloadChunkPacket(data) {
  const chunkX = data.readInt32BE(0);
  const chunkZ = data.readInt32BE(4);
  return { chunkX, chunkZ };
}

// ---------- World state ------------------------------------------------------

/**
 * Minimal world: Map<"cx,cz", PCChunk>
 * All block lookups go through getBlock(worldX, worldY, worldZ).
 */
class World {
  constructor() {
    this.chunks = new Map();
  }

  _key(cx, cz) { return `${cx},${cz}`; }

  setChunk(cx, cz, chunk) {
    this.chunks.set(this._key(cx, cz), chunk);
  }

  removeChunk(cx, cz) {
    this.chunks.delete(this._key(cx, cz));
  }

  getChunk(cx, cz) {
    return this.chunks.get(this._key(cx, cz)) || null;
  }

  /** Returns global block state ID, or 0 (air) if not loaded. */
  getBlockState(worldX, worldY, worldZ) {
    if (worldY < WORLD_MIN_Y || worldY >= WORLD_MIN_Y + WORLD_HEIGHT) return 0;
    const cx = Math.floor(worldX / 16);
    const cz = Math.floor(worldZ / 16);
    const chunk = this.getChunk(cx, cz);
    if (!chunk) return 0;
    try {
      const lx = ((worldX % 16) + 16) % 16;
      const lz = ((worldZ % 16) + 16) % 16;
      return chunk.getBlockStateId(new Vec3(lx, worldY, lz)) || 0;
    } catch {
      return 0;
    }
  }

  setBlockState(worldX, worldY, worldZ, stateId) {
    if (worldY < WORLD_MIN_Y || worldY >= WORLD_MIN_Y + WORLD_HEIGHT) return;
    const cx = Math.floor(worldX / 16);
    const cz = Math.floor(worldZ / 16);
    let chunk = this.getChunk(cx, cz);
    if (!chunk) {
      chunk = new PCChunk({ minY: WORLD_MIN_Y, worldHeight: WORLD_HEIGHT });
      this.setChunk(cx, cz, chunk);
    }
    try {
      const lx = ((worldX % 16) + 16) % 16;
      const lz = ((worldZ % 16) + 16) % 16;
      chunk.setBlockStateId(new Vec3(lx, worldY, lz), stateId);
    } catch { /* ignore malformed state IDs */ }
  }

  /**
   * Snapshot a 32×32×32 block window centred on (px, py, pz) as a Uint16Array.
   * Cell (i,j,k) → world (px-15+i, py-15+j, pz-15+k).
   */
  snapshot32(px, py, pz) {
    const out = new Uint16Array(32 * 32 * 32);
    const ox  = px - 15;
    const oy  = py - 15;
    const oz  = pz - 15;
    let idx   = 0;
    for (let i = 0; i < 32; i++) {
      for (let j = 0; j < 32; j++) {
        for (let k = 0; k < 32; k++) {
          out[idx++] = this.getBlockState(ox + i, oy + j, oz + k);
        }
      }
    }
    return out;
  }

  clear() {
    this.chunks.clear();
  }
}

// ---------- tmcpr streaming --------------------------------------------------

/**
 * Read all tmcpr packets from a Buffer.
 * Yields { timestampMs, data } objects.
 * Each record: [int32BE ts_ms] [int32BE length] [length bytes data]
 */
function* iterPackets(buf) {
  let off = 0;
  while (off + 8 <= buf.length) {
    const ts  = buf.readInt32BE(off);     off += 4;
    const len = buf.readInt32BE(off);     off += 4;
    if (off + len > buf.length) break;
    const data = buf.slice(off, off + len);
    off += len;
    yield { timestampMs: ts, data };
  }
}

// ---------- tmcpr extraction from .mcpr (zip) --------------------------------

function extractTmcpr(mcprPath) {
  return new Promise((resolve, reject) => {
    yauzl.open(mcprPath, { lazyEntries: true }, (err, zipfile) => {
      if (err) return reject(err);
      zipfile.readEntry();
      zipfile.on('entry', entry => {
        if (entry.fileName === 'recording.tmcpr') {
          zipfile.openReadStream(entry, (err2, stream) => {
            if (err2) return reject(err2);
            const chunks = [];
            stream.on('data', c => chunks.push(c));
            stream.on('end', () => resolve(Buffer.concat(chunks)));
            stream.on('error', reject);
          });
        } else {
          zipfile.readEntry();
        }
      });
      zipfile.on('end', () => reject(new Error('recording.tmcpr not found in .mcpr')));
      zipfile.on('error', reject);
    });
  });
}

// ---------- login-state tracking --------------------------------------------
// The tmcpr starts in LOGIN state. After 0x02 (success), switches to PLAY.
// 0x04 in LOGIN = login_plugin_request (also present at start).

const LOGIN_SUCCESS_ID = 0x02;

// PLAY state IDs
const PLAY_MAP_CHUNK         = 0x24;
const PLAY_BLOCK_CHANGE      = 0x0a;
const PLAY_MULTI_BLOCK_CHANGE = 0x43;
const PLAY_RESPAWN           = 0x41;
const PLAY_UNLOAD_CHUNK      = 0x1e;
// 0x28 = login (Join Game) — already in PLAY, just means a fresh spawn

// ---------- main entry point -------------------------------------------------

async function main() {
  const [,, sessionDir, outDir] = process.argv;
  if (!sessionDir || !outDir) {
    console.error('Usage: node parse_mcpr.js <session_dir> <out_dir>');
    process.exit(1);
  }

  // Find the .mcpr file
  const files  = fs.readdirSync(sessionDir);
  const mcprName = files.find(f => f.endsWith('.mcpr'));
  if (!mcprName) throw new Error(`No .mcpr file found in ${sessionDir}`);
  const mcprPath = path.join(sessionDir, mcprName);
  console.log(`Reading ${mcprPath}`);

  // Load ticks
  const ticksPath = path.join(sessionDir, 'ticks.jsonl');
  const ticks = fs.readFileSync(ticksPath, 'utf8')
    .split('\n').filter(Boolean)
    .map(l => JSON.parse(l));
  console.log(`Loaded ${ticks.length} ticks`);

  // Extract tmcpr
  console.log('Extracting recording.tmcpr...');
  const tmcprBuf = await extractTmcpr(mcprPath);
  console.log(`tmcpr size: ${(tmcprBuf.length / 1024 / 1024).toFixed(1)} MB`);

  // Prepare output buffer pieces
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, 'world_states.bin');

  // We'll collect all tick snapshots into typed arrays
  const tickCount = ticks.length;
  const pxArr = new Int32Array(tickCount);
  const pyArr = new Int32Array(tickCount);
  const pzArr = new Int32Array(tickCount);
  // One Uint16Array per tick, but that's 137 MB total — allocate upfront
  const allBlocks = new Uint16Array(tickCount * 32 * 32 * 32);

  // Build sorted list of (replayMs, tickIndex) for efficient tick lookup
  const tickByMs = ticks.map((t, i) => ({ replayMs: t.replayMs, idx: i }));
  tickByMs.sort((a, b) => a.replayMs - b.replayMs);
  let nextTickPtr = 0; // pointer into tickByMs

  // World state
  const world = new World();
  let inPlayState = false;
  let packetCount = 0;
  let chunkCount = 0;
  let blockUpdateCount = 0;
  let lastPrintedTs = -1000;

  // For each tick we need the player's integer block coords from ticks.jsonl
  // We use floor() on the camera position per the spec's window definition.
  for (let i = 0; i < tickCount; i++) {
    const t = ticks[i];
    const cam = t.player.camera;
    pxArr[i] = Math.floor(cam.x);
    pyArr[i] = Math.floor(cam.y);
    pzArr[i] = Math.floor(cam.z);
  }

  // Walk packets chronologically
  for (const { timestampMs: ts, data } of iterPackets(tmcprBuf)) {
    packetCount++;

    // Print progress every 5 seconds of replay time
    if (ts - lastPrintedTs >= 5000) {
      const sec = (ts / 1000).toFixed(1);
      process.stderr.write(`  [parse_mcpr] t=${sec}s  packets=${packetCount}  chunks=${chunkCount}  blockUpdates=${blockUpdateCount}  ticks_snapped=${nextTickPtr}\n`);
      lastPrintedTs = ts;
    }

    // Take snapshots for all ticks whose replayMs <= current packet timestamp
    while (nextTickPtr < tickByMs.length && tickByMs[nextTickPtr].replayMs <= ts) {
      const { idx } = tickByMs[nextTickPtr++];
      const snap = world.snapshot32(pxArr[idx], pyArr[idx], pzArr[idx]);
      allBlocks.set(snap, idx * 32 * 32 * 32);
    }

    // Read VarInt packet ID
    const { value: packetId, size: idBytes } = readVarInt(data, 0);
    const payload = data.slice(idBytes);

    if (!inPlayState) {
      // LOGIN state
      if (packetId === LOGIN_SUCCESS_ID) {
        inPlayState = true;
        process.stderr.write('  [parse_mcpr] Switched to PLAY state\n');
      }
      continue;
    }

    // PLAY state
    try {
      switch (packetId) {
        case PLAY_MAP_CHUNK: {
          const { chunkX, chunkZ, chunk } = parseMapChunkPacket(payload);
          world.setChunk(chunkX, chunkZ, chunk);
          chunkCount++;
          break;
        }
        case PLAY_BLOCK_CHANGE: {
          const { worldX, worldY, worldZ, stateId } = parseBlockChangePacket(payload);
          world.setBlockState(worldX, worldY, worldZ, stateId);
          blockUpdateCount++;
          break;
        }
        case PLAY_MULTI_BLOCK_CHANGE: {
          const { sectionX, sectionY, sectionZ, changes } = parseMultiBlockChangePacket(payload);
          const baseX = sectionX * 16;
          const baseY = sectionY * 16;
          const baseZ = sectionZ * 16;
          for (const { localX, localY, localZ, stateId } of changes) {
            world.setBlockState(baseX + localX, baseY + localY, baseZ + localZ, stateId);
          }
          blockUpdateCount += changes.length;
          break;
        }
        case PLAY_RESPAWN: {
          // Dimension change — clear world state to avoid stale chunks
          world.clear();
          process.stderr.write('  [parse_mcpr] Respawn: cleared world state\n');
          break;
        }
        case PLAY_UNLOAD_CHUNK: {
          const { chunkX, chunkZ } = parseUnloadChunkPacket(payload);
          world.removeChunk(chunkX, chunkZ);
          break;
        }
        // 0x28 (login/JoinGame) just continues — world state accumulates normally
      }
    } catch (e) {
      // Non-fatal: a malformed packet shouldn't abort the whole session
      process.stderr.write(`  [parse_mcpr] WARN: packet 0x${packetId.toString(16)} at ts=${ts}: ${e.message}\n`);
    }
  }

  // Snap any remaining ticks (after the last packet)
  while (nextTickPtr < tickByMs.length) {
    const { idx } = tickByMs[nextTickPtr++];
    const snap = world.snapshot32(pxArr[idx], pyArr[idx], pzArr[idx]);
    allBlocks.set(snap, idx * 32 * 32 * 32);
  }

  process.stderr.write(`  [parse_mcpr] Done. packets=${packetCount}  chunks=${chunkCount}  blockUpdates=${blockUpdateCount}\n`);

  // Write world_states.bin
  console.log(`Writing ${outPath}...`);
  const magic = Buffer.from('WSBIN001');
  const header = Buffer.allocUnsafe(4);
  header.writeUInt32LE(tickCount, 0);

  const pxBuf = Buffer.from(pxArr.buffer);
  const pyBuf = Buffer.from(pyArr.buffer);
  const pzBuf = Buffer.from(pzArr.buffer);
  const dataBuf = Buffer.from(allBlocks.buffer);

  const fd = fs.openSync(outPath, 'w');
  fs.writeSync(fd, magic);
  fs.writeSync(fd, header);
  fs.writeSync(fd, pxBuf);
  fs.writeSync(fd, pyBuf);
  fs.writeSync(fd, pzBuf);
  fs.writeSync(fd, dataBuf);
  fs.closeSync(fd);

  const sizeMB = (fs.statSync(outPath).size / 1024 / 1024).toFixed(1);
  console.log(`Wrote ${outPath}  (${sizeMB} MB,  ${tickCount} ticks)`);

  // Quick sanity: print block state at player position for a mid-session tick
  const midTick = ticks[Math.floor(tickCount / 2)];
  const midIdx  = Math.floor(tickCount / 2);
  const midCam  = midTick.player.camera;
  const midPx   = pxArr[midIdx], midPy = pyArr[midIdx], midPz = pzArr[midIdx];
  // Cell (15,15,15) = player's camera cell
  const centerStateId = allBlocks[midIdx * 32 * 32 * 32 + 15 * 32 * 32 + 15 * 32 + 15];
  console.log(`\nMid-session sanity (tick ${midTick.tick}):`);
  console.log(`  camera pos: (${midCam.x.toFixed(2)}, ${midCam.y.toFixed(2)}, ${midCam.z.toFixed(2)})`);
  console.log(`  player block: (${midPx}, ${midPy}, ${midPz})`);
  console.log(`  block state at player cell (15,15,15): ${centerStateId}  (should be air=0 or similar)`);
}

main().catch(e => { console.error(e); process.exit(1); });
