#!/usr/bin/env node
/**
 * Blockscope production collection controller (Mineflayer + mineflayer-pathfinder, MC 1.19.4).
 *
 * One controller bot drives a scripted exploration episode; the BlockscopeMirror Paper
 * plugin teleport-mirrors a record-only camera onto it every tick, so the camera's frames
 * (and Furnace voxel/visibility labels) come from the controller's exact pose. Approach A:
 * the camera only records — it never paths — so labels stay valid.
 *
 * What this hardens over the prototype controller.js:
 *   - DISTANCE-SCALED per-goal timeouts: timeout = clamp(base + perBlock*dist, min, max).
 *     A 6-block hop and a 90-block cross-map route no longer share one fixed budget.
 *   - Raised pathfinder thinkTimeout / tickTimeout so deep multi-story interiors and long
 *     routes get enough planning time before a goal is abandoned.
 *   - Optional WAYPOINTING for deep climbs: a far/high goal is split into intermediate
 *     GoalNear hops (--waypoint-step N) so the planner solves short legs instead of one
 *     monster path that blows the think budget.
 *   - Active stuck detection: abandon-and-retry-next instead of silently wedging the run.
 *   - Auto pair + startcam via server RCON-less console (the launch script ops the bot and
 *     runs /mirror + /startcam); this controller just drives motion for `--duration` seconds.
 *
 * Episodes:
 *   explore        — random GoalNear walk (original)
 *   walklook       — small-radius walk + look-around (original)
 *   interior_hunt  — seek hermit-style interior signature blocks, slow look-around on arrival
 *   rare_block     — prioritised hunt for Tier1/2/3 rare/visual blocks; slow look on arrival
 *   vertical       — cycle rooftop → ground → underground views at varying XZ drift
 *   mine           — navigate to ore level, find + dig ore veins (survives ADVENTURE mode)
 *   hermitcraft    — composite: rotates interior_hunt/rare_block/vertical in 30s sub-cycles
 *
 * Usage:
 *   node controller.js --host 127.0.0.1 --port 25599 --user ctrl_bot \
 *        --episode explore --duration 300 \
 *        [--think-timeout 12000] [--goal-base 8000] [--goal-per-block 700] \
 *        [--goal-min 8000] [--goal-max 60000] [--waypoint-step 24] [--span 80] \
 *        [--targets x,y,z;x,y,z;...]   (explicit target list overrides random walk)
 */

const mineflayer = require("mineflayer");
const { pathfinder, Movements, goals } = require("mineflayer-pathfinder");
const { GoalNear, GoalBlock, GoalY } = goals;
const Vec3 = require("vec3").Vec3;

// ----------------------------------------------------------------------- args
function parseArgs(argv) {
  const a = {
    host: "127.0.0.1",
    port: 25599,
    user: "ctrl_bot",
    episode: "explore", // explore | walklook | interior_hunt | rare_block | vertical | mine | hermitcraft
    duration: 300,
    span: 80, // random-target radius around spawn (blocks)
    // pathfinding tuning
    thinkTimeout: 12000, // ms pathfinder may spend planning a single path (default lib: 5000)
    tickTimeout: 60, // ms of planning work per tick (default lib: 40)
    goalBase: 8000, // base per-goal timeout (ms)
    goalPerBlock: 700, // + ms per block of straight-line distance to the goal
    goalMin: 8000,
    goalMax: 60000,
    waypointStep: 24, // if a goal is farther than this, split into hops of this size (0 = off)
    targets: null, // "x,y,z;x,y,z;..." explicit ordered targets
    waitForCamera: 90, // seconds to wait after spawn for camera to load + join (0 = skip)
  };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i], v = argv[i + 1];
    const num = () => { i++; return parseFloat(v); };
    const int = () => { i++; return parseInt(v, 10); };
    switch (k) {
      case "--host": a.host = v; i++; break;
      case "--port": a.port = int(); break;
      case "--user": a.user = v; i++; break;
      case "--episode": a.episode = v; i++; break;
      case "--duration": a.duration = int(); break;
      case "--span": a.span = num(); break;
      case "--think-timeout": a.thinkTimeout = int(); break;
      case "--tick-timeout": a.tickTimeout = int(); break;
      case "--goal-base": a.goalBase = int(); break;
      case "--wait-for-camera": a.waitForCamera = num(); break;
      case "--goal-per-block": a.goalPerBlock = num(); break;
      case "--goal-min": a.goalMin = int(); break;
      case "--goal-max": a.goalMax = int(); break;
      case "--waypoint-step": a.waypointStep = num(); break;
      case "--targets": a.targets = v; i++; break;
      default: break;
    }
  }
  return a;
}

const args = parseArgs(process.argv);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (p) => `${p.x.toFixed(1)},${p.y.toFixed(1)},${p.z.toFixed(1)}`;

// --------------------------------------------------- look primitives (ported)
function sampleLognormal(mu, sigma) {
  let u1 = 0;
  while (u1 === 0) u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return Math.exp(mu + z * sigma);
}
function meanPreservingScale(volatility) {
  if (volatility <= 0) return 1;
  return sampleLognormal((-volatility * volatility) / 2, volatility);
}
async function lookSmooth(bot, targetYaw, targetPitch, degPerSec = 90, volatility = 0.35) {
  let speed = degPerSec;
  if (volatility > 0) {
    speed = degPerSec * meanPreservingScale(volatility);
    speed = Math.max(degPerSec * 0.4, Math.min(171, speed));
  }
  await bot.look(targetYaw, targetPitch, false, speed, speed, true);
}
async function lookAtSmooth(bot, target, degPerSec = 60) {
  const p = bot.entity.position;
  const dx = target.x - p.x, dy = target.y - (p.y + bot.entity.height), dz = target.z - p.z;
  const yaw = Math.atan2(-dx, -dz);
  const pitch = Math.atan2(dy, Math.sqrt(dx * dx + dz * dz));
  await lookSmooth(bot, yaw, pitch, degPerSec);
}

/**
 * Slow 360-degree look-around from current position: sweeps yaw 0→2π while
 * oscillating pitch between pitchLo and pitchHi. Useful for a "settling" scan
 * after arriving at an interesting location. totalMs controls total sweep time.
 */
async function lookAround360(bot, totalMs = 8000, pitchLo = -0.35, pitchHi = 0.70) {
  const steps = 24;
  const stepMs = totalMs / steps;
  for (let i = 0; i <= steps; i++) {
    const yaw = (i / steps) * Math.PI * 2;
    // Oscillate pitch: sine wave between pitchLo and pitchHi
    const pitchFrac = (Math.sin((i / steps) * Math.PI * 4) + 1) / 2; // 0..1
    const pitch = pitchLo + pitchFrac * (pitchHi - pitchLo);
    await bot.look(yaw, pitch, false);
    await sleep(stepMs);
  }
}

// ------------------------------------------------ pathfinder helpers (tuned)
function initPathfinder(bot, mcData) {
  const m = new Movements(bot, mcData);
  m.allowSprinting = true;
  m.allowParkour = true;
  m.canDig = false;       // never tear up the build map
  m.canPlaceOn = false;
  m.allowEntityDetection = true;
  m.maxDropDown = 4;      // conservative: don't suicide off ledges
  bot.pathfinder.setMovements(m);
  // TUNING: give the planner room. The library defaults (thinkTimeout 5000, tickTimeout 40)
  // are why long cross-map routes and deep multi-story climbs time out before a path is found.
  bot.pathfinder.thinkTimeout = args.thinkTimeout;
  bot.pathfinder.tickTimeout = args.tickTimeout;
  console.log(`[ctrl] pathfinder thinkTimeout=${args.thinkTimeout}ms tickTimeout=${args.tickTimeout}ms`);
}

// distance-scaled per-goal timeout
function goalTimeout(distBlocks) {
  const t = args.goalBase + args.goalPerBlock * distBlocks;
  return Math.round(Math.max(args.goalMin, Math.min(args.goalMax, t)));
}

// goto with a distance-scaled timeout + active stuck detection
async function gotoTracked(bot, goalVec, range, timeoutMs) {
  const goal = new GoalNear(goalVec.x, goalVec.y, goalVec.z, range);
  let to, stuckTimer;
  let lastPos = bot.entity.position.clone();
  let lastMoveT = Date.now();
  let stuckEvents = 0;
  const gotoP = bot.pathfinder.goto(goal);
  stuckTimer = setInterval(() => {
    const p = bot.entity.position;
    if (p.distanceTo(lastPos) > 0.5) { lastMoveT = Date.now(); lastPos = p.clone(); }
    else if (Date.now() - lastMoveT > 4000) { stuckEvents++; lastMoveT = Date.now(); }
  }, 1500);
  const toP = new Promise((_, reject) => {
    to = setTimeout(() => {
      try { bot.pathfinder.setGoal(null); } catch (_) {}
      reject(new Error("timeout"));
    }, timeoutMs);
  });
  let outcome;
  try { await Promise.race([gotoP, toP]); outcome = "reached"; }
  catch (e) { outcome = (e.message === "timeout") ? "timeout" : ("error:" + e.message); }
  finally { clearTimeout(to); clearInterval(stuckTimer); }
  const p = bot.entity.position;
  return { outcome, finalDist: p.distanceTo(goalVec), stuckEvents };
}

/**
 * Navigate to a goal, optionally splitting a long/high route into waypoint legs so the
 * planner solves short paths it can actually finish inside thinkTimeout. Each leg gets its
 * own distance-scaled timeout; a failed leg is retried directly to the final goal once.
 */
async function navigateTo(bot, goalVec, range) {
  const start = bot.entity.position.clone();
  const total = start.distanceTo(goalVec);

  if (args.waypointStep > 0 && total > args.waypointStep * 1.5) {
    const legs = Math.ceil(total / args.waypointStep);
    for (let leg = 1; leg <= legs; leg++) {
      const t = leg / legs;
      const wp = new Vec3(
        start.x + (goalVec.x - start.x) * t,
        start.y + (goalVec.y - start.y) * t,
        start.z + (goalVec.z - start.z) * t
      );
      const wpRange = leg === legs ? range : Math.max(range, 3);
      const d = bot.entity.position.distanceTo(wp);
      const r = await gotoTracked(bot, wp, wpRange, goalTimeout(d));
      if (r.outcome !== "reached" && leg < legs) {
        // A leg failed mid-route — try a direct shot to the final goal, then bail to next target.
        console.log(`[ctrl]   leg ${leg}/${legs} ${r.outcome}; direct attempt to final goal`);
        return await gotoTracked(bot, goalVec, range, goalTimeout(bot.entity.position.distanceTo(goalVec)));
      }
      if (leg === legs) return r;
    }
  }
  return await gotoTracked(bot, goalVec, range, goalTimeout(total));
}

// ---------------------------------------------------------------- episodes
function parseTargets(s) {
  return s.split(";").map((t) => t.split(",").map(Number)).filter((a) => a.length === 3)
          .map(([x, y, z]) => new Vec3(x, y, z));
}

// ---------------------------------------------------------------- exploreEpisode (original)
async function exploreEpisode(bot) {
  const origin = bot.entity.position.clone();
  const explicit = args.targets ? parseTargets(args.targets) : null;
  const endAt = Date.now() + args.duration * 1000;
  let n = 0, reached = 0, timeouts = 0;
  while (Date.now() < endAt) {
    let goalVec;
    if (explicit) {
      goalVec = explicit[n % explicit.length];
    } else {
      goalVec = new Vec3(
        Math.round(origin.x + (Math.random() * 2 - 1) * args.span),
        Math.round(origin.y),
        Math.round(origin.z + (Math.random() * 2 - 1) * args.span)
      );
    }
    n++;
    const d = bot.entity.position.distanceTo(goalVec);
    console.log(`[ctrl] target ${n} -> ${fmt(goalVec)} (dist ${d.toFixed(1)}, budget ${goalTimeout(d)}ms)`);
    await lookAtSmooth(bot, goalVec.offset(0, 1, 0), 60);
    const r = await navigateTo(bot, goalVec, 2);
    if (r.outcome === "reached") reached++; else if (r.outcome === "timeout") timeouts++;
    console.log(`[ctrl]   ${r.outcome} finalDist=${r.finalDist.toFixed(1)} stuck=${r.stuckEvents}`);
    // brief look-around in place so the camera captures a human-like glance
    await lookSmooth(bot, Math.random() * Math.PI * 2, Math.random() * 0.6 - 0.3, 50);
    await sleep(700);
  }
  console.log(`[ctrl] episode done: ${reached} reached, ${timeouts} timeouts, ${n} attempts`);
}

// ---------------------------------------------------------------- walkLookEpisode (original)
async function walkLookEpisode(bot) {
  const origin = bot.entity.position.clone();
  const span = 10;
  const endAt = Date.now() + args.duration * 1000;
  let n = 0;
  while (Date.now() < endAt) {
    n++;
    const tx = Math.round(origin.x + (Math.random() * 2 - 1) * span);
    const tz = Math.round(origin.z + (Math.random() * 2 - 1) * span);
    const ty = Math.round(origin.y);
    const goalVec = new Vec3(tx, ty, tz);
    console.log(`[ctrl] walk-look ${n}: -> ${fmt(goalVec)}`);
    await lookAtSmooth(bot, goalVec.offset(0, 1, 0), 60);
    const r = await navigateTo(bot, goalVec, 2);
    console.log(`[ctrl]   ${r.outcome} finalDist=${r.finalDist.toFixed(1)}`);
    const yaw = Math.random() * Math.PI * 2;
    const pitch = (Math.random() * 0.6 - 0.3);
    await lookSmooth(bot, yaw, pitch, 50);
    await sleep(800);
  }
}

// ================================================================ Episode A: interior_hunt
/**
 * Find hermit-style building interiors by seeking clusters of signature blocks
 * (chest, furnace, crafting_table, bookshelf, beds, enchanting_table, anvil).
 * Navigate to each cluster, do a slow 360° look-around, repeat until time runs out.
 * Falls back to walklook if no signature blocks are found.
 */
async function interiorHuntEpisode(bot, mcData) {
  // Build the list of signature block IDs that signal a human-built interior.
  const INTERIOR_NAMES = [
    "chest", "trapped_chest", "barrel",
    "furnace", "blast_furnace", "smoker",
    "crafting_table",
    "bookshelf",
    "enchanting_table",
    "anvil", "chipped_anvil", "damaged_anvil",
    // Beds: all 16 colors
    "white_bed", "orange_bed", "magenta_bed", "light_blue_bed",
    "yellow_bed", "lime_bed", "pink_bed", "gray_bed",
    "light_gray_bed", "cyan_bed", "purple_bed", "blue_bed",
    "brown_bed", "green_bed", "red_bed", "black_bed",
    // Workstations
    "loom", "cartography_table", "fletching_table", "smithing_table",
    "stonecutter", "grindstone", "brewing_stand",
  ];
  const interiorIds = INTERIOR_NAMES
    .map((n) => mcData.blocksByName[n]?.id)
    .filter((id) => id !== undefined);

  if (interiorIds.length === 0) {
    console.log("[ctrl:interior_hunt] no signature block IDs resolved, falling back to walklook");
    return walkLookEpisode(bot);
  }

  const endAt = Date.now() + args.duration * 1000;
  const visited = new Set(); // "x,y,z" keys of already-visited cluster anchors
  let clusterCount = 0;

  /**
   * Find the nearest unvisited cluster. A "cluster" is any signature block that
   * has at least one other signature block within CLUSTER_RADIUS of it.
   * Returns the Vec3 position of the best cluster anchor, or null.
   */
  const CLUSTER_RADIUS = 8;
  const SCAN_DISTANCE = 64;
  function findNextCluster() {
    const candidates = bot.findBlock({
      matching: interiorIds,
      maxDistance: SCAN_DISTANCE,
      count: 32,
    });
    if (!candidates || candidates.length === 0) return null;

    // Score each candidate: count how many other candidates are within CLUSTER_RADIUS.
    let bestBlock = null, bestScore = 0;
    for (const block of candidates) {
      const key = `${block.position.x},${block.position.y},${block.position.z}`;
      if (visited.has(key)) continue;
      let score = 0;
      for (const other of candidates) {
        if (other === block) continue;
        if (block.position.distanceTo(other.position) <= CLUSTER_RADIUS) score++;
      }
      // Prefer clusters (score >= 1), but fall back to any unvisited block.
      if (score > bestScore || (bestScore === 0 && bestBlock === null)) {
        bestScore = score;
        bestBlock = block;
      }
    }
    return bestBlock;
  }

  while (Date.now() < endAt) {
    const anchor = findNextCluster();
    if (!anchor) {
      console.log("[ctrl:interior_hunt] no signature blocks in loaded chunks; exploring to load more");
      // Drift randomly to load new chunks then retry.
      const pos = bot.entity.position;
      const drift = new Vec3(
        pos.x + (Math.random() * 2 - 1) * 40,
        pos.y,
        pos.z + (Math.random() * 2 - 1) * 40
      );
      await navigateTo(bot, drift, 3);
      await sleep(1500);
      continue;
    }

    const key = `${anchor.position.x},${anchor.position.y},${anchor.position.z}`;
    visited.add(key);
    clusterCount++;
    console.log(`[ctrl:interior_hunt] cluster ${clusterCount} -> ${fmt(anchor.position)} (${anchor.name})`);

    // Look toward the cluster as we approach.
    await lookAtSmooth(bot, anchor.position.offset(0, 1, 0), 60);
    const r = await navigateTo(bot, anchor.position, 3);
    console.log(`[ctrl:interior_hunt]   nav ${r.outcome} finalDist=${r.finalDist.toFixed(1)}`);

    if (Date.now() >= endAt) break;

    // Arrived (or close enough) — do a slow 360° sweep to capture the interior.
    console.log("[ctrl:interior_hunt] doing 360° look-around");
    await lookAround360(bot, 8000, -0.35, 0.65);

    await sleep(500);
  }

  console.log(`[ctrl:interior_hunt] done: visited ${clusterCount} clusters`);
}

// ================================================================ Episode B: rare_block_seeker
/**
 * Actively seek out visually-interesting / rare blocks and get close for a good camera view.
 * Tier 1 (rarest) → navigate immediately. Tier 2 → if no T1 found. Tier 3 → fallback.
 * On arrival: slow look-around. Then scan again. If nothing found: explore + scan.
 */

// Tier definitions: arrays of block name strings. We resolve IDs at runtime.
const RARE_TIER1_NAMES = [
  "beacon", "conduit", "end_portal_frame", "dragon_egg",
  // Shulker boxes — all 17 variants (undyed + 16 colors)
  "shulker_box",
  "white_shulker_box", "orange_shulker_box", "magenta_shulker_box", "light_blue_shulker_box",
  "yellow_shulker_box", "lime_shulker_box", "pink_shulker_box", "gray_shulker_box",
  "light_gray_shulker_box", "cyan_shulker_box", "purple_shulker_box", "blue_shulker_box",
  "brown_shulker_box", "green_shulker_box", "red_shulker_box", "black_shulker_box",
  "command_block", "chain_command_block", "repeating_command_block",
];
const RARE_TIER2_NAMES = [
  // Stained glass + panes (all 16 colors)
  "white_stained_glass", "orange_stained_glass", "magenta_stained_glass", "light_blue_stained_glass",
  "yellow_stained_glass", "lime_stained_glass", "pink_stained_glass", "gray_stained_glass",
  "light_gray_stained_glass", "cyan_stained_glass", "purple_stained_glass", "blue_stained_glass",
  "brown_stained_glass", "green_stained_glass", "red_stained_glass", "black_stained_glass",
  "white_stained_glass_pane", "orange_stained_glass_pane", "magenta_stained_glass_pane",
  "light_blue_stained_glass_pane", "yellow_stained_glass_pane", "lime_stained_glass_pane",
  "pink_stained_glass_pane", "gray_stained_glass_pane", "light_gray_stained_glass_pane",
  "cyan_stained_glass_pane", "purple_stained_glass_pane", "blue_stained_glass_pane",
  "brown_stained_glass_pane", "green_stained_glass_pane", "red_stained_glass_pane",
  "black_stained_glass_pane",
  // Banners (all 16 colors, floor + wall)
  "white_banner", "orange_banner", "magenta_banner", "light_blue_banner",
  "yellow_banner", "lime_banner", "pink_banner", "gray_banner",
  "light_gray_banner", "cyan_banner", "purple_banner", "blue_banner",
  "brown_banner", "green_banner", "red_banner", "black_banner",
  "white_wall_banner", "orange_wall_banner", "magenta_wall_banner", "light_blue_wall_banner",
  "yellow_wall_banner", "lime_wall_banner", "pink_wall_banner", "gray_wall_banner",
  "light_gray_wall_banner", "cyan_wall_banner", "purple_wall_banner", "blue_wall_banner",
  "brown_wall_banner", "green_wall_banner", "red_wall_banner", "black_wall_banner",
  // Decorative / interactive
  "item_frame", "glow_item_frame", "painting",
  "armor_stand",
  "bell", "lantern", "soul_lantern", "chain",
  "lodestone", "respawn_anchor",
];
const RARE_TIER3_NAMES = [
  "bookshelf", "enchanting_table", "anvil", "chipped_anvil", "damaged_anvil",
  "brewing_stand", "cauldron", "water_cauldron", "lava_cauldron", "powder_snow_cauldron",
  "jukebox", "note_block", "observer", "dispenser", "dropper", "hopper",
  // Terracotta — plain + all 16 stained
  "terracotta",
  "white_terracotta", "orange_terracotta", "magenta_terracotta", "light_blue_terracotta",
  "yellow_terracotta", "lime_terracotta", "pink_terracotta", "gray_terracotta",
  "light_gray_terracotta", "cyan_terracotta", "purple_terracotta", "blue_terracotta",
  "brown_terracotta", "green_terracotta", "red_terracotta", "black_terracotta",
  // Glazed terracotta — all 16 colors
  "white_glazed_terracotta", "orange_glazed_terracotta", "magenta_glazed_terracotta",
  "light_blue_glazed_terracotta", "yellow_glazed_terracotta", "lime_glazed_terracotta",
  "pink_glazed_terracotta", "gray_glazed_terracotta", "light_gray_glazed_terracotta",
  "cyan_glazed_terracotta", "purple_glazed_terracotta", "blue_glazed_terracotta",
  "brown_glazed_terracotta", "green_glazed_terracotta", "red_glazed_terracotta",
  "black_glazed_terracotta",
  // Concrete — all 16 colors
  "white_concrete", "orange_concrete", "magenta_concrete", "light_blue_concrete",
  "yellow_concrete", "lime_concrete", "pink_concrete", "gray_concrete",
  "light_gray_concrete", "cyan_concrete", "purple_concrete", "blue_concrete",
  "brown_concrete", "green_concrete", "red_concrete", "black_concrete",
];

async function rareBlockEpisode(bot, mcData) {
  // Resolve IDs once at episode start (null-safe).
  const tier1Ids = RARE_TIER1_NAMES.map((n) => mcData.blocksByName[n]?.id).filter((id) => id !== undefined);
  const tier2Ids = RARE_TIER2_NAMES.map((n) => mcData.blocksByName[n]?.id).filter((id) => id !== undefined);
  const tier3Ids = RARE_TIER3_NAMES.map((n) => mcData.blocksByName[n]?.id).filter((id) => id !== undefined);

  console.log(`[ctrl:rare_block] tier1=${tier1Ids.length} tier2=${tier2Ids.length} tier3=${tier3Ids.length} IDs resolved`);

  const endAt = Date.now() + args.duration * 1000;
  const visited = new Set(); // "x,y,z" keys of visited block positions
  const SCAN_DIST = 64;
  const RESCAN_INTERVAL_MS = 15000;
  let lastScanAt = 0;

  function pickBestUnvisited(ids) {
    if (ids.length === 0) return null;
    const blocks = bot.findBlock({ matching: ids, maxDistance: SCAN_DIST, count: 16 });
    if (!blocks || blocks.length === 0) return null;
    // Return closest unvisited block.
    let best = null, bestDist = Infinity;
    const pos = bot.entity.position;
    for (const b of blocks) {
      const key = `${b.position.x},${b.position.y},${b.position.z}`;
      if (visited.has(key)) continue;
      const d = pos.distanceTo(b.position);
      if (d < bestDist) { bestDist = d; best = b; }
    }
    return best;
  }

  while (Date.now() < endAt) {
    const now = Date.now();

    // Only rescan on the scheduled interval to avoid hammering findBlock.
    if (now - lastScanAt < RESCAN_INTERVAL_MS && lastScanAt !== 0) {
      await sleep(500);
      continue;
    }
    lastScanAt = now;

    // Find the highest-tier available block.
    let target = pickBestUnvisited(tier1Ids);
    let tierLabel = "T1";
    if (!target) { target = pickBestUnvisited(tier2Ids); tierLabel = "T2"; }
    if (!target) { target = pickBestUnvisited(tier3Ids); tierLabel = "T3"; }

    if (!target) {
      // Nothing found — explore to load new chunks, then rescan.
      console.log("[ctrl:rare_block] no rare blocks found; exploring 10s");
      const pos = bot.entity.position;
      const drift = new Vec3(
        pos.x + (Math.random() * 2 - 1) * 30,
        pos.y,
        pos.z + (Math.random() * 2 - 1) * 30
      );
      await navigateTo(bot, drift, 3);
      lastScanAt = 0; // force immediate rescan after explore
      continue;
    }

    const key = `${target.position.x},${target.position.y},${target.position.z}`;
    visited.add(key);
    console.log(`[ctrl:rare_block] ${tierLabel} target: ${target.name} @ ${fmt(target.position)}`);

    // Look toward target, navigate close.
    await lookAtSmooth(bot, target.position.offset(0, 0.5, 0), 60);
    const r = await navigateTo(bot, target.position, 3);
    console.log(`[ctrl:rare_block]   nav ${r.outcome} finalDist=${r.finalDist.toFixed(1)}`);

    if (Date.now() >= endAt) break;

    // Slow look-around on arrival so the camera really sees the block.
    console.log("[ctrl:rare_block] dwelling on arrival");
    // First look directly at the block, then sweep around.
    await lookAtSmooth(bot, target.position.offset(0, 0.5, 0), 30);
    await sleep(800);
    await lookAround360(bot, 5000, -0.25, 0.55);
    await sleep(300);
  }

  console.log("[ctrl:rare_block] episode done");
}

// ================================================================ Episode C: vertical_explorer
/**
 * Visit different Y-levels to get varied camera perspectives:
 * rooftop (surface + 10) → ground level → underground (surface - 5+).
 * Drifts XZ slowly so the bot doesn't just yo-yo in the same column.
 */
async function verticalExplorerEpisode(bot, mcData) {
  const endAt = Date.now() + args.duration * 1000;
  let cycle = 0;

  // Find the approximate surface Y at a given (x, z) by scanning down from sky limit.
  function surfaceYAt(x, z) {
    const yStart = Math.min(255, bot.entity.position.y + 30);
    for (let y = yStart; y >= 0; y--) {
      const block = bot.blockAt(new Vec3(Math.floor(x), y, Math.floor(z)));
      if (block && block.name !== "air" && block.name !== "cave_air" && block.name !== "void_air") {
        return y;
      }
    }
    return Math.floor(bot.entity.position.y); // fallback: current Y
  }

  // XZ drift: each cycle we shift a little so we don't stay in one column.
  let driftX = bot.entity.position.x;
  let driftZ = bot.entity.position.z;

  while (Date.now() < endAt) {
    cycle++;
    // Drift 15–35 blocks per cycle in a random direction.
    driftX += (Math.random() * 2 - 1) * 25;
    driftZ += (Math.random() * 2 - 1) * 25;

    const surfY = surfaceYAt(driftX, driftZ);
    console.log(`[ctrl:vertical] cycle ${cycle} driftXZ=(${driftX.toFixed(1)},${driftZ.toFixed(1)}) surfaceY=${surfY}`);

    // --- Phase 1: rooftop view (surface + 10, looking outward) ---
    const rooftopVec = new Vec3(driftX, surfY + 10, driftZ);
    if (Date.now() < endAt) {
      console.log(`[ctrl:vertical]   → rooftop ${fmt(rooftopVec)}`);
      await navigateTo(bot, rooftopVec, 3);
      await lookAround360(bot, 5000, -0.60, 0.20);
    }

    // --- Phase 2: ground level ---
    const groundVec = new Vec3(driftX, surfY + 1, driftZ);
    if (Date.now() < endAt) {
      console.log(`[ctrl:vertical]   → ground ${fmt(groundVec)}`);
      await navigateTo(bot, groundVec, 2);
      await lookAround360(bot, 3500, -0.20, 0.50);
    }

    // --- Phase 3: underground/basement (surface - 5, clamped to y >= 1) ---
    const underY = Math.max(1, surfY - 5);
    const underVec = new Vec3(driftX, underY, driftZ);
    if (Date.now() < endAt) {
      console.log(`[ctrl:vertical]   → underground ${fmt(underVec)}`);
      await navigateTo(bot, underVec, 3);
      await lookAround360(bot, 4000, -0.30, 0.70);
    }

    await sleep(400);
  }

  console.log(`[ctrl:vertical] done: ${cycle} cycles`);
}

// ================================================================ Episode D: mine_and_collect
/**
 * Navigate to ore level (Y~30), scan for ores, mine them, explore the vein.
 * Survives ADVENTURE mode gracefully: dig() failures are caught and skipped.
 * Every 60s surfaces for a look-around before descending again.
 */

const ORE_NAMES_1194 = [
  // 1.19.4 ore names (deepslate variants included)
  "coal_ore", "deepslate_coal_ore",
  "iron_ore", "deepslate_iron_ore",
  "gold_ore", "deepslate_gold_ore",
  "diamond_ore", "deepslate_diamond_ore",
  "lapis_ore", "deepslate_lapis_ore",
  "redstone_ore", "deepslate_redstone_ore",
  "emerald_ore", "deepslate_emerald_ore",
  "copper_ore", "deepslate_copper_ore",
  "nether_quartz_ore", "nether_gold_ore",
];
// 1.8 / legacy names (mc_converter worlds)
const ORE_NAMES_LEGACY = [
  "coal_ore", "iron_ore", "gold_ore", "diamond_ore",
  "lapis_ore", "redstone_ore", "emerald_ore",
  // 1.8 had no deepslate
];

async function mineEpisode(bot, mcData) {
  // Build ore ID list — try 1.19.4 names first, fall back to legacy names.
  const allOreNames = [...new Set([...ORE_NAMES_1194, ...ORE_NAMES_LEGACY])];
  const oreIds = allOreNames.map((n) => mcData.blocksByName[n]?.id).filter((id) => id !== undefined);
  console.log(`[ctrl:mine] ${oreIds.length} ore type IDs resolved`);

  const endAt = Date.now() + args.duration * 1000;
  const ORE_Y = 30;          // target Y for ore level
  const SURFACE_INTERVAL = 60000; // ms between surface excursions
  const ORE_SCAN_DIST = 32;
  const VEIN_ADJACENT_RANGE = 2; // blocks to scan around a mined ore for vein continuation
  const MAX_VEIN_BLOCKS = 8;     // max blocks to mine per vein before moving on

  let lastSurfaceAt = Date.now() - SURFACE_INTERVAL; // start underground immediately

  function surfaceY() {
    const x = bot.entity.position.x, z = bot.entity.position.z;
    for (let y = 255; y >= 0; y--) {
      const b = bot.blockAt(new Vec3(Math.floor(x), y, Math.floor(z)));
      if (b && b.name !== "air" && b.name !== "cave_air" && b.name !== "void_air") return y;
    }
    return 64;
  }

  /**
   * Safely dig a single block. In ADVENTURE mode bot.dig() throws; we catch and skip.
   * Returns true if the block was dug, false if skipped (mode or other error).
   */
  async function safeDig(block) {
    try {
      await bot.dig(block);
      return true;
    } catch (e) {
      const msg = (e.message || "").toLowerCase();
      if (msg.includes("adventure") || msg.includes("cannot dig") || msg.includes("not allowed")) {
        // ADVENTURE mode — skip digging gracefully and just look at the block.
        console.log(`[ctrl:mine] dig skipped (${e.message}); looking at block`);
        await lookAtSmooth(bot, block.position.offset(0.5, 0.5, 0.5), 30);
        await sleep(600);
        return false;
      }
      console.log(`[ctrl:mine] dig error (non-fatal): ${e.message}`);
      return false;
    }
  }

  /**
   * Mine a vein starting from seedBlock. Scans adjacent blocks for matching ores,
   * navigates + mines up to MAX_VEIN_BLOCKS total.
   */
  async function mineVein(seedBlock) {
    const toMine = [seedBlock];
    const minedKeys = new Set();
    let count = 0;
    while (toMine.length > 0 && count < MAX_VEIN_BLOCKS && Date.now() < endAt) {
      const block = toMine.shift();
      const key = `${block.position.x},${block.position.y},${block.position.z}`;
      if (minedKeys.has(key)) continue;
      minedKeys.add(key);

      // Navigate to within 3 blocks of the ore.
      const r = await gotoTracked(bot, block.position, 3, goalTimeout(bot.entity.position.distanceTo(block.position)));
      if (r.outcome !== "reached" && r.finalDist > 5) {
        console.log(`[ctrl:mine] couldn't reach ore @ ${fmt(block.position)}, skipping`);
        continue;
      }

      // Look at it first (good for camera footage even if we can't dig).
      await lookAtSmooth(bot, block.position.offset(0.5, 0.5, 0.5), 40);
      await sleep(300);

      // Attempt to dig.
      const dug = await safeDig(block);
      if (dug) {
        count++;
        console.log(`[ctrl:mine] mined ${block.name} @ ${fmt(block.position)} (${count}/${MAX_VEIN_BLOCKS})`);
        // Scan adjacent area for more ore of the same type.
        if (count < MAX_VEIN_BLOCKS) {
          const adjacent = bot.findBlock({
            matching: [block.type],
            maxDistance: VEIN_ADJACENT_RANGE,
            count: 4,
            point: block.position,
          });
          if (adjacent) {
            for (const adj of adjacent) {
              const adjKey = `${adj.position.x},${adj.position.y},${adj.position.z}`;
              if (!minedKeys.has(adjKey)) toMine.push(adj);
            }
          }
        }
      }
    }
    return count;
  }

  while (Date.now() < endAt) {
    // Surface excursion every SURFACE_INTERVAL.
    if (Date.now() - lastSurfaceAt >= SURFACE_INTERVAL) {
      const sy = surfaceY();
      const surfVec = new Vec3(bot.entity.position.x, sy + 1, bot.entity.position.z);
      console.log("[ctrl:mine] surfacing for look-around");
      await navigateTo(bot, surfVec, 3);
      await lookAround360(bot, 4000, -0.40, 0.30);
      lastSurfaceAt = Date.now();
    }

    // Navigate toward ore level (Y=30).
    const pos = bot.entity.position;
    if (Math.abs(pos.y - ORE_Y) > 8) {
      console.log(`[ctrl:mine] descending to Y=${ORE_Y} (currently Y=${pos.y.toFixed(1)})`);
      // Use GoalY wrapped in a timeout via gotoTracked with a synthetic vec.
      const targetVec = new Vec3(pos.x, ORE_Y, pos.z);
      // GoalY can be overkill; use GoalNear at ore level with a generous Y tolerance.
      await gotoTracked(bot, targetVec, 4, 30000);
    }

    // Scan for ores.
    if (oreIds.length === 0) {
      // No ore IDs — just explore underground.
      const p = bot.entity.position;
      const randVec = new Vec3(p.x + (Math.random() * 2 - 1) * 20, ORE_Y, p.z + (Math.random() * 2 - 1) * 20);
      await navigateTo(bot, randVec, 3);
      await sleep(2000);
      continue;
    }

    const ores = bot.findBlock({ matching: oreIds, maxDistance: ORE_SCAN_DIST, count: 8 });
    if (!ores || ores.length === 0) {
      // No ores visible — explore underground for 20s to find new chunks.
      console.log("[ctrl:mine] no ores in range; exploring underground");
      const p = bot.entity.position;
      const randVec = new Vec3(
        p.x + (Math.random() * 2 - 1) * 25,
        ORE_Y + (Math.random() * 6 - 3),
        p.z + (Math.random() * 2 - 1) * 25
      );
      await navigateTo(bot, randVec, 3);
      await sleep(2000);
      continue;
    }

    // Pick closest ore.
    let closest = null, closestDist = Infinity;
    const cpos = bot.entity.position;
    for (const b of ores) {
      const d = cpos.distanceTo(b.position);
      if (d < closestDist) { closestDist = d; closest = b; }
    }

    console.log(`[ctrl:mine] ore target: ${closest.name} @ ${fmt(closest.position)} (${closestDist.toFixed(1)} blocks)`);
    const veinCount = await mineVein(closest);
    console.log(`[ctrl:mine] vein done: ${veinCount} blocks mined`);
    await sleep(500);
  }

  console.log("[ctrl:mine] episode done");
}

// ================================================================ hermitcraft composite
/**
 * Composite episode for hermitcraft worlds: rotates through interior_hunt (40%),
 * rare_block (35%), vertical (25%) in 30-second sub-cycles.
 * Picks sub-episode at the start of each sub-cycle so the distribution is maintained
 * without needing to restart the controller.
 */
async function hermitcraftEpisode(bot, mcData) {
  const endAt = Date.now() + args.duration * 1000;
  const SUB_CYCLE_MS = 30000;
  let subCycle = 0;

  // We re-use the episode functions but pass a short duration each time.
  // The episodes respect Date.now() < endAt so we set a temporary args.duration override.
  // Instead of overriding globals, we pass the endAt into each helper via a shared closure.
  // Simpler: spin up a sub-episode with a capped duration by creating a sub-endAt.

  // We need local variants that accept an explicit endAt rather than reading args.duration.
  // Easiest: temporarily monkey-patch args.duration, restore after. (No concurrency issues —
  // this is single-threaded Node.js.)
  const origDuration = args.duration;

  while (Date.now() < endAt) {
    subCycle++;
    const remaining = endAt - Date.now();
    if (remaining <= 0) break;
    const subDur = Math.min(SUB_CYCLE_MS, remaining);

    // Pick sub-episode by weighted random.
    const r = Math.random();
    let subName;
    if (r < 0.40) subName = "interior_hunt";
    else if (r < 0.75) subName = "rare_block";
    else subName = "vertical";

    console.log(`[ctrl:hermitcraft] sub-cycle ${subCycle} → ${subName} for ${(subDur/1000).toFixed(0)}s`);

    // Run sub-episode with a temporary duration.
    args.duration = subDur / 1000;
    try {
      if (subName === "interior_hunt") await interiorHuntEpisode(bot, mcData);
      else if (subName === "rare_block") await rareBlockEpisode(bot, mcData);
      else await verticalExplorerEpisode(bot, mcData);
    } catch (e) {
      console.log(`[ctrl:hermitcraft] sub-episode ${subName} error: ${e.message}`);
    }
  }

  // Restore original duration (in case anything reads it after).
  args.duration = origDuration;
  console.log(`[ctrl:hermitcraft] done: ${subCycle} sub-cycles`);
}

// ---------------------------------------------------------------------- main
function main() {
  console.log(`[ctrl] connecting ${args.user}@${args.host}:${args.port} episode=${args.episode} dur=${args.duration}s`);
  const bot = mineflayer.createBot({
    host: args.host, port: args.port, username: args.user, version: "1.19.4", auth: "offline",
  });
  bot.loadPlugin(pathfinder);
  bot.on("kicked", (r) => console.log("[ctrl] kicked:", r));
  bot.on("error", (e) => console.log("[ctrl] error:", e.message));
  bot.on("end", (r) => { console.log("[ctrl] disconnected:", r); process.exit(0); });

  bot.once("spawn", async () => {
    console.log(`[ctrl] spawned at ${fmt(bot.entity.position)}`);
    const mcData = require("minecraft-data")(bot.version);
    initPathfinder(bot, mcData);
    // Wait for the camera client to finish loading Fabric + assets + auto-connect.
    // A cold Docker container takes ~90s on first boot (warm = ~30s with cached assets).
    // Overridable via --wait-for-camera <seconds>. The camera logs 'Recording started'
    // once it's fully in; we can't read that from here, so use a fixed wall-clock wait.
    const camWait = args.waitForCamera != null ? args.waitForCamera * 1000 : 90000;
    if (camWait > 0) {
      console.log(`[ctrl] waiting ${camWait/1000}s for camera to load + join (set --wait-for-camera 0 to skip) ...`);
      await sleep(camWait);
    }
    try {
      switch (args.episode) {
        case "interior_hunt": await interiorHuntEpisode(bot, mcData); break;
        case "rare_block":    await rareBlockEpisode(bot, mcData); break;
        case "vertical":      await verticalExplorerEpisode(bot, mcData); break;
        case "mine":          await mineEpisode(bot, mcData); break;
        case "hermitcraft":   await hermitcraftEpisode(bot, mcData); break;
        case "walklook":      await walkLookEpisode(bot); break;
        case "explore":
        default:              await exploreEpisode(bot); break;
      }
    }
    catch (e) { console.log("[ctrl] episode error:", e.message); }
    console.log("[ctrl] episode complete — staying 5s then quitting");
    await sleep(5000);
    bot.quit("episode complete");
  });
}

main();
