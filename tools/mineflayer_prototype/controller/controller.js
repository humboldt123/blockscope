#!/usr/bin/env node
/**
 * Single-agent Blockscope controller harness (Mineflayer + pathfinder, MC 1.19.4).
 *
 * This is a self-contained port of the *essence* of solaris-engine's single-agent
 * episodes (walk-look / orbit) and movement primitives. Solaris' real episodes are
 * tightly coupled to a two-bot "coordinator" that orbits a shared midpoint and
 * exchanges phase events; that coupling is irrelevant to proving the camera rig, so
 * here we drive ONE bot through a scripted walk-look (or orbit-a-point) loop and let
 * the mirror plugin mirror its motion onto the recording camera.
 *
 * Ported primitives (from controller/primitives/movement.js):
 *   - initializePathfinder()  (Movements config)
 *   - lookAtSmooth() / lookSmooth() with mean-preserving log-normal speed jitter
 *   - gotoWithTimeout()       (pathfinder.goto with a hard timeout so a stuck path
 *                              can't wedge the run)
 *
 * Usage:
 *   node controller.js --host 127.0.0.1 --port 25599 --user ctrl_bot \
 *        --episode walklook --duration 180 [--center x,y,z] [--radius 8]
 */

const mineflayer = require("mineflayer");
const { pathfinder, Movements, goals } = require("mineflayer-pathfinder");
const { GoalNear, GoalBlock } = goals;
const Vec3 = require("vec3").Vec3;

// ----------------------------------------------------------------------- args
function parseArgs(argv) {
  const a = {
    host: "127.0.0.1",
    port: 25599,
    user: "ctrl_bot",
    episode: "walklook", // walklook | orbit
    duration: 180, // seconds of episode activity
    radius: 8,
    center: null, // "x,y,z"
  };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i];
    const v = argv[i + 1];
    if (k === "--host") { a.host = v; i++; }
    else if (k === "--port") { a.port = parseInt(v, 10); i++; }
    else if (k === "--user") { a.user = v; i++; }
    else if (k === "--episode") { a.episode = v; i++; }
    else if (k === "--duration") { a.duration = parseInt(v, 10); i++; }
    else if (k === "--radius") { a.radius = parseFloat(v); i++; }
    else if (k === "--center") { a.center = v; i++; }
  }
  return a;
}

const args = parseArgs(process.argv);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --------------------------------------------------- look primitives (ported)
function sampleLognormal(mu, sigma) {
  let u1 = 0, u2 = 0;
  while (u1 === 0) u1 = Math.random();
  u2 = Math.random();
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

// ------------------------------------------------ pathfinder helpers (ported)
function initPathfinder(bot, mcData) {
  const m = new Movements(bot, mcData);
  m.allowSprinting = true;
  m.allowParkour = true;
  m.canDig = false;        // record-on-a-flat-world prototype: don't tear up the world
  m.canPlaceOn = false;
  m.allowEntityDetection = true;
  m.maxDropDown = 8;
  bot.pathfinder.setMovements(m);
}

async function gotoWithTimeout(bot, goal, timeoutMs) {
  let to;
  const gotoP = bot.pathfinder.goto(goal);
  const toP = new Promise((_, reject) => {
    to = setTimeout(() => {
      try { bot.pathfinder.setGoal(null); } catch (_) {}
      reject(new Error(`goto timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });
  try { await Promise.race([gotoP, toP]); }
  finally { if (to) clearTimeout(to); }
}

// ---------------------------------------------------------------- episodes
async function walkLookEpisode(bot, mcData) {
  // Walk to a sequence of nearby random ground targets; smooth-look between them.
  const origin = bot.entity.position.clone();
  const span = 10;
  const endAt = Date.now() + args.duration * 1000;
  let n = 0;
  while (Date.now() < endAt) {
    n++;
    const tx = Math.round(origin.x + (Math.random() * 2 - 1) * span);
    const tz = Math.round(origin.z + (Math.random() * 2 - 1) * span);
    const ty = Math.round(origin.y);
    console.log(`[ctrl] walk-look ${n}: -> (${tx}, ${ty}, ${tz})`);

    // Look toward the target first (smooth pan -> camera mirrors a real human glance).
    await lookAtSmooth(bot, new Vec3(tx, ty + 1, tz), 60);

    try {
      await gotoWithTimeout(bot, new GoalNear(tx, ty, tz, 1.5), 12000);
      console.log(`[ctrl] reached target ${n}`);
    } catch (e) {
      console.log(`[ctrl] target ${n} ${e.message} (pos ${fmt(bot.entity.position)})`);
    }

    // Pan around in place a little (look phase).
    const yaw = Math.random() * Math.PI * 2;
    const pitch = (Math.random() * 0.6 - 0.3);
    await lookSmooth(bot, yaw, pitch, 50);
    await sleep(800);
  }
}

async function orbitEpisode(bot, mcData) {
  // Single-bot orbit: circle a fixed center point, looking inward at the center.
  const center = args.center
    ? vecFromStr(args.center)
    : bot.entity.position.clone().offset(args.radius, 0, 0);
  const checkpoints = 8;
  const endAt = Date.now() + args.duration * 1000;
  console.log(`[ctrl] orbit center=${fmt(center)} r=${args.radius}`);
  let i = 0;
  while (Date.now() < endAt) {
    const angle = (i % checkpoints) * (2 * Math.PI / checkpoints);
    const tx = Math.round(center.x + args.radius * Math.cos(angle));
    const tz = Math.round(center.z + args.radius * Math.sin(angle));
    const ty = Math.round(center.y);
    console.log(`[ctrl] orbit checkpoint ${i}: -> (${tx}, ${ty}, ${tz})`);
    try {
      await gotoWithTimeout(bot, new GoalNear(tx, ty, tz, 1.5), 8000);
    } catch (e) {
      console.log(`[ctrl] orbit cp ${i} ${e.message}`);
    }
    // Make eye contact with the center (inward look) at each checkpoint.
    await lookAtSmooth(bot, center.offset(0, 1, 0), 90);
    await sleep(1000);
    i++;
  }
}

// ------------------------------------------------------------------ helpers
const fmt = (p) => `${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)}`;
function vecFromStr(s) {
  const [x, y, z] = s.split(",").map(Number);
  return new Vec3(x, y, z);
}

// ---------------------------------------------------------------------- main
function main() {
  console.log(`[ctrl] connecting ${args.user}@${args.host}:${args.port} episode=${args.episode}`);
  const bot = mineflayer.createBot({
    host: args.host,
    port: args.port,
    username: args.user,
    version: "1.19.4",
    auth: "offline",
  });

  bot.loadPlugin(pathfinder);

  bot.on("kicked", (r) => console.log("[ctrl] kicked:", r));
  bot.on("error", (e) => console.log("[ctrl] error:", e.message));
  bot.on("end", (r) => { console.log("[ctrl] disconnected:", r); process.exit(0); });

  bot.once("spawn", async () => {
    console.log(`[ctrl] spawned at ${fmt(bot.entity.position)}`);
    const mcData = require("minecraft-data")(bot.version);
    initPathfinder(bot, mcData);

    // Settle a moment so the mirror plugin can pair + the camera can teleport in.
    await sleep(3000);

    try {
      if (args.episode === "orbit") await orbitEpisode(bot, mcData);
      else await walkLookEpisode(bot, mcData);
    } catch (e) {
      console.log("[ctrl] episode error:", e.message);
    }

    console.log("[ctrl] episode complete — staying connected 5s then quitting");
    await sleep(5000);
    bot.quit("episode complete");
  });
}

main();
