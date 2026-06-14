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
 * Usage:
 *   node controller.js --host 127.0.0.1 --port 25599 --user ctrl_bot \
 *        --episode explore --duration 300 \
 *        [--think-timeout 12000] [--goal-base 8000] [--goal-per-block 700] \
 *        [--goal-min 8000] [--goal-max 60000] [--waypoint-step 24] [--span 80] \
 *        [--targets x,y,z;x,y,z;...]   (explicit target list overrides random walk)
 */

const mineflayer = require("mineflayer");
const { pathfinder, Movements, goals } = require("mineflayer-pathfinder");
const { GoalNear } = goals;
const Vec3 = require("vec3").Vec3;

// ----------------------------------------------------------------------- args
function parseArgs(argv) {
  const a = {
    host: "127.0.0.1",
    port: 25599,
    user: "ctrl_bot",
    episode: "explore", // explore | walklook
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
    // settle so the mirror plugin can pair + the camera teleports in + recording starts
    await sleep(4000);
    try { await exploreEpisode(bot); }
    catch (e) { console.log("[ctrl] episode error:", e.message); }
    console.log("[ctrl] episode complete — staying 5s then quitting");
    await sleep(5000);
    bot.quit("episode complete");
  });
}

main();
