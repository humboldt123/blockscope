#!/usr/bin/env node
/**
 * Blockscope navigation stress test — DENSE world edition.
 *
 * Drives ONE Mineflayer + mineflayer-pathfinder bot through a battery of GoalNear
 * targets across a dense, non-flat build map (converted "Fallen Kingdom"), and
 * measures pathing reliability to decide whether mineflayer-pathfinder can replace
 * Baritone as the data-collection driver.
 *
 * Per target we record:
 *   - reached / timed_out
 *   - stuck/no-progress events (position barely changes for a sustained window
 *     while a goal is active) — detects silent wedging
 *   - falls/walk-offs into the void (Y drops below VOID_Y) and deaths
 *   - time-to-reach
 *   - final distance-to-goal
 *
 * Targets are derived from the survey (surface heights Y55-114). We aim at the
 * exact surveyed (x, surfaceY, z) points so each target is on a real reachable
 * block; pathfinder's GoalNear tolerance handles small offsets.
 *
 * Usage: node navtest.js --host 127.0.0.1 --port 25599 --user ctrl_bot
 */
const mineflayer = require("mineflayer");
const { pathfinder, Movements, goals } = require("mineflayer-pathfinder");
const { GoalNear } = goals;
const Vec3 = require("vec3").Vec3;

function parseArgs(argv) {
  const a = { host: "127.0.0.1", port: 25599, user: "ctrl_bot", perGoalMs: 30000 };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i], v = argv[i + 1];
    if (k === "--host") { a.host = v; i++; }
    else if (k === "--port") { a.port = parseInt(v, 10); i++; }
    else if (k === "--user") { a.user = v; i++; }
    else if (k === "--timeout") { a.perGoalMs = parseInt(v, 10) * 1000; i++; }
  }
  return a;
}
const args = parseArgs(process.argv);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (p) => `${p.x.toFixed(1)},${p.y.toFixed(1)},${p.z.toFixed(1)}`;

// Targets: (x, y, z) from the survey. Mix of low ground, mid terrain, castle tops,
// forest, sand/ice/water-edge — varied elevation and reachability. Castle-top
// targets (Y 91/108/114) are the hard cases: interior stairs, walls, multi-story.
const TARGETS = [
  [190, 62, 170], [230, 62, 190], [250, 63, 230], [270, 63, 250],  // ground, grass
  [210, 64, 270], [190, 64, 290], [230, 65, 290], [210, 63, 290],  // near-spawn courtyard
  [190, 91, 310], [190, 108, 330], [210, 114, 330],                // CASTLE multi-story / stairs (hard)
  [210, 74, 310], [230, 66, 310], [250, 68, 250],                  // mid elevation walls/courtyard
  [170, 85, 330], [170, 62, 350],                                  // west wing + sandy lowland
  [290, 64, 190], [310, 63, 170], [310, 69, 250],                  // east edge
  [250, 60, 330], [270, 61, 330], [290, 59, 330],                  // sand/clay shore (water-edge, fall risk)
  [210, 72, 350], [230, 70, 350], [250, 68, 350],                  // south terrain rise
  [270, 80, 370], [230, 80, 370], [310, 77, 370],                  // dense forest (jungle/oak canopy)
  [190, 62, 270], [210, 62, 170], [250, 62, 170],                  // back to low ground / ice
  [205, 64, 286],                                                  // return to spawn
];

let walkOffEvents = 0, deaths = 0;
const VOID_Y = 40; // structures sit ~Y55-114; below 40 = fell off into void/water hole

function initPathfinder(bot, mcData) {
  const m = new Movements(bot, mcData);
  m.allowSprinting = true;
  m.allowParkour = true;
  m.canDig = false;      // pure-nav: never tear up the build
  m.canPlaceOn = false;
  m.allowEntityDetection = true;
  m.maxDropDown = 4;     // conservative: don't suicide off ledges
  bot.pathfinder.setMovements(m);
}

// goto with per-goal timeout + active stuck detection
async function gotoTracked(bot, goal, timeoutMs) {
  let to, stuckTimer;
  let lastPos = bot.entity.position.clone();
  let lastMoveT = Date.now();
  let stuckEvents = 0;
  let minDist = Infinity;
  const goalVec = new Vec3(goal.x, goal.y, goal.z);

  const gotoP = bot.pathfinder.goto(goal);
  // stuck monitor: every 1.5s, if moved < 0.5 block and not within goal range, count stuck
  stuckTimer = setInterval(() => {
    const p = bot.entity.position;
    const moved = p.distanceTo(lastPos);
    const d = p.distanceTo(goalVec);
    if (d < minDist) minDist = d;
    if (moved > 0.5) { lastMoveT = Date.now(); lastPos = p.clone(); }
    else if (Date.now() - lastMoveT > 3500 && d > goal.rangeSq ** 0.5 + 1) {
      stuckEvents++;
      lastMoveT = Date.now(); // count once per 3.5s window of no progress
    }
  }, 1500);

  const toP = new Promise((_, reject) => {
    to = setTimeout(() => {
      try { bot.pathfinder.setGoal(null); } catch (_) {}
      reject(new Error("timeout"));
    }, timeoutMs);
  });

  const t0 = Date.now();
  let outcome;
  try { await Promise.race([gotoP, toP]); outcome = "reached"; }
  catch (e) { outcome = e.message === "timeout" ? "timeout" : ("error:" + e.message); }
  finally { clearTimeout(to); clearInterval(stuckTimer); }

  const p = bot.entity.position;
  return { outcome, ms: Date.now() - t0, finalDist: p.distanceTo(goalVec), stuckEvents, minDist };
}

function main() {
  const bot = mineflayer.createBot({
    host: args.host, port: args.port, username: args.user, version: "1.19.4", auth: "offline",
  });
  bot.loadPlugin(pathfinder);
  bot.on("kicked", (r) => console.log("[nav] kicked:", r));
  bot.on("error", (e) => console.log("[nav] error:", e.message));
  bot.on("death", () => { deaths++; console.log("[nav] *** DEATH (respawn) ***"); });
  bot.on("end", () => process.exit(0));

  bot.once("spawn", async () => {
    console.log(`[nav] spawned at ${fmt(bot.entity.position)}`);
    const mcData = require("minecraft-data")(bot.version);
    initPathfinder(bot, mcData);
    await sleep(3000);

    // global void watcher
    const voidWatch = setInterval(() => {
      if (bot.entity && bot.entity.position.y < VOID_Y) {
        walkOffEvents++;
        console.log(`[nav] !!! WALK-OFF/FALL: Y=${bot.entity.position.y.toFixed(1)} at ${fmt(bot.entity.position)}`);
      }
    }, 1000);

    const results = [];
    for (let i = 0; i < TARGETS.length; i++) {
      const [x, y, z] = TARGETS[i];
      const goal = new GoalNear(x, y, z, 2);
      const start = bot.entity.position.clone();
      process.stdout.write(`[nav] #${(i + 1).toString().padStart(2)} -> (${x},${y},${z}) from ${fmt(start)} ... `);
      const r = await gotoTracked(bot, goal, args.perGoalMs);
      results.push({ i: i + 1, x, y, z, ...r });
      console.log(`${r.outcome} in ${(r.ms / 1000).toFixed(1)}s finalDist=${r.finalDist.toFixed(1)} minDist=${r.minDist.toFixed(1)} stuck=${r.stuckEvents}`);
      await sleep(700);
    }
    clearInterval(voidWatch);

    // ---- summary ----
    const reached = results.filter((r) => r.outcome === "reached").length;
    const timeouts = results.filter((r) => r.outcome === "timeout").length;
    const errors = results.filter((r) => r.outcome.startsWith("error")).length;
    // "effectively reached" = pathfinder reported reached OR ended within 3 blocks
    const effReached = results.filter((r) => r.outcome === "reached" || r.finalDist <= 3).length;
    const stuckGoals = results.filter((r) => r.stuckEvents > 0).length;
    const reachedTimes = results.filter((r) => r.outcome === "reached").map((r) => r.ms / 1000);
    const avgT = reachedTimes.length ? (reachedTimes.reduce((a, b) => a + b, 0) / reachedTimes.length) : 0;

    console.log("\n================ NAV TEST SUMMARY ================");
    console.log(`world: converted Fallen Kingdom (dense build, 1.19.4)`);
    console.log(`targets: ${TARGETS.length}`);
    console.log(`pathfinder 'reached': ${reached}/${TARGETS.length}`);
    console.log(`effectively reached (<=3 blocks): ${effReached}/${TARGETS.length}`);
    console.log(`timeouts: ${timeouts}   errors: ${errors}`);
    console.log(`goals with stuck/no-progress events: ${stuckGoals}/${TARGETS.length}`);
    console.log(`walk-off/void-fall events: ${walkOffEvents}   deaths: ${deaths}`);
    console.log(`avg time per reached target: ${avgT.toFixed(1)}s`);
    console.log("\nfailed / partial goals:");
    for (const r of results.filter((r) => r.outcome !== "reached")) {
      console.log(`  #${r.i} (${r.x},${r.y},${r.z}) ${r.outcome} finalDist=${r.finalDist.toFixed(1)} minDist=${r.minDist.toFixed(1)} stuck=${r.stuckEvents}`);
    }
    console.log("==================================================");
    await sleep(1000);
    bot.quit("done");
  });
}
main();
