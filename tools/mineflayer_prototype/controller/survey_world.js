#!/usr/bin/env node
// Quick world survey: connect, report spawn, sample surface heights across the
// build area so we can pick reachable nav targets at varied elevations.
const mineflayer = require("mineflayer");
const Vec3 = require("vec3").Vec3;

const HOST = "127.0.0.1", PORT = 25599, USER = "survey_bot";
const bot = mineflayer.createBot({ host: HOST, port: PORT, username: USER, version: "1.19.4", auth: "offline" });

bot.on("error", (e) => console.log("[survey] error:", e.message));
bot.on("kicked", (r) => console.log("[survey] kicked:", r));
bot.on("end", () => process.exit(0));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

bot.once("spawn", async () => {
  const p = bot.entity.position;
  console.log(`[survey] spawn at ${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)}`);
  // Teleport into the structure cluster (warnings showed builds ~X75-329, Z163-406)
  // so chunks load. We op ourselves via server console separately; here just observe.
  // Wait for chunks to stream in.
  await sleep(4000);

  // Sample surface height (highest solid block) on a grid across the build area.
  function surfaceY(x, z) {
    for (let y = 120; y >= -5; y--) {
      const b = bot.blockAt(new Vec3(x, y, z));
      if (b && b.boundingBox === "block") return { y, name: b.name };
    }
    return null;
  }

  const samples = [];
  for (let x = 90; x <= 320; x += 20) {
    for (let z = 170; z <= 400; z += 20) {
      const s = surfaceY(x, z);
      if (s) samples.push({ x, z, ...s });
    }
  }
  console.log(`[survey] ${samples.length} surface samples (only loaded chunks return data):`);
  for (const s of samples) {
    console.log(`  (${s.x}, ${s.y}, ${s.z}) top=${s.name}`);
  }
  // Also report block at spawn neighborhood
  console.log("[survey] done");
  bot.quit("survey done");
});
