#!/usr/bin/env python3
"""
Generate fresh random-seed Minecraft 1.8 natural-terrain worlds and land them in
the Blockscope VPS map_library so they enter the data-collection rotation as
old-block (pre-flattening) "library" worlds.

This is Option A (vanilla, small worlds) of the worldgen plan — VERIFIED working.

Per seed, the pipeline is:
  1. (once) download & cache the vanilla 1.8.9 server.jar at ~/mc18/server.jar
  2. in a fresh temp run dir, write eula.txt + server.properties (with the seed),
     then run the 1.8.9 server headlessly, piping `save-all\nstop\n` into it. The
     server generates the spawn-area chunks (~625) and exits. Result: world/ is
     pure 1.8 (level.dat + region/*.mca).
  3. convert 1.8 -> 1.19.4 with ~/mc-converter.jar (DataVersion 3337):
       java -jar ~/mc-converter.jar --input <run>/world --output <library>/generated_1_8_<seed>
  4. generate spawn candidates:
       python3 ~/generate_spawns.py --world <library>/generated_1_8_<seed>
  5. verify (DataVersion==3337, non-empty spawn_candidates.json) and clean up the run dir.

Re-runnable. Existing worlds in map_library are skipped (idempotent on seed).

Usage (run on the VPS):
    python3 generate_world.py --count 3            # 3 fresh random seeds
    python3 generate_world.py --seeds 123,-456,789 # specific seeds
    python3 generate_world.py --count 5 --keep-runs # keep temp dirs for debugging

After generating, restart Lodestone so it discovers the new worlds:
    cd ~/blockscope && screen -S mc -X quit; sleep 3; \
      screen -dmS mc bash -c './start.sh 2>&1 | tee server.log'
    sleep 25; grep 'Discovered .* library worlds' server.log

Requires: Java 21 (for both the 1.8.9 server and the converter), python3 + nbtlib
(for generate_spawns.py), curl. All present on the VPS.
"""

import argparse
import gzip
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

HOME = os.path.expanduser("~")

# --- VPS layout (overridable via flags) ------------------------------------
DEFAULT_LIBRARY = os.path.join(HOME, "blockscope", "map_library")
DEFAULT_CONVERTER = os.path.join(HOME, "mc-converter.jar")
DEFAULT_SPAWNGEN = os.path.join(HOME, "generate_spawns.py")
DEFAULT_SERVERJAR = os.path.join(HOME, "mc18", "server.jar")

# Vanilla 1.8.9 server.jar (Mojang official object store).
SERVERJAR_URL = (
    "https://piston-data.mojang.com/v1/objects/"
    "b58b2ceb36e01bcd8dbf49c8fb66c55a9f0676cd/server.jar"
)
SERVERJAR_SHA1 = "b58b2ceb36e01bcd8dbf49c8fb66c55a9f0676cd"

EXPECTED_DATAVERSION = 3337  # 1.19.4, what mc-converter.jar emits
WORLD_PREFIX = "generated_1_8_"
SERVER_TIMEOUT = 600  # seconds to let the 1.8 server generate + stop


def log(msg):
    print(msg, flush=True)


def rand_seed():
    """A random signed 64-bit seed, as Minecraft level-seed accepts."""
    return random.randint(-(2**63), 2**63 - 1)


def ensure_server_jar(path):
    """Download & cache the 1.8.9 server.jar once; verify SHA1 if present."""
    if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
        log(f"[jar] cached 1.8.9 server.jar present: {path}")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log(f"[jar] downloading 1.8.9 server.jar -> {path}")
    tmp = path + ".part"
    with urllib.request.urlopen(SERVERJAR_URL, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    # Optional integrity check.
    try:
        import hashlib
        h = hashlib.sha1()
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != SERVERJAR_SHA1:
            os.remove(tmp)
            raise SystemExit(
                f"[jar] SHA1 mismatch (got {h.hexdigest()}, want {SERVERJAR_SHA1})"
            )
    except ImportError:
        pass
    os.replace(tmp, path)
    log(f"[jar] downloaded & verified ({os.path.getsize(path)} bytes)")
    return path


def write_server_config(run_dir, seed):
    with open(os.path.join(run_dir, "eula.txt"), "w") as f:
        f.write("eula=true\n")
    props = {
        "level-seed": str(seed),
        "level-type": "DEFAULT",
        "online-mode": "false",
        "spawn-protection": "0",
        "level-name": "world",
        # keep generation cheap & deterministic-ish for spawn area only
        "generate-structures": "true",
        "view-distance": "10",
        "max-players": "1",
        "allow-nether": "false",
        "spawn-npcs": "false",
        "spawn-animals": "true",
        "spawn-monsters": "false",
    }
    with open(os.path.join(run_dir, "server.properties"), "w") as f:
        for k, v in props.items():
            f.write(f"{k}={v}\n")


def run_18_server(run_dir, server_jar, xmx="2G"):
    """Run the 1.8.9 server headlessly; pipe save-all + stop so it generates & exits."""
    cmd = ["java", f"-Xmx{xmx}", "-jar", server_jar, "nogui"]
    log(f"[gen] running 1.8.9 server in {run_dir} ...")
    proc = subprocess.run(
        cmd,
        cwd=run_dir,
        input="save-all\nstop\n",
        text=True,
        capture_output=True,
        timeout=SERVER_TIMEOUT,
    )
    world = os.path.join(run_dir, "world")
    region = os.path.join(world, "region")
    if not (os.path.isfile(os.path.join(world, "level.dat")) and os.path.isdir(region)):
        tail = (proc.stdout or "")[-1500:] + "\n--STDERR--\n" + (proc.stderr or "")[-800:]
        raise RuntimeError(f"1.8 server did not produce world/region:\n{tail}")
    mca = [f for f in os.listdir(region) if f.endswith(".mca")]
    if not mca:
        raise RuntimeError("1.8 server produced an empty region/ dir")
    log(f"[gen] generated world/ with {len(mca)} region file(s)")
    return world


def convert_world(converter, src_world, out_dir):
    log(f"[conv] converting -> {out_dir}")
    r = subprocess.run(
        ["java", "-jar", converter, "--input", src_world, "--output", out_dir],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if r.returncode != 0 or not os.path.isfile(os.path.join(out_dir, "level.dat")):
        tail = (r.stderr or "")[-1000:] or (r.stdout or "")[-1000:]
        raise RuntimeError(f"converter failed (rc={r.returncode}):\n{tail}")


def gen_spawns(spawngen, out_dir):
    log(f"[spawn] generating spawn candidates for {out_dir}")
    subprocess.run(["python3", spawngen, "--world", out_dir], check=True, timeout=600)


def world_dataversion(world_dir):
    """Read DataVersion from a (gzip) level.dat without nbtlib."""
    p = os.path.join(world_dir, "level.dat")
    try:
        data = gzip.open(p, "rb").read()
    except OSError:
        data = open(p, "rb").read()
    i = data.find(b"DataVersion")
    if i < 0:
        return None
    try:
        return struct.unpack(">i", data[i + 11 : i + 15])[0]
    except struct.error:
        return None


def verify_world(out_dir):
    """Return (ok, dataversion, spawn_count, msg)."""
    dv = world_dataversion(out_dir)
    if dv != EXPECTED_DATAVERSION:
        return False, dv, 0, f"DataVersion {dv} != {EXPECTED_DATAVERSION}"
    sc = os.path.join(out_dir, "spawn_candidates.json")
    if not os.path.isfile(sc):
        return False, dv, 0, "spawn_candidates.json missing"
    try:
        spawns = json.load(open(sc)).get("spawns", [])
    except json.JSONDecodeError as e:
        return False, dv, 0, f"spawn_candidates.json invalid: {e}"
    if not spawns:
        return False, dv, 0, "spawn_candidates.json empty"
    return True, dv, len(spawns), "ok"


def generate_one(seed, args):
    name = f"{WORLD_PREFIX}{seed}"
    out_dir = os.path.join(args.library, name)
    if os.path.exists(out_dir):
        log(f"== seed {seed}: {name} already exists in map_library, skipping ==")
        ok, dv, n, msg = verify_world(out_dir)
        return {"seed": seed, "name": name, "status": "exists",
                "spawns": n, "dataversion": dv, "msg": msg}

    log(f"\n===== seed {seed} -> {name} =====")
    run_dir = tempfile.mkdtemp(prefix=f"mc18run_{seed}_", dir=args.workdir or None)
    try:
        write_server_config(run_dir, seed)
        src_world = run_18_server(run_dir, args.server_jar, xmx=args.xmx)
        convert_world(args.converter, src_world, out_dir)
        gen_spawns(args.spawngen, out_dir)
        ok, dv, n, msg = verify_world(out_dir)
        if not ok:
            log(f"[verify] FAILED: {msg}")
            shutil.rmtree(out_dir, ignore_errors=True)
            return {"seed": seed, "name": name, "status": "failed",
                    "spawns": 0, "dataversion": dv, "msg": msg}
        log(f"[verify] OK: DataVersion={dv}, {n} spawn candidates")
        return {"seed": seed, "name": name, "status": "generated",
                "spawns": n, "dataversion": dv, "msg": "ok"}
    except Exception as e:  # noqa: BLE001 - report per-seed, continue the batch
        log(f"[error] seed {seed}: {e}")
        shutil.rmtree(out_dir, ignore_errors=True)
        return {"seed": seed, "name": name, "status": "error",
                "spawns": 0, "dataversion": None, "msg": str(e)}
    finally:
        if args.keep_runs:
            log(f"[cleanup] keeping run dir {run_dir}")
        else:
            shutil.rmtree(run_dir, ignore_errors=True)


def parse_seeds(args):
    if args.seeds:
        out = []
        for tok in args.seeds.split(","):
            tok = tok.strip()
            if tok:
                out.append(int(tok))
        return out
    return [rand_seed() for _ in range(args.count)]


def main():
    ap = argparse.ArgumentParser(
        description="Generate fresh random-seed MC 1.8 worlds into the Blockscope map_library."
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--count", type=int, default=3, help="number of random-seed worlds (default 3)")
    g.add_argument("--seeds", type=str, help="comma-separated explicit seeds, e.g. 123,-456")
    ap.add_argument("--library", default=DEFAULT_LIBRARY, help="map_library dir")
    ap.add_argument("--converter", default=DEFAULT_CONVERTER, help="path to mc-converter.jar")
    ap.add_argument("--spawngen", default=DEFAULT_SPAWNGEN, help="path to generate_spawns.py")
    ap.add_argument("--server-jar", default=DEFAULT_SERVERJAR, help="path to cache 1.8.9 server.jar")
    ap.add_argument("--workdir", default=None, help="parent dir for temp run dirs (default: system tmp)")
    ap.add_argument("--xmx", default="2G", help="max heap for the 1.8 server (default 2G)")
    ap.add_argument("--keep-runs", action="store_true", help="don't delete temp run dirs")
    args = ap.parse_args()

    for p, what in [(args.converter, "mc-converter.jar"), (args.spawngen, "generate_spawns.py")]:
        if not os.path.isfile(p):
            raise SystemExit(f"required file not found: {what} at {p}")
    os.makedirs(args.library, exist_ok=True)

    ensure_server_jar(args.server_jar)
    seeds = parse_seeds(args)
    log(f"[plan] {len(seeds)} world(s): {seeds}")

    t0 = time.time()
    results = [generate_one(seed, args) for seed in seeds]

    log("\n===== SUMMARY =====")
    by = {}
    for r in results:
        by.setdefault(r["status"], []).append(r)
        log(f"  {r['status']:10s} {r['name']}  spawns={r['spawns']}  dv={r['dataversion']}  {r['msg']}")
    gen = by.get("generated", [])
    log(f"\nGenerated {len(gen)} new world(s) in {time.time()-t0:.0f}s.")
    if gen:
        log("Names: " + ", ".join(r["name"] for r in gen))
    if by.get("failed") or by.get("error"):
        log("Some seeds did not succeed; see rows above.")
    log(
        "\nTo activate, restart Lodestone so it discovers the new worlds:\n"
        "  cd ~/blockscope && screen -S mc -X quit; sleep 3; "
        "screen -dmS mc bash -c './start.sh 2>&1 | tee server.log'\n"
        "  sleep 25; grep 'Discovered .* library worlds' server.log"
    )
    # nonzero exit if nothing succeeded and something was attempted
    if seeds and not gen and not by.get("exists"):
        sys.exit(1)


if __name__ == "__main__":
    main()
