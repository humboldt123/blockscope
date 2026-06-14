#!/usr/bin/env python3
"""
Batch-convert every archive in map_downloads/ to 1.19.4 in map_library/.

For each .zip (and .rar if unrar is available):
  1. extract to map_staging/<slug>/
  2. locate the world dir (the dir containing level.dat)
  3. read DataVersion: convert only PRE-1.13 worlds (DataVersion absent or < 1519).
     1.13+ worlds are already "flattened" and the numeric-ID converter would corrupt
     them — those are skipped (need a DFU upgrade path instead).
  4. run mc-converter.jar  -> map_library/<slug>
  5. run generate_spawns.py on the converted world

Run on the VPS:  python3 batch_convert.py
"""
import gzip, os, re, shutil, struct, subprocess, sys, zipfile

HOME       = os.path.expanduser("~")
BASE       = os.path.join(HOME, "blockscope")
DOWNLOADS  = os.path.join(BASE, "map_downloads")
STAGING    = os.path.join(BASE, "map_staging")
LIBRARY    = os.path.join(BASE, "map_library")
CONVERTER  = os.path.join(HOME, "mc-converter.jar")
SPAWNGEN   = os.path.join(HOME, "generate_spawns.py")
FLATTEN_DV = 1519  # 1.13 — first flattened DataVersion

def slug(name):
    s = re.sub(r"\.(zip|rar)$", "", name, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s

def data_version(level_dat):
    """Return DataVersion int, or None if absent/unreadable."""
    try:
        d = gzip.open(level_dat, "rb").read()
    except Exception:
        try:
            d = open(level_dat, "rb").read()
        except Exception:
            return None
    i = d.find(b"DataVersion")
    if i < 0:
        return None
    try:
        return struct.unpack(">i", d[i+11:i+15])[0]
    except Exception:
        return None

def find_world_dir(root):
    """First dir containing level.dat AND a region/ folder."""
    for dirpath, dirnames, filenames in os.walk(root):
        if "level.dat" in filenames and os.path.isdir(os.path.join(dirpath, "region")):
            return dirpath
    # fall back to any dir with level.dat
    for dirpath, dirnames, filenames in os.walk(root):
        if "level.dat" in filenames:
            return dirpath
    return None

def extract(archive, dest):
    os.makedirs(dest, exist_ok=True)
    if archive.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
        return True
    if archive.lower().endswith(".rar"):
        if shutil.which("unrar"):
            subprocess.run(["unrar", "x", "-o+", archive, dest + "/"], check=True)
            return True
        print(f"  SKIP {os.path.basename(archive)}: unrar not installed")
        return False
    return False

def main():
    os.makedirs(STAGING, exist_ok=True)
    os.makedirs(LIBRARY, exist_ok=True)
    results = {"converted": [], "skipped_modern": [], "skipped_other": [], "failed": []}

    archives = sorted(f for f in os.listdir(DOWNLOADS) if f.lower().endswith((".zip", ".rar")))
    for arc in archives:
        name = slug(arc)
        out  = os.path.join(LIBRARY, name)
        if os.path.exists(out):
            print(f"== {arc}: already in map_library/{name}, skipping ==")
            continue
        print(f"== {arc} -> {name} ==")
        stage = os.path.join(STAGING, name)
        shutil.rmtree(stage, ignore_errors=True)
        try:
            if not extract(os.path.join(DOWNLOADS, arc), stage):
                results["skipped_other"].append(arc); continue
        except Exception as e:
            print(f"  extract failed: {e}"); results["failed"].append(arc); continue

        world = find_world_dir(stage)
        if not world:
            print("  no level.dat found"); results["skipped_other"].append(arc); continue

        dv = data_version(os.path.join(world, "level.dat"))
        print(f"  world dir: {os.path.relpath(world, stage)}  DataVersion={dv}")
        if dv is not None and dv >= FLATTEN_DV:
            print(f"  1.13+ (flattened) — converter can't handle; SKIP")
            results["skipped_modern"].append(arc); shutil.rmtree(stage, ignore_errors=True); continue

        try:
            r = subprocess.run(["java", "-jar", CONVERTER, "--input", world, "--output", out],
                               capture_output=True, text=True, timeout=1800)
            if r.returncode != 0 or not os.path.exists(os.path.join(out, "level.dat")):
                print(f"  CONVERT FAILED: {r.stderr[-300:]}")
                results["failed"].append(arc); shutil.rmtree(out, ignore_errors=True)
                shutil.rmtree(stage, ignore_errors=True); continue
        except Exception as e:
            print(f"  CONVERT EXC: {e}"); results["failed"].append(arc)
            shutil.rmtree(stage, ignore_errors=True); continue

        # spawns
        if os.path.exists(SPAWNGEN):
            try:
                subprocess.run(["python3", SPAWNGEN, "--world", out], timeout=600)
            except Exception as e:
                print(f"  spawn gen failed: {e}")
        print(f"  CONVERTED -> map_library/{name}")
        results["converted"].append(name)
        shutil.rmtree(stage, ignore_errors=True)

    print("\n===== SUMMARY =====")
    for k, v in results.items():
        print(f"{k} ({len(v)}): {v}")

if __name__ == "__main__":
    main()
