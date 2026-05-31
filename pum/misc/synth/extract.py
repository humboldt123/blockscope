"""
synth/extract.py — Download Minecraft 1.19.2 jar and extract assets needed
for synthetic inventory generation:
  - assets/minecraft/textures/item/   (2D item sprites)
  - assets/minecraft/textures/block/  (block face textures)
  - assets/minecraft/models/item/     (model JSONs — tells us how to render each item)
  - assets/minecraft/models/block/    (block model JSONs)

Output directory structure (under synth/assets/):
  textures/item/*.png
  textures/block/*.png
  models/item/*.json
  models/block/*.json

Run once:
  python synth/extract.py
"""

import json
import os
import urllib.request
import zipfile
from pathlib import Path

OUT_DIR      = Path(__file__).parent / "assets"
CACHE_DIR    = Path(__file__).parent / ".cache"
MC_VERSION   = "1.19.2"

MANIFEST_URL = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"

EXTRACT_PREFIXES = (
    "assets/minecraft/textures/item/",
    "assets/minecraft/textures/block/",
    "assets/minecraft/models/item/",
    "assets/minecraft/models/block/",
)


def _download(url: str, dest: Path, desc: str = "") -> Path:
    if dest.exists():
        print(f"  [cached] {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or dest.name} …")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved → {dest}")
    return dest


def get_jar_url(version: str) -> str:
    manifest_path = CACHE_DIR / "version_manifest_v2.json"
    _download(MANIFEST_URL, manifest_path, "version manifest")
    with open(manifest_path) as f:
        manifest = json.load(f)

    version_url = next(
        v["url"] for v in manifest["versions"] if v["id"] == version
    )
    version_meta_path = CACHE_DIR / f"{version}.json"
    _download(version_url, version_meta_path, f"{version} metadata")
    with open(version_meta_path) as f:
        meta = json.load(f)

    return meta["downloads"]["client"]["url"]


def extract_assets(jar_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExtracting assets from {jar_path.name} …")
    n = 0
    with zipfile.ZipFile(jar_path) as zf:
        for name in zf.namelist():
            if not any(name.startswith(p) for p in EXTRACT_PREFIXES):
                continue
            # Strip "assets/minecraft/" prefix so paths become textures/item/*, etc.
            rel = name[len("assets/minecraft/"):]
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            n += 1
    print(f"Extracted {n} files → {out_dir}")


def main():
    print(f"=== Minecraft {MC_VERSION} asset extraction ===\n")
    jar_url  = get_jar_url(MC_VERSION)
    jar_path = _download(jar_url, CACHE_DIR / f"minecraft-{MC_VERSION}.jar",
                         f"Minecraft {MC_VERSION} client jar")
    extract_assets(jar_path, OUT_DIR)
    print("\nDone.")
    _report(OUT_DIR)


def _report(out_dir: Path):
    for subdir in ("textures/item", "textures/block", "models/item", "models/block"):
        p = out_dir / subdir
        if p.exists():
            n = len(list(p.iterdir()))
            print(f"  {subdir:<30} {n} files")


if __name__ == "__main__":
    main()
