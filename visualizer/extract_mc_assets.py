"""
Extract Minecraft 1.19.4 assets for inventory rendering in visualizer.
Downloads client JAR and extracts:
  - assets/minecraft/textures/item/*.png
  - assets/minecraft/models/item/*.json

Run once: python extract_mc_assets.py
"""

import json
import urllib.request
import zipfile
from pathlib import Path

OUT_DIR = Path(__file__).parent / "mc_assets"
CACHE_DIR = Path(__file__).parent / ".cache"
MC_VERSION = "1.19.4"

MANIFEST_URL = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"

EXTRACT_PREFIXES = (
    "assets/minecraft/textures/item/",
    "assets/minecraft/models/item/",
)


def download(url: str, dest: Path, desc: str = "") -> Path:
    if dest.exists():
        print(f"  [cached] {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or dest.name}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved -> {dest}")
    return dest


def get_jar_url(version: str) -> str:
    manifest_path = CACHE_DIR / "version_manifest_v2.json"
    download(MANIFEST_URL, manifest_path, "version manifest")
    with open(manifest_path) as f:
        manifest = json.load(f)

    version_url = next(v["url"] for v in manifest["versions"] if v["id"] == version)
    version_meta_path = CACHE_DIR / f"{version}.json"
    download(version_url, version_meta_path, f"{version} metadata")
    with open(version_meta_path) as f:
        meta = json.load(f)

    return meta["downloads"]["client"]["url"]


def extract_assets(jar_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExtracting assets from {jar_path.name}...")
    n = 0
    with zipfile.ZipFile(jar_path) as zf:
        for name in zf.namelist():
            if not any(name.startswith(p) for p in EXTRACT_PREFIXES):
                continue
            # Strip "assets/minecraft/" prefix
            rel = name[len("assets/minecraft/"):]
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            n += 1
    print(f"Extracted {n} files -> {out_dir}")


if __name__ == "__main__":
    print(f"=== Minecraft {MC_VERSION} asset extraction ===\n")
    jar_url = get_jar_url(MC_VERSION)
    jar_path = download(jar_url, CACHE_DIR / f"minecraft-{MC_VERSION}.jar",
                        f"Minecraft {MC_VERSION} client jar")
    extract_assets(jar_path, OUT_DIR)
    print("\nDone. Assets extracted to:", OUT_DIR)
