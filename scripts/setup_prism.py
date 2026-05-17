#!/usr/bin/env python3
"""Copy instance config files to the PrismLauncher directory for the Blockscope dev environment."""
import sys
import shutil
from pathlib import Path

INSTANCE_NAME = "Blockscope"
REPO_ROOT = Path(__file__).parent.parent
INSTANCE_SRC = REPO_ROOT / "instance"


def find_prism_instances() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "PrismLauncher" / "instances"
    elif sys.platform == "win32":
        import os
        appdata = os.environ.get("APPDATA", "")
        if not appdata:
            print("APPDATA not set — are you on Windows?")
            sys.exit(1)
        return Path(appdata) / "PrismLauncher" / "instances"
    else:
        return Path.home() / ".local" / "share" / "PrismLauncher" / "instances"


def main():
    instances_dir = find_prism_instances()
    if not instances_dir.exists():
        print(f"PrismLauncher instances not found at:\n  {instances_dir}")
        print("Install PrismLauncher: https://prismlauncher.org/")
        sys.exit(1)

    instance_dir = instances_dir / INSTANCE_NAME
    mc_dir = instance_dir / "minecraft"
    (mc_dir / "config").mkdir(parents=True, exist_ok=True)

    files = [
        ("mmc-pack.json",           instance_dir / "mmc-pack.json"),
        ("options.txt",             mc_dir / "options.txt"),
        ("blockscope.properties",   mc_dir / "blockscope.properties"),
        ("config/replaymod.json",   mc_dir / "config" / "replaymod.json"),
    ]

    print(f"Installing Blockscope instance config to:\n  {instance_dir}\n")
    for src_name, dest in files:
        shutil.copy2(INSTANCE_SRC / src_name, dest)
        print(f"  {src_name}")

    print(f"\nDone. Next steps:")
    print(f"  1. Build the mod:  cd mods/blockscope && ./gradlew build")
    print(f"  2. Copy build/libs/blockscope-*.jar into the instance mods/ folder")
    print(f"  3. Add replaymod, baritone, and fabric-api jars (see instance/README.md)")
    print(f"  4. Launch Prism and select the Blockscope instance")


if __name__ == "__main__":
    main()
