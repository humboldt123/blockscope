#!/usr/bin/env python3
"""
Install Fabric 1.19.4 (offline) and launch the client with the blockscope mod set.

- Uses minecraft-launcher-lib to download the vanilla 1.19.4 client + libraries and
  the Fabric loader, then assembles the java launch command.
- Offline auth: the VPS runs online-mode=false, so any username works and no
  Microsoft login is needed.
- Mods and config are copied from /assets into the game dir before launch.
- The blockscope mod auto-connects to BLOCKSCOPE_AUTOCONNECT_HOST/PORT once the
  title screen appears (1.19.4 has no --quickPlayMultiplayer flag).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import minecraft_launcher_lib as mll

MC_VERSION = "1.19.4"
GAME_DIR = Path(os.environ.get("BLOCKSCOPE_GAME_DIR", "/app/game"))
ASSETS = Path("/assets")
USERNAME = os.environ.get("BLOCKSCOPE_USERNAME", "blockscope_bot")


def log(msg: str) -> None:
    print(f"[launch] {msg}", flush=True)


def install_fabric() -> str:
    GAME_DIR.mkdir(parents=True, exist_ok=True)

    def _cb(value):
        pass

    callback = {"setStatus": lambda s: log(s), "setProgress": _cb, "setMax": _cb}

    log(f"Installing vanilla Minecraft {MC_VERSION} ...")
    mll.install.install_minecraft_version(MC_VERSION, str(GAME_DIR), callback=callback)

    log("Installing Fabric loader ...")
    mll.fabric.install_fabric(MC_VERSION, str(GAME_DIR), callback=callback)

    # Find the installed fabric-loader version id.
    versions = mll.utils.get_installed_versions(str(GAME_DIR))
    fabric_ids = [v["id"] for v in versions if "fabric" in v["id"].lower() and MC_VERSION in v["id"]]
    if not fabric_ids:
        raise RuntimeError(f"No fabric version installed; have: {[v['id'] for v in versions]}")
    fabric_id = sorted(fabric_ids)[-1]
    log(f"Using version profile: {fabric_id}")
    return fabric_id


def stage_mods_and_config() -> None:
    mods_dst = GAME_DIR / "mods"
    mods_dst.mkdir(parents=True, exist_ok=True)
    src_mods = ASSETS / "mods"
    count = 0
    for jar in src_mods.glob("*.jar"):
        shutil.copy2(jar, mods_dst / jar.name)
        count += 1
    log(f"Staged {count} mod jars into {mods_dst}")

    # blockscope.properties must sit in the game dir (mod reads it from CWD).
    cfg_src = ASSETS / "blockscope.properties"
    if cfg_src.exists():
        shutil.copy2(cfg_src, GAME_DIR / "blockscope.properties")
        log("Staged blockscope.properties")
    else:
        log("WARNING: no blockscope.properties in /assets")


def build_command(fabric_id: str) -> list:
    options = {
        "username": USERNAME,
        "uuid": "00000000-0000-0000-0000-000000000000",
        "token": "0",
        "jvmArguments": [
            "-Xmx4G",
            "-Djava.awt.headless=false",
            # Tell our vendored ReplayMod to skip the startup recovery scan (no
            # "recover recording?" modal, no race with the live recording).
            "-Dblockscope.headless=true",
            # Fail fast & loud if GL init dies, so logs show the real error.
            "-Dfml.earlyprogresswindow=false",
        ],
        "launcherName": "blockscope-headless",
        "launcherVersion": "1.0",
        "gameDirectory": str(GAME_DIR),
    }
    cmd = mll.command.get_minecraft_command(fabric_id, str(GAME_DIR), options)
    return cmd


def main() -> int:
    fabric_id = install_fabric()
    stage_mods_and_config()
    cmd = build_command(fabric_id)

    log("Launch command:")
    log(" ".join(cmd))

    # Run from the game dir so the mod finds blockscope.properties in CWD.
    proc = subprocess.run(cmd, cwd=str(GAME_DIR))
    log(f"Minecraft exited with code {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
