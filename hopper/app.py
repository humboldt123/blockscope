#!/usr/bin/env python3
"""
Blockscope Server - FastAPI upload endpoint for recording sessions
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Blockscope Server",
    version="1.0.0",
    description="Upload endpoint for Blockscope Minecraft recording sessions"
)

# Add CORS middleware to allow uploads from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base data directory (local for debugging)
DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / "blockscope_data"))

# Create base directory if it doesn't exist
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory initialized: {DATA_DIR}")

    # Test write permissions
    test_file = DATA_DIR / ".write_test"
    test_file.write_text("test")
    test_file.unlink()
    logger.info("Write permissions verified")

except Exception as e:
    logger.error(f"Failed to initialize data directory: {e}")
    logger.error(f"Please ensure {DATA_DIR} exists and is writable")
    raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "blockscope-server",
        "version": "1.0.0",
        "data_dir": str(DATA_DIR),
        "writable": DATA_DIR.exists() and os.access(DATA_DIR, os.W_OK)
    }


@app.post("/upload")
async def upload_session(
    session_id: str = Form(...),
    ticks: Optional[UploadFile] = File(None),
    inputs: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None)
):
    """
    Upload endpoint for Blockscope recording sessions.

    Accepts multipart form data with:
    - session_id: string identifier for the session
    - ticks.jsonl: game state data (optional)
    - inputs.jsonl: input events (optional)
    - video.mp4: recorded video (optional)

    Saves files to /data/vvm33/BLOCKSCOPE_DATA/<session_id>/
    """
    try:
        # Create session directory
        session_dir = DATA_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Receiving session: {session_id}")
        print(f"[Upload] Receiving session: {session_id}")

        files_saved = []

        # Save ticks.jsonl
        if ticks:
            ticks_path = session_dir / "ticks.jsonl"
            with open(ticks_path, "wb") as f:
                shutil.copyfileobj(ticks.file, f)
            file_size_mb = ticks_path.stat().st_size / (1024 * 1024)
            print(f"[Upload] Saved ticks.jsonl ({file_size_mb:.2f} MB)")
            files_saved.append("ticks.jsonl")

        # Save inputs.jsonl
        if inputs:
            inputs_path = session_dir / "inputs.jsonl"
            with open(inputs_path, "wb") as f:
                shutil.copyfileobj(inputs.file, f)
            file_size_mb = inputs_path.stat().st_size / (1024 * 1024)
            print(f"[Upload] Saved inputs.jsonl ({file_size_mb:.2f} MB)")
            files_saved.append("inputs.jsonl")

        # Save video.mp4
        if video:
            video_path = session_dir / "video.mp4"
            with open(video_path, "wb") as f:
                shutil.copyfileobj(video.file, f)
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"[Upload] Saved video.mp4 ({file_size_mb:.2f} MB)")
            files_saved.append("video.mp4")

        if not files_saved:
            raise HTTPException(status_code=400, detail="No files were uploaded")

        print(f"[Upload] Session {session_id} upload complete: {', '.join(files_saved)}")

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "session_id": session_id,
                "files_saved": files_saved,
                "save_path": str(session_dir)
            }
        )

    except Exception as e:
        print(f"[Upload] Error processing session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream-video")
async def stream_video(
    request: Request,
    session_id: str = Query(...)
):
    """
    Streaming video upload endpoint.
    Accepts raw H264 stream, converts to MP4 using ffmpeg.
    """
    try:
        import subprocess

        session_dir = DATA_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        h264_path = session_dir / "video.h264"
        mp4_path = session_dir / "video.mp4"

        logger.info(f"Starting H264 stream for session: {session_id}")
        print(f"[StreamVideo] Receiving raw H264 stream: {session_id}")

        # Stream raw H264 to file
        bytes_written = 0
        with open(h264_path, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                bytes_written += len(chunk)

        size_mb = bytes_written / (1024 * 1024)
        logger.info(f"H264 stream complete: {session_id}, {size_mb:.2f} MB")
        print(f"[StreamVideo] H264 received: {size_mb:.2f} MB, converting to MP4...")

        # Convert H264 to MP4 using ffmpeg
        result = subprocess.run([
            "ffmpeg",
            "-f", "h264",
            "-i", str(h264_path),
            "-c:v", "copy",
            "-movflags", "+faststart",
            "-y",
            str(mp4_path)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {result.stderr}")
            print(f"[StreamVideo] ffmpeg error: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Video conversion failed: {result.stderr}")

        # Delete H264 file to save space
        h264_path.unlink()
        mp4_size = mp4_path.stat().st_size / (1024 * 1024)

        logger.info(f"Conversion complete: {session_id}, MP4: {mp4_size:.2f} MB")
        print(f"[StreamVideo] Complete: {mp4_size:.2f} MB MP4 created")

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "session_id": session_id,
                "h264_bytes": bytes_written,
                "mp4_file": str(mp4_path),
                "mp4_size_mb": round(mp4_size, 2)
            }
        )
    except Exception as e:
        logger.error(f"Stream video error: {str(e)}")
        print(f"[StreamVideo] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream-data")
async def stream_data(
    request: Request,
    session_id: str = Query(...),
    data_type: str = Query(...)  # "ticks" or "inputs"
):
    """
    Stream ticks.jsonl or inputs.jsonl line-by-line as recording progresses.
    Appends to file as data arrives.
    """
    try:
        session_dir = DATA_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        if data_type not in ["ticks", "inputs", "world_events", "frame_mapping"]:
            raise HTTPException(status_code=400, detail="data_type must be 'ticks', 'inputs', or 'world_events'")

        file_path = session_dir / f"{data_type}.jsonl"

        logger.info(f"Streaming {data_type} data for session: {session_id}")
        print(f"[StreamData] Receiving {data_type} stream: {session_id}")

        # Append to file as chunks arrive
        lines_written = 0
        bytes_written = 0
        with open(file_path, "ab") as f:  # Append mode
            async for chunk in request.stream():
                f.write(chunk)
                bytes_written += len(chunk)
                # Count newlines to estimate lines
                lines_written += chunk.count(b'\n')

        size_mb = bytes_written / (1024 * 1024)
        logger.info(f"{data_type} stream complete: {session_id}, {size_mb:.2f} MB, ~{lines_written} lines")
        print(f"[StreamData] {data_type} received: {size_mb:.2f} MB, ~{lines_written} lines")

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "session_id": session_id,
                "data_type": data_type,
                "size_mb": round(size_mb, 2),
                "lines": lines_written
            }
        )
    except Exception as e:
        logger.error(f"Stream data error: {str(e)}")
        print(f"[StreamData] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/init-session")
async def init_session(session_id: str = Query(...)):
    """Create session directory and register the session."""
    try:
        session_dir = DATA_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Session initialized: {session_id}")
        return JSONResponse(status_code=200, content={"ok": True, "session_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-replay")
async def upload_replay(
    request: Request,
    session_id: str = Query(...),
    filename: str = Query(...)
):
    """
    Upload the .mcpr replay file recorded by ReplayMod.
    Called after recording stops (once ReplayMod finishes writing the file).
    """
    try:
        session_dir = DATA_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        replay_path = session_dir / filename
        logger.info(f"Receiving replay: {session_id}/{filename}")
        print(f"[UploadReplay] Receiving: {session_id}/{filename}")

        bytes_written = 0
        with open(replay_path, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                bytes_written += len(chunk)

        size_mb = bytes_written / (1024 * 1024)
        logger.info(f"Replay saved: {filename}, {size_mb:.2f} MB")
        print(f"[UploadReplay] Saved: {filename} ({size_mb:.2f} MB)")

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "session_id": session_id,
                "filename": filename,
                "size_mb": round(size_mb, 2)
            }
        )
    except Exception as e:
        logger.error(f"Upload replay error: {str(e)}")
        print(f"[UploadReplay] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    """List all uploaded sessions"""
    try:
        sessions = []
        for session_dir in DATA_DIR.iterdir():
            if session_dir.is_dir():
                files = [f.name for f in session_dir.iterdir() if f.is_file()]
                total_size = sum(f.stat().st_size for f in session_dir.iterdir() if f.is_file())
                sessions.append({
                    "session_id": session_dir.name,
                    "files": files,
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                })

        return {
            "ok": True,
            "count": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import asyncio
from datetime import datetime

# Track last segment received time per session for auto-finalization
session_last_activity = {}
FINALIZE_TIMEOUT_SECONDS = 15

@app.post("/upload-segment")
async def upload_segment(
    request: Request,
    session_id: str = Query(...)
):
    """
    Upload a single MPEG-TS segment.
    Segments are uploaded as they complete and concatenated at the end.
    """
    try:
        session_dir = DATA_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Get segment filename from header
        segment_name = request.headers.get("X-Segment-Name", "segment.ts")
        segment_path = session_dir / segment_name

        logger.info(f"Receiving segment: {session_id}/{segment_name}")
        print(f"[UploadSegment] Receiving: {session_id}/{segment_name}")

        # Write segment
        bytes_written = 0
        with open(segment_path, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                bytes_written += len(chunk)

        size_mb = bytes_written / (1024 * 1024)
        logger.info(f"Segment saved: {segment_name}, {size_mb:.2f} MB")
        print(f"[UploadSegment] Saved: {segment_name} ({size_mb:.2f} MB)")

        # Update last activity time and schedule auto-finalization
        session_last_activity[session_id] = datetime.now()
        asyncio.create_task(auto_finalize_if_idle(session_id))

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "session_id": session_id,
                "segment_name": segment_name,
                "size_mb": round(size_mb, 2)
            }
        )
    except Exception as e:
        logger.error(f"Upload segment error: {str(e)}")
        print(f"[UploadSegment] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def auto_finalize_if_idle(session_id: str):
    """
    Wait for timeout, then finalize if no new segments arrived.
    """
    await asyncio.sleep(FINALIZE_TIMEOUT_SECONDS)

    last_activity = session_last_activity.get(session_id)
    if last_activity is None:
        return  # Already finalized manually

    # Check if still idle
    elapsed = (datetime.now() - last_activity).total_seconds()
    if elapsed >= FINALIZE_TIMEOUT_SECONDS:
        print(f"[AutoFinalize] No activity for {FINALIZE_TIMEOUT_SECONDS}s, finalizing {session_id}")
        try:
            await finalize_video_internal(session_id)
        except Exception as e:
            logger.error(f"Auto-finalize failed for {session_id}: {e}")


@app.post("/finalize-video")
async def finalize_video(session_id: str = Query(...)):
    """
    Concatenate all TS segments into final MP4.
    Called when recording completes.
    """
    return await finalize_video_internal(session_id)


async def finalize_video_internal(session_id: str):
    """
    Internal finalization logic (can be called by manual endpoint or auto-finalize).
    """
    try:
        import subprocess
        import glob

        # Remove from tracking to prevent duplicate finalization
        session_last_activity.pop(session_id, None)

        session_dir = DATA_DIR / session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Find all segments — sort numerically, not lexicographically,
        # so segment_10 comes after segment_9 not after segment_1.
        import re
        def _seg_key(p):
            m = re.search(r'(\d+)', Path(p).stem)
            return int(m.group(1)) if m else 0

        segments = sorted(glob.glob(str(session_dir / "segment_*.ts")), key=_seg_key)

        if not segments:
            logger.warning(f"No segments found for {session_id}, skipping finalization")
            return JSONResponse(
                status_code=200,
                content={
                    "ok": True,
                    "session_id": session_id,
                    "note": "No video segments to finalize"
                }
            )

        logger.info(f"Finalizing video: {session_id}, {len(segments)} segments")
        print(f"[FinalizeVideo] Concatenating {len(segments)} segments for {session_id}")

        # Create concat file list
        concat_file = session_dir / "concat_list.txt"
        with open(concat_file, "w") as f:
            for seg in segments:
                f.write(f"file '{seg}'\n")

        # Concatenate segments to MP4
        mp4_path = session_dir / "video.mp4"
        result = subprocess.run([
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",  # No re-encoding, just remux
            "-y",
            str(mp4_path)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"ffmpeg concat failed: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Video concat failed: {result.stderr}")

        # Delete segments and concat file
        for seg in segments:
            Path(seg).unlink()
        concat_file.unlink()

        mp4_size = mp4_path.stat().st_size / (1024 * 1024)
        logger.info(f"Video finalized: {session_id}, {mp4_size:.2f} MB")
        print(f"[FinalizeVideo] Complete: {mp4_size:.2f} MB MP4 created")

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "session_id": session_id,
                "mp4_file": str(mp4_path),
                "mp4_size_mb": round(mp4_size, 2)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Finalize video error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-metadata")
async def upload_metadata(
    request: Request,
    session_id: str = Query(...)
):
    """
    Upload metadata.json for a session.
    Called at recording start and end to keep metadata in sync.
    """
    try:
        session_dir = DATA_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = session_dir / "metadata.json"

        logger.info(f"Receiving metadata for session: {session_id}")
        print(f"[UploadMetadata] Receiving metadata: {session_id}")

        # Read JSON from request body
        metadata_bytes = await request.body()

        # Write to file
        with open(metadata_path, "wb") as f:
            f.write(metadata_bytes)

        size_kb = len(metadata_bytes) / 1024
        logger.info(f"Metadata saved: {session_id}, {size_kb:.2f} KB")
        print(f"[UploadMetadata] Saved: {size_kb:.2f} KB")

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "session_id": session_id,
                "size_kb": round(size_kb, 2)
            }
        )
    except Exception as e:
        logger.error(f"Upload metadata error: {str(e)}")
        print(f"[UploadMetadata] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session"""
    try:
        session_dir = DATA_DIR / session_id

        if not session_dir.exists():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        files = []
        for f in session_dir.iterdir():
            if f.is_file():
                files.append({
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "path": str(f)
                })

        return {
            "ok": True,
            "session_id": session_id,
            "path": str(session_dir),
            "files": files
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/tick-at-time")
async def tick_at_time(session_id: str, replay_ms: int = Query(...)):
    """
    Given a replay timestamp (ms from the start of the .mcpr recording),
    return the tick entry whose replayMs is closest to that value.

    The ticks.jsonl file must contain entries with a 'replayMs' field
    (populated by Blockscope's PacketListener.getCurrentDuration() call).
    """
    try:
        ticks_path = DATA_DIR / session_id / "ticks.jsonl"
        if not ticks_path.exists():
            raise HTTPException(status_code=404, detail=f"No tick data for session {session_id}")

        import json

        best_tick = None
        best_delta = float("inf")

        with open(ticks_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tick = json.loads(line)
                    tick_ms = tick.get("replayMs", 0)
                    delta = abs(tick_ms - replay_ms)
                    if delta < best_delta:
                        best_delta = delta
                        best_tick = tick
                    # Early exit: once we pass the target and delta is growing, stop
                    if tick_ms > replay_ms and delta > best_delta:
                        break
                except json.JSONDecodeError:
                    continue

        if best_tick is None:
            raise HTTPException(status_code=404, detail="No ticks found in session")

        return JSONResponse(status_code=200, content={
            "ok": True,
            "session_id": session_id,
            "requested_ms": replay_ms,
            "actual_ms": best_tick.get("replayMs", 0),
            "delta_ms": best_delta,
            "tick": best_tick
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"tick-at-time error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Blockscope Server...")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Listening on: http://0.0.0.0:9000")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        log_level="info"
    )
