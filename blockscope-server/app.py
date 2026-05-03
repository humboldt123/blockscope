from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
import subprocess
from datetime import datetime
import asyncio
import os

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path("C:/Users/venus/OneDrive/Desktop/BLOCKSCOPE_DATA")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Verify write permissions
try:
    test_file = DATA_DIR / ".write_test"
    test_file.write_text("test")
    test_file.unlink()
    logger.info(f"Data directory initialized: {DATA_DIR}")
    logger.info("Write permissions verified")
except Exception as e:
    logger.error(f"Cannot write to data directory: {e}")
    raise

# Session tracking for auto-finalization
session_last_activity = {}
FINALIZE_TIMEOUT_SECONDS = 15

@app.get("/")
async def health_check():
    return {"status": "ok", "data_dir": str(DATA_DIR)}

@app.post("/init-session")
async def init_session(session_id: str = Query(...)):
    """Initialize a new recording session"""
    session_dir = DATA_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Create segments subdirectory for video chunks
    segments_dir = session_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    session_last_activity[session_id] = datetime.now()

    logger.info(f"Initialized session: {session_id}")
    return {"status": "ok", "session_dir": str(session_dir)}

@app.post("/upload-metadata")
async def upload_metadata(request: Request, session_id: str = Query(...)):
    """Receive metadata.json"""
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        session_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = session_dir / "metadata.json"
    body = await request.body()

    with open(metadata_file, "wb") as f:
        f.write(body)

    logger.info(f"Saved metadata for session {session_id}")
    return {"status": "ok"}


@app.post("/upload-segment")
async def upload_segment(request: Request, session_id: str = Query(...), segment_index: int = Query(...)):
    """Receive video segment (MPEG-TS format)"""
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    segments_dir = session_dir / "segments"
    segment_file = segments_dir / f"segment_{segment_index:04d}.ts"

    # Write segment file
    chunk_size = 1024 * 1024  # 1MB chunks
    with open(segment_file, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)

    file_size = segment_file.stat().st_size
    logger.info(f"Received segment {segment_index} for session {session_id}: {file_size} bytes")

    # Update activity timestamp and schedule auto-finalization
    session_last_activity[session_id] = datetime.now()
    asyncio.create_task(auto_finalize_if_idle(session_id))

    return {"status": "ok", "segment_index": segment_index, "size": file_size}

@app.post("/stream-data")
async def stream_data(request: Request, session_id: str = Query(...), data_type: str = Query(...)):
    """Stream ticks.jsonl, inputs.jsonl, block_changes.jsonl, or frame_mapping.jsonl data"""
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    if data_type not in ["ticks", "inputs", "block_changes", "frame_mapping"]:
        return JSONResponse(status_code=400, content={"error": "Invalid data_type"})

    file_path = session_dir / f"{data_type}.jsonl"

    # Append streamed data to file
    bytes_written = 0
    with open(file_path, "ab") as f:
        async for chunk in request.stream():
            f.write(chunk)
            bytes_written += len(chunk)

    logger.info(f"Streamed {bytes_written} bytes to {data_type}.jsonl for session {session_id}")

    # Update activity timestamp
    session_last_activity[session_id] = datetime.now()

    return {"status": "ok", "bytes_written": bytes_written}

@app.post("/upload-replay")
async def upload_replay(request: Request, session_id: str = Query(...), filename: str = Query(...)):
    """Receive a ReplayMod .mcpr file for the session"""
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        session_dir.mkdir(parents=True, exist_ok=True)

    # Strip any path components from the filename for safety
    safe_name = Path(filename).name
    replay_file = session_dir / safe_name

    bytes_written = 0
    with open(replay_file, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            bytes_written += len(chunk)

    logger.info(f"Saved replay {safe_name} for session {session_id}: {bytes_written} bytes")
    return {"status": "ok", "filename": safe_name, "bytes_written": bytes_written}

@app.post("/finalize")
@app.post("/finalize-video")
async def finalize_video(session_id: str = Query(...)):
    """Concatenate MPEG-TS segments into final MP4"""
    return await finalize_video_internal(session_id)

async def finalize_video_internal(session_id: str):
    """Internal finalization logic"""
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    segments_dir = session_dir / "segments"
    if not segments_dir.exists():
        logger.warning(f"No segments directory for session {session_id}")
        return {"status": "ok", "message": "No video data to finalize"}

    # Get sorted list of segment files, filter out empty ones
    all_segment_files = sorted(segments_dir.glob("segment_*.ts"))
    segment_files = [f for f in all_segment_files if f.stat().st_size > 0]

    if not segment_files:
        logger.warning(f"No valid video segments for session {session_id}")
        return {"status": "ok", "message": "No video segments found"}

    logger.info(f"Found {len(segment_files)} valid segments (filtered {len(all_segment_files) - len(segment_files)} empty)")

    # Create concat file list for ffmpeg
    concat_list = session_dir / "concat_list.txt"
    with open(concat_list, "w") as f:
        for segment_file in segment_files:
            # FFmpeg concat requires relative or absolute paths
            f.write(f"file '{segment_file.absolute()}'\n")

    output_file = session_dir / "video.mp4"

    # Run ffmpeg concatenation
    ffmpeg_cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",  # Copy streams without re-encoding
        "-y",
        str(output_file)
    ]

    logger.info(f"Concatenating {len(segment_files)} segments for session {session_id}")

    try:
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            output_size = output_file.stat().st_size
            logger.info(f"Finalized video for session {session_id}: {output_size} bytes")

            # Clean up ALL segments (including empty ones) and concat list
            for segment_file in all_segment_files:
                segment_file.unlink()
            concat_list.unlink()
            segments_dir.rmdir()

            return {"status": "ok", "video_size": output_size, "segments_concatenated": len(segment_files)}
        else:
            logger.error(f"FFmpeg failed: {result.stderr}")
            return JSONResponse(status_code=500, content={"error": "FFmpeg concatenation failed", "stderr": result.stderr})

    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timeout for session {session_id}")
        return JSONResponse(status_code=500, content={"error": "FFmpeg timeout"})
    except Exception as e:
        logger.error(f"Finalization error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

async def auto_finalize_if_idle(session_id: str):
    """Auto-finalize video if no new segments received within timeout"""
    await asyncio.sleep(FINALIZE_TIMEOUT_SECONDS)

    if session_id not in session_last_activity:
        return

    last_activity = session_last_activity[session_id]
    elapsed = (datetime.now() - last_activity).total_seconds()

    if elapsed >= FINALIZE_TIMEOUT_SECONDS:
        logger.info(f"Auto-finalizing session {session_id} after {elapsed:.1f}s idle")
        await finalize_video_internal(session_id)
        del session_last_activity[session_id]

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Blockscope Server...")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info("Listening on: http://0.0.0.0:9000")
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
