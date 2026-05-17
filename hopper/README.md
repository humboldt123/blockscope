# hopper

FastAPI server that receives recording session uploads from N Blockscope mod users and stores them for processing.

Runs on Brev (`vvm33`). Data is stored in `/data/BLOCKSCOPE_DATA/`.

## Local dev

```bash
pip install -r requirements.txt
./start.sh        # starts on http://0.0.0.0:9000
```

## Brev deployment

```bash
# From local machine — copy to Brev instance
scp -r hopper/ vvm33:/data/blockscope-hopper/

ssh vvm33
cd /data/blockscope-hopper
chmod +x start.sh deploy-to-brev.sh
./start.sh
```

Keep it running with screen or tmux:
```bash
screen -S blockscope
./start.sh
# Ctrl+A D to detach, screen -r blockscope to reattach
```

Or as a systemd service:
```ini
[Unit]
Description=Blockscope Upload Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/data/blockscope-hopper
ExecStart=/data/blockscope-hopper/venv/bin/python /data/blockscope-hopper/app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload a session (multipart: `session_id`, `ticks.jsonl`, `inputs.jsonl`, `video.mp4`) |
| `GET`  | `/sessions` | List all sessions |
| `GET`  | `/sessions/{id}` | Session detail + file sizes |

## Data layout

```
/data/BLOCKSCOPE_DATA/
└── session_<timestamp>/
    ├── metadata.json
    ├── ticks.jsonl
    ├── inputs.jsonl
    ├── frame_mapping.jsonl
    └── video.mp4
```
