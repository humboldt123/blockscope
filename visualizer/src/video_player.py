"""
Video player that decodes frames from the session video.mp4 and uploads them
as OpenGL textures for display in the imgui overlay.

Uses PyAV (FFmpeg) for decoding. Tick-to-frame mapping comes from either:
  - frame_mapping.jsonl (exact, per-frame tick recorded by the mod)
  - Linear interpolation fallback (for old recordings without the mapping)

Two decode modes:
  - Sequential: during normal playback, just decode the next frame (fast).
  - Seek: when scrubbing or jumping, seek to nearest keyframe then decode
    forward to the exact target frame.
"""

import bisect
import os

import av
import numpy as np


class VideoPlayer:
    def __init__(self, session_dir: str, ctx, max_tick: int,
                 frame_to_tick: list[int] | None = None):
        """
        Args:
            session_dir: Path to the recording session directory.
            ctx: moderngl context.
            max_tick: Maximum tick number in the replay.
            frame_to_tick: List where index=frame, value=tick. From frame_mapping.jsonl.
                           If None, falls back to linear interpolation.
        """
        self.ctx = ctx
        self._current_frame = -1
        self._texture = None
        self._max_tick = max(max_tick, 1)

        # Build tick-to-frame lookup
        self._frame_to_tick = frame_to_tick  # frame_index -> tick
        self._tick_to_frame_map: dict[int, int] | None = None
        if frame_to_tick:
            # Build reverse mapping: for each tick, the frame to show is the
            # last frame captured at or before that tick.
            # We store sorted tick list for binary search.
            self._mapping_ticks = frame_to_tick  # already sorted by frame order
            print(f'[Video] Frame mapping loaded: {len(frame_to_tick)} entries')
        else:
            self._mapping_ticks = None

        video_path = os.path.join(session_dir, 'video.mp4')
        self._available = os.path.exists(video_path)
        if not self._available:
            print(f'[Video] No video.mp4 found in {session_dir}')
            return

        self._container = av.open(video_path)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = 'AUTO'

        self.width = self._stream.codec_context.width
        self.height = self._stream.codec_context.height
        self.num_frames = self._stream.frames or 0
        self.fps = float(self._stream.average_rate or 20)
        self._time_base = float(self._stream.time_base)

        if self.num_frames <= 0:
            self.num_frames = self._count_frames()

        self._texture = self.ctx.texture((self.width, self.height), 3)
        self._texture.filter = (self.ctx.LINEAR, self.ctx.LINEAR)

        self._decoder = None
        self._reset_decoder()

        mode = "exact mapping" if self._mapping_ticks else "linear interpolation"
        print(f'[Video] Loaded {self.width}x{self.height} @ {self.fps:.0f}fps, '
              f'{self.num_frames} frames for {self._max_tick} ticks ({mode})')

    def _count_frames(self):
        count = 0
        self._container.seek(0, stream=self._stream)
        for _ in self._container.decode(video=0):
            count += 1
        self._container.seek(0, stream=self._stream)
        return count

    def _tick_to_frame(self, tick: int) -> int:
        """Map a replay tick to a video frame index."""
        if self._mapping_ticks:
            # Binary search: find the last frame whose tick <= target tick
            # _mapping_ticks[i] = the tick at which frame i was captured
            idx = bisect.bisect_right(self._mapping_ticks, tick) - 1
            return max(0, min(idx, self.num_frames - 1))

        # Fallback: linear interpolation
        if self.num_frames <= 0:
            return 0
        frame = int(tick * self.num_frames / self._max_tick)
        return max(0, min(frame, self.num_frames - 1))

    def _reset_decoder(self):
        self._container.seek(0, stream=self._stream)
        self._decoder = self._container.decode(video=0)
        self._current_frame = -1

    def _upload_frame(self, frame):
        rgb = frame.to_ndarray(format='rgb24')
        rgb = np.ascontiguousarray(rgb[::-1])
        self._texture.write(rgb.tobytes())

    def _frame_index(self, frame):
        if frame.pts is not None:
            return int(round(frame.pts * self._time_base * self.fps))
        return -1

    @property
    def available(self):
        return self._available

    @property
    def texture_id(self):
        if self._texture:
            return self._texture.glo
        return 0

    def seek_to_tick(self, tick: int):
        if not self._available:
            return

        target = self._tick_to_frame(max(0, tick))
        if target == self._current_frame:
            return

        # Sequential: if target is just 1-3 frames ahead, decode forward
        if 0 < (target - self._current_frame) <= 3 and self._decoder is not None:
            try:
                frame = None
                while self._current_frame < target:
                    frame = next(self._decoder)
                    self._current_frame = self._frame_index(frame)
                    if self._current_frame < 0:
                        self._current_frame = target
                if frame is not None:
                    self._upload_frame(frame)
                return
            except StopIteration:
                self._current_frame = target
                return

        self._seek_exact(target)

    def _seek_exact(self, target_frame: int):
        target_pts = int(target_frame / self.fps / self._time_base)

        try:
            self._container.seek(max(0, target_pts), stream=self._stream)
            self._decoder = self._container.decode(video=0)

            last_frame = None
            for frame in self._decoder:
                idx = self._frame_index(frame)
                last_frame = frame
                if idx >= target_frame:
                    break

            if last_frame is not None:
                self._upload_frame(last_frame)
            self._current_frame = target_frame

        except Exception as e:
            print(f'[Video] Seek error at frame {target_frame}: {e}')
            self._current_frame = target_frame

    def release(self):
        if self._texture:
            self._texture.release()
            self._texture = None
        if self._available:
            self._container.close()
            self._available = False
