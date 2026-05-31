Remember, not every tick has a valid frame because of how frames are recorded:
tick 1  → frame 0,1     (two video frames for the same tick — client ran faster than 20fps)
tick 11 → frame 2       (ticks 2-10 = 9 ticks with NO video frames = loading screen)
tick 21 → frame 3       (ticks 12-20 = same — still loading)
tick 28 → frame 9
tick 29 → frame 10
...
tick 43 → frame 24
tick 45 → frame 25      (tick 44 skipped — one frame dropped during gameplay)
For tick 43 specifically: raw tick 44 is simply absent from the file. The game briefly didn't render a frame at that tick (common during world state updates).

Implication for the ML model:

This is expected behavior. The fix is simple — filter at training time:


# Only use ticks that have a corresponding video frame
valid_ticks = [t for t in range(tick_count) if frame_mapping.get(t) is not None]

