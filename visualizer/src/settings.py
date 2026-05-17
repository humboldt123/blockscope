import math
import os

import glm
import numpy as np

# src folder path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# resolution
WIN_RES = glm.vec2(1600, 900)

# camera
ASPECT_RATIO = WIN_RES.x / WIN_RES.y
FOV_DEG = 70
V_FOV = glm.radians(FOV_DEG)
H_FOV = 2 * math.atan(math.tan(V_FOV * 0.5) * ASPECT_RATIO)
NEAR = 0.1
FAR = 2000.0
PITCH_MAX = glm.radians(89)

# spectator
PLAYER_SPEED = 0.05
MOUSE_SENSITIVITY = 0.002

# colors
BG_COLOR = glm.vec3(0.58, 0.83, 0.99)

# default recording path (can be overridden via command line)
DEFAULT_SESSION_DIR = os.path.normpath(
    os.path.join(ROOT_DIR, '..', '..', 'run', 'recordings', 'session_1771078638')
)

# playback
TICKS_PER_SECOND = 20  # Minecraft default TPS
