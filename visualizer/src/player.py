import pygame as pg

from camera import Camera
from settings import *


class Player(Camera):
    def __init__(self, app, position=(0, 80, 0), yaw=-90, pitch=-30):
        self.app = app
        super().__init__(position, yaw, pitch)

    def update(self):
        self.keyboard_control()
        self.mouse_control()
        super().update()

    def mouse_control(self):
        mouse_dx, mouse_dy = pg.mouse.get_rel()
        # Only rotate camera when mouse is grabbed (like Minecraft)
        if not self.app._mouse_grabbed:
            return
        if mouse_dx:
            self.rotate_yaw(delta_x=mouse_dx * MOUSE_SENSITIVITY)
        if mouse_dy:
            self.rotate_pitch(delta_y=mouse_dy * MOUSE_SENSITIVITY)

    def keyboard_control(self):
        # Don't move when mouse is ungrabbed (UI mode)
        if not self.app._mouse_grabbed:
            return

        key_state = pg.key.get_pressed()
        vel = PLAYER_SPEED * self.app.delta_time

        # Ctrl for speed boost
        if key_state[pg.K_LCTRL]:
            vel *= 5.0

        if key_state[pg.K_w]:
            self.move_forward(vel)
        if key_state[pg.K_s]:
            self.move_back(vel)
        if key_state[pg.K_d]:
            self.move_right(vel)
        if key_state[pg.K_a]:
            self.move_left(vel)
        if key_state[pg.K_SPACE]:
            self.move_up(vel)
        if key_state[pg.K_LSHIFT]:
            self.move_down(vel)
