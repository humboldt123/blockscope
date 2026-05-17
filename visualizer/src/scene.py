import moderngl as mgl

from world import ReplayWorld
from player_marker import PlayerMarker


class Scene:
    def __init__(self, app, replay_world: ReplayWorld):
        self.app = app
        self.world = replay_world
        self.player_marker = PlayerMarker(app)

    def render(self):
        # Disable face culling for world - cross plants are double-sided
        # and leaf blocks need back faces visible through transparency
        self.app.ctx.disable(mgl.CULL_FACE)
        self.world.render()
        self.player_marker.render()
        self.app.ctx.enable(mgl.CULL_FACE)
