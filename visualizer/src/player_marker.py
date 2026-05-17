"""
Renders a small colored cube at the recorded player's position.
"""

import numpy as np

from settings import *


# Unit cube vertices (36 vertices for 12 triangles)
def _make_cube_vertices():
    """Generate a unit cube centered at origin, size 0.6 blocks."""
    s = 0.3  # half-size
    # 8 corners
    v = [
        (-s, 0.0, -s), (s, 0.0, -s), (s, 1.8, -s), (-s, 1.8, -s),  # back
        (-s, 0.0,  s), (s, 0.0,  s), (s, 1.8,  s), (-s, 1.8,  s),  # front
    ]
    # 12 triangles (6 faces x 2 tris)
    faces = [
        # back
        (0, 2, 1), (0, 3, 2),
        # front
        (4, 5, 6), (4, 6, 7),
        # left
        (0, 4, 7), (0, 7, 3),
        # right
        (1, 2, 6), (1, 6, 5),
        # bottom
        (0, 1, 5), (0, 5, 4),
        # top
        (3, 7, 6), (3, 6, 2),
    ]
    verts = []
    for tri in faces:
        for i in tri:
            verts.extend(v[i])
    return np.array(verts, dtype='f4')


class PlayerMarker:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx
        self.program = app.shader_program.marker

        vertex_data = _make_cube_vertices()
        self.vbo = self.ctx.buffer(vertex_data.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f', 'in_position')],
            skip_errors=True
        )

        self.position = glm.vec3(0, 0, 0)

    def set_position(self, x, y, z):
        self.position = glm.vec3(x, y, z)

    def render(self):
        m_model = glm.translate(glm.mat4(), self.position)
        self.program['m_model'].write(m_model)
        self.vao.render()
