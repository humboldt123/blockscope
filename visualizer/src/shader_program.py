from settings import *


class ShaderProgram:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx
        self.player = app.player
        # shaders
        self.chunk = self.get_program('chunk')
        self.marker = self.get_program('marker')
        self.set_uniforms_on_init()

    def set_uniforms_on_init(self):
        # chunk
        self.chunk['m_proj'].write(self.player.m_proj)
        self.chunk['u_texture_array_0'] = 1
        self.chunk['bg_color'].write(BG_COLOR)

        # marker
        self.marker['m_proj'].write(self.player.m_proj)
        self.marker['m_model'].write(glm.mat4())
        self.marker['u_color'].write(glm.vec3(1.0, 0.2, 0.2))

    def update(self):
        self.chunk['m_view'].write(self.player.m_view)
        self.marker['m_view'].write(self.player.m_view)

    def get_program(self, shader_name):
        with open(f'{ROOT_DIR}/shaders/{shader_name}.vert') as file:
            vertex_shader = file.read()
        with open(f'{ROOT_DIR}/shaders/{shader_name}.frag') as file:
            fragment_shader = file.read()
        program = self.ctx.program(vertex_shader=vertex_shader,
                                   fragment_shader=fragment_shader)
        return program
