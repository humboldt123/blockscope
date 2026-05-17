import moderngl as mgl

from block_registry import BlockRegistry, TEX_SIZE


class Textures:
    def __init__(self, app, block_registry: BlockRegistry):
        self.app = app
        self.ctx = app.ctx

        # Build texture array from block registry
        num_layers, data = block_registry.build_texture_array()
        print(f'[Textures] Building texture array: {num_layers} layers of {TEX_SIZE}x{TEX_SIZE}')

        self.texture_array_0 = self.ctx.texture_array(
            size=(TEX_SIZE, TEX_SIZE, num_layers),
            components=4,
            data=data
        )
        self.texture_array_0.anisotropy = 32.0
        self.texture_array_0.build_mipmaps()
        self.texture_array_0.filter = (mgl.NEAREST, mgl.NEAREST)

        # Assign to texture unit 1
        self.texture_array_0.use(location=1)
