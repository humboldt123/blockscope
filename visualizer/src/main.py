import sys

import imgui
import moderngl as mgl
import pygame as pg
from imgui.integrations.opengl import ProgrammablePipelineRenderer

from block_registry import BlockRegistry
from inventory_renderer import InventoryRenderer
from player import Player
from replay_loader import ReplayLoader
from scene import Scene
from settings import *
from shader_program import ShaderProgram
from textures import Textures
from video_player import VideoPlayer
from world import ReplayWorld


class ImGuiPygameRenderer(ProgrammablePipelineRenderer):
    """imgui renderer for pygame + OpenGL 3.3 core profile."""

    def __init__(self):
        super().__init__()
        self._gui_time = None
        self._key_map = {}
        self._setup_key_map()

    def _custom_key(self, key):
        if key not in self._key_map:
            self._key_map[key] = len(self._key_map)
        return self._key_map[key]

    def _setup_key_map(self):
        key_map = self.io.key_map
        key_map[imgui.KEY_TAB] = self._custom_key(pg.K_TAB)
        key_map[imgui.KEY_LEFT_ARROW] = self._custom_key(pg.K_LEFT)
        key_map[imgui.KEY_RIGHT_ARROW] = self._custom_key(pg.K_RIGHT)
        key_map[imgui.KEY_UP_ARROW] = self._custom_key(pg.K_UP)
        key_map[imgui.KEY_DOWN_ARROW] = self._custom_key(pg.K_DOWN)
        key_map[imgui.KEY_PAGE_UP] = self._custom_key(pg.K_PAGEUP)
        key_map[imgui.KEY_PAGE_DOWN] = self._custom_key(pg.K_PAGEDOWN)
        key_map[imgui.KEY_HOME] = self._custom_key(pg.K_HOME)
        key_map[imgui.KEY_END] = self._custom_key(pg.K_END)
        key_map[imgui.KEY_DELETE] = self._custom_key(pg.K_DELETE)
        key_map[imgui.KEY_BACKSPACE] = self._custom_key(pg.K_BACKSPACE)
        key_map[imgui.KEY_SPACE] = self._custom_key(pg.K_SPACE)
        key_map[imgui.KEY_ENTER] = self._custom_key(pg.K_RETURN)
        key_map[imgui.KEY_ESCAPE] = self._custom_key(pg.K_ESCAPE)
        key_map[imgui.KEY_A] = self._custom_key(pg.K_a)
        key_map[imgui.KEY_C] = self._custom_key(pg.K_c)
        key_map[imgui.KEY_V] = self._custom_key(pg.K_v)
        key_map[imgui.KEY_X] = self._custom_key(pg.K_x)
        key_map[imgui.KEY_Y] = self._custom_key(pg.K_y)
        key_map[imgui.KEY_Z] = self._custom_key(pg.K_z)

    def process_event(self, event):
        io = self.io

        if event.type == pg.MOUSEMOTION:
            io.mouse_pos = event.pos
            return True

        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1: io.mouse_down[0] = 1
            if event.button == 2: io.mouse_down[1] = 1
            if event.button == 3: io.mouse_down[2] = 1
            return True

        if event.type == pg.MOUSEBUTTONUP:
            if event.button == 1: io.mouse_down[0] = 0
            if event.button == 2: io.mouse_down[1] = 0
            if event.button == 3: io.mouse_down[2] = 0
            if event.button == 4: io.mouse_wheel = 0.5
            if event.button == 5: io.mouse_wheel = -0.5
            return True

        if event.type == pg.KEYDOWN:
            for char in event.unicode:
                code = ord(char)
                if 0 < code < 0x10000:
                    io.add_input_character(code)
            io.keys_down[self._custom_key(event.key)] = True

        if event.type == pg.KEYUP:
            io.keys_down[self._custom_key(event.key)] = False

        if event.type in (pg.KEYDOWN, pg.KEYUP):
            io.key_ctrl = (io.keys_down[self._custom_key(pg.K_LCTRL)] or
                           io.keys_down[self._custom_key(pg.K_RCTRL)])
            io.key_alt = (io.keys_down[self._custom_key(pg.K_LALT)] or
                          io.keys_down[self._custom_key(pg.K_RALT)])
            io.key_shift = (io.keys_down[self._custom_key(pg.K_LSHIFT)] or
                            io.keys_down[self._custom_key(pg.K_RSHIFT)])
            return True

        return False

    def process_inputs(self):
        io = imgui.get_io()
        current_time = pg.time.get_ticks() / 1000.0
        if self._gui_time:
            io.delta_time = current_time - self._gui_time
        else:
            io.delta_time = 1.0 / 60.0
        if io.delta_time <= 0.0:
            io.delta_time = 1.0 / 1000.0
        self._gui_time = current_time


class VoxelEngine:
    def __init__(self, session_dir=None):
        if session_dir is None:
            session_dir = DEFAULT_SESSION_DIR

        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE, 24)

        pg.display.set_mode(WIN_RES, flags=pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
        self.ctx.gc_mode = 'auto'

        self.clock = pg.time.Clock()
        self.delta_time = 0
        self.time = 0

        # Mouse grab state (toggle with Tab)
        self._mouse_grabbed = True
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)

        self.is_running = True

        # ImGui setup
        imgui.create_context()
        self.imgui_renderer = ImGuiPygameRenderer()
        io = imgui.get_io()
        io.display_size = (int(WIN_RES.x), int(WIN_RES.y))

        # Load replay
        import time
        print(f'[Replay] Loading session from: {session_dir}')
        t0 = time.time()
        self.replay = ReplayLoader(session_dir)
        print(f'[TIME] ReplayLoader: {time.time() - t0:.2f}s')
        print(f'[Replay] {self.replay.max_tick} ticks, '
              f'{len(self.replay.get_all_unique_block_ids())} unique block types')

        # Build block registry and pre-register all textures + variants
        print('[Registry] Building block texture registry...')
        t0 = time.time()
        self.block_registry = BlockRegistry()
        print(f'[TIME] BlockRegistry init: {time.time() - t0:.2f}s')
        initial_state = self.replay.get_player_state(0)
        biome = 'minecraft:plains'
        if initial_state:
            biome = initial_state.get('world', {}).get('biome', 'minecraft:plains')
        t0 = time.time()
        for bid in self.replay.get_all_unique_block_ids():
            self.block_registry.register_block(bid, biome=biome)
        print(f'[TIME] Register blocks: {time.time() - t0:.2f}s')
        # All blocks already registered (no separate variants needed)
        print(f'[Registry] {self.block_registry.num_textures} texture layers, '
              f'{len(self.replay.get_all_unique_block_ids())} unique blocks')

        # Position spectator camera above initial player position, looking down 30deg
        px, py, pz = self.replay.get_initial_player_pos()
        cam_y = py + 20
        cam_z_offset = 15
        self.player = Player(self, position=(px, cam_y, pz + cam_z_offset),
                             yaw=-90, pitch=-30)

        # Build world FIRST so variant textures get registered during tick-0 processing
        t0 = time.time()
        self.replay_world = ReplayWorld(self, self.replay, self.block_registry)
        print(f'[TIME] ReplayWorld init: {time.time() - t0:.2f}s')

        # Now build texture array — includes all variant/UV-cropped textures
        t0 = time.time()
        self.textures = Textures(self, self.block_registry)
        print(f'[TIME] Textures: {time.time() - t0:.2f}s')
        t0 = time.time()
        self.shader_program = ShaderProgram(self)
        print(f'[TIME] ShaderProgram: {time.time() - t0:.2f}s')

        self.scene = Scene(self, self.replay_world)

        # Inventory renderer
        self.inventory_renderer = InventoryRenderer(self.ctx)

        # Video player
        frame_mapping = self.replay.frame_to_tick if self.replay.has_frame_mapping else None
        self.video_player = VideoPlayer(session_dir, self.ctx, self.replay.max_tick,
                                        frame_to_tick=frame_mapping)

        # Playback state
        self.playback_tick = 0
        self.playback_time = 0.0
        self.playing = False  # Start paused so user can see tick 0
        self.playback_speed = 1.0
        self._scrubbing = False

        # Build initial meshes (needs shader_program to be ready)
        t0 = time.time()
        self.replay_world.rebuild_mesh()
        print(f'[TIME] Initial mesh rebuild: {time.time() - t0:.2f}s')

        print(f'[Ready] Blocks loaded at tick 0: {len(self.replay_world.blocks)}')
        print(f'[Controls] WASD=move, Space=up, Shift=down, Ctrl=fast, '
              f'Esc=toggle mouse')

    def _advance_world(self, target_tick):
        """Advance world to target tick and update player marker."""
        self.replay_world.advance_to_tick(target_tick)

        state = self.replay.get_player_state(target_tick)
        if state:
            p = state['player']
            self.scene.player_marker.set_position(p['x'], p['y'], p['z'])

    def _seek_to_tick(self, target_tick):
        """Seek to any tick (forward or backward). Handles reset if needed."""
        target_tick = max(0, min(target_tick, self.replay.max_tick))

        if target_tick < self.replay_world.current_tick:
            self.replay_world.reset()
            self._advance_world(target_tick)
        elif target_tick > self.replay_world.current_tick:
            self._advance_world(target_tick)

        self.playback_tick = target_tick
        self.playback_time = target_tick / TICKS_PER_SECOND
        self.video_player.seek_to_tick(target_tick)

    def update(self):
        self.player.update()
        self.shader_program.update()

        self.delta_time = self.clock.tick()
        self.time = pg.time.get_ticks() * 0.001

        # Advance playback (only when not scrubbing)
        if self.playing and not self._scrubbing and self.playback_tick < self.replay.max_tick:
            self.playback_time += self.delta_time * 0.001 * self.playback_speed
            target_tick = int(self.playback_time * TICKS_PER_SECOND)
            target_tick = min(target_tick, self.replay.max_tick)

            if target_tick > self.playback_tick:
                self._advance_world(target_tick)
                self.playback_tick = target_tick
                self.video_player.seek_to_tick(target_tick)

        pg.display.set_caption(
            f'Blockscope | FPS: {self.clock.get_fps():.0f} | '
            f'Tick: {self.playback_tick}/{self.replay.max_tick} | '
            f'Blocks: {len(self.replay_world.blocks)}'
        )

    def render(self):
        self.ctx.clear(color=BG_COLOR)
        self.scene.render()
        self._render_ui()
        pg.display.flip()

    def _render_ui(self):
        """Render imgui overlay."""
        self.imgui_renderer.process_inputs()
        imgui.new_frame()

        win_w, win_h = int(WIN_RES.x), int(WIN_RES.y)

        # --- Playback controls bar at bottom (centered, half width) ---
        bar_height = 64
        bar_w = win_w // 2
        imgui.set_next_window_position((win_w - bar_w) // 2, win_h - bar_height)
        imgui.set_next_window_size(bar_w, bar_height)

        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                 imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR |
                 imgui.WINDOW_NO_SAVED_SETTINGS)

        imgui.set_next_window_bg_alpha(0.75)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (12, 6))
        imgui.begin("Playback", flags=flags)

        avail_w = imgui.get_content_region_available_width()

        # Row 1: controls
        # Play/Pause button
        label = "Pause" if self.playing else "Play "
        if imgui.button(label, 50, 0):
            self.playing = not self.playing
        imgui.same_line()

        # Restart button
        if imgui.button("<<", 28, 0):
            self._seek_to_tick(0)
            self.playing = True
        imgui.same_line()

        # Speed buttons
        speeds = [0.5, 1.0, 2.0, 4.0]
        for spd in speeds:
            if abs(self.playback_speed - spd) < 0.01:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.26, 0.59, 0.98, 1.0)
                imgui.button(f"{spd}x", 32, 0)
                imgui.pop_style_color()
            else:
                if imgui.button(f"{spd}x", 32, 0):
                    self.playback_speed = spd
            imgui.same_line()

        # Time label
        secs = self.playback_tick / TICKS_PER_SECOND
        total_secs = self.replay.max_tick / TICKS_PER_SECOND
        imgui.text(f"{secs:.1f}s/{total_secs:.1f}s")
        imgui.same_line()

        # Quit button (push to right edge of content area)
        quit_w = 40
        imgui.same_line(avail_w - quit_w + 12)
        if imgui.button("Quit", quit_w, 0):
            self.is_running = False

        # Row 2: scrubber (use available content width)
        imgui.push_item_width(avail_w)
        changed, value = imgui.slider_int(
            "##scrubber", self.playback_tick, 0, self.replay.max_tick,
            format=""
        )
        if imgui.is_item_active():
            self._scrubbing = True
            if changed:
                self._seek_to_tick(value)
        elif self._scrubbing:
            self._scrubbing = False
        imgui.pop_item_width()

        imgui.end()
        imgui.pop_style_var()

        # --- Video overlay (top-right corner) ---
        if self.video_player.available and self.video_player.texture_id:
            vid_w = 320
            vid_h = int(vid_w * self.video_player.height / self.video_player.width)
            margin = 10

            imgui.set_next_window_position(win_w - vid_w - margin, margin,
                                           imgui.ONCE)
            imgui.set_next_window_size(vid_w + 16, vid_h + 36, imgui.ONCE)
            imgui.set_next_window_bg_alpha(0.85)

            expanded, opened = imgui.begin("POV", True,
                                           imgui.WINDOW_NO_SCROLLBAR |
                                           imgui.WINDOW_NO_SAVED_SETTINGS)
            if expanded:
                avail_w = imgui.get_content_region_available_width()
                display_h = int(avail_w * self.video_player.height / self.video_player.width)
                imgui.image(self.video_player.texture_id, avail_w, display_h,
                            uv0=(0, 1), uv1=(1, 0))
            imgui.end()

        # --- Inventory panel (bottom-left corner) ---
        self._render_inventory_panel()

        # Render imgui
        imgui.render()
        self.ctx.disable(mgl.DEPTH_TEST | mgl.CULL_FACE)
        self.imgui_renderer.render(imgui.get_draw_data())
        self.ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE)

    def _render_inventory_panel(self):
        """Render inventory visualization panel with item icons."""
        tick_data = self.replay.get_player_state(self.playback_tick)
        if not tick_data or 'inventory' not in tick_data:
            return

        inventory = tick_data['inventory']
        if not inventory:
            return

        win_w, win_h = int(WIN_RES.x), int(WIN_RES.y)
        margin = 10
        panel_w = 360
        panel_h = 600

        imgui.set_next_window_position(margin, win_h - panel_h - 80, imgui.ONCE)
        imgui.set_next_window_size(panel_w, panel_h, imgui.ONCE)
        imgui.set_next_window_bg_alpha(0.90)

        expanded, opened = imgui.begin("Inventory", True,
                                       imgui.WINDOW_NO_SAVED_SETTINGS)
        if not expanded:
            imgui.end()
            return

        # Render inventory grid
        slots = inventory.get('slots', [])
        selected_slot = inventory.get('selectedSlot', -1)
        self.inventory_renderer.render_inventory_grid(slots, selected_slot)

        # Cursor stack (mouse-held item)
        cursor_stack = inventory.get('cursorStack')
        if cursor_stack:
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.text("Cursor (mouse):")
            self.inventory_renderer.render_slot(
                cursor_stack['id'],
                cursor_stack['count'],
                cursor_stack.get('durability')
            )

        imgui.end()

    def handle_events(self):
        for event in pg.event.get():
            # Let imgui process events
            self.imgui_renderer.process_event(event)

            if event.type == pg.QUIT:
                self.is_running = False
                continue

            io = imgui.get_io()

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self._mouse_grabbed = not self._mouse_grabbed
                    pg.event.set_grab(self._mouse_grabbed)
                    pg.mouse.set_visible(not self._mouse_grabbed)
                elif event.key == pg.K_p:
                    self.playing = not self.playing
                elif event.key == pg.K_EQUALS or event.key == pg.K_PLUS:
                    self.playback_speed = min(self.playback_speed * 2.0, 64.0)
                elif event.key == pg.K_MINUS:
                    self.playback_speed = max(self.playback_speed / 2.0, 0.125)
                elif event.key == pg.K_r:
                    self._seek_to_tick(0)
                    self.playing = True

    def run(self):
        while self.is_running:
            self.handle_events()
            self.update()
            self.render()

        self.video_player.release()
        pg.quit()
        sys.exit()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        session = sys.argv[1]
    else:
        print(f'Usage: python main.py <session_dir>')
        print(f'  e.g.: python main.py ../../run/recordings/session_1771078638')
        print(f'  Falling back to default: {DEFAULT_SESSION_DIR}')
        session = None
    app = VoxelEngine(session)
    app.run()
