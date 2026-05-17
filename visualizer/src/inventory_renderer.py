"""
Render Minecraft inventory as a grid with item icons.
"""

import numpy as np
import imgui
from PIL import Image
from item_icons import get_item_icon


class InventoryRenderer:
    def __init__(self, ctx):
        """
        Args:
            ctx: moderngl context for creating textures
        """
        self.ctx = ctx
        self._texture_cache = {}  # item_id -> (texture_glo, size)
        self._texture_objects = {}  # item_id -> texture object (keep alive)
        self.slot_size = 32  # pixels per slot

    def _get_item_texture(self, item_id: str):
        """Get or create OpenGL texture for an item icon."""
        if item_id in self._texture_cache:
            return self._texture_cache[item_id]

        try:
            # Render icon
            icon = get_item_icon(item_id, size=self.slot_size)
            if icon is None:
                self._texture_cache[item_id] = None
                return None

            # Convert PIL to numpy and upload to OpenGL
            arr = np.array(icon, dtype=np.uint8)
            # Flip vertically for OpenGL
            arr = np.flipud(arr)

            tex = self.ctx.texture((self.slot_size, self.slot_size), 4, arr.tobytes())
            tex.filter = (self.ctx.NEAREST, self.ctx.NEAREST)

            # Keep texture object alive
            self._texture_objects[item_id] = tex
            self._texture_cache[item_id] = (tex.glo, self.slot_size)
            return (tex.glo, self.slot_size)
        except Exception as e:
            print(f"[InventoryRenderer] Failed to create texture for {item_id}: {e}")
            self._texture_cache[item_id] = None
            return None

    def render_slot(self, item_id: str, count: int, durability: float | None = None):
        """
        Render a single inventory slot with item icon and count.
        Args:
            item_id: Item ID like 'minecraft:diamond_sword'
            count: Stack size
            durability: Durability 0-1, or None if not damageable
        """
        tex_info = self._get_item_texture(item_id)

        if tex_info:
            tex_glo, size = tex_info
            # Draw item icon (flip vertically with uv0/uv1)
            imgui.image(tex_glo, size, size, uv0=(0, 1), uv1=(1, 0))
        else:
            # Fallback: colored square
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()
            draw_list.add_rect_filled(
                pos[0], pos[1],
                pos[0] + self.slot_size, pos[1] + self.slot_size,
                imgui.get_color_u32_rgba(0.3, 0.3, 0.3, 1.0)
            )
            imgui.dummy(self.slot_size, self.slot_size)

        # Draw count badge (bottom-right)
        if count > 1:
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_item_rect_max()
            text = str(count)
            text_size = imgui.calc_text_size(text)
            # Shadow
            draw_list.add_text(
                pos[0] - text_size.x - 2, pos[1] - text_size.y - 2,
                imgui.get_color_u32_rgba(0, 0, 0, 1), text
            )
            # Text
            draw_list.add_text(
                pos[0] - text_size.x - 3, pos[1] - text_size.y - 3,
                imgui.get_color_u32_rgba(1, 1, 1, 1), text
            )

        # Draw durability bar if applicable
        if durability is not None:
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_item_rect_min()
            bar_width = int(self.slot_size * durability)
            bar_height = 2
            y = pos[1] + self.slot_size - bar_height - 2

            # Background
            draw_list.add_rect_filled(
                pos[0] + 2, y,
                pos[0] + self.slot_size - 2, y + bar_height,
                imgui.get_color_u32_rgba(0, 0, 0, 0.8)
            )

            # Durability bar (color based on percentage)
            if durability > 0.5:
                color = imgui.get_color_u32_rgba(0.2, 1.0, 0.2, 1.0)
            elif durability > 0.2:
                color = imgui.get_color_u32_rgba(1.0, 0.8, 0.0, 1.0)
            else:
                color = imgui.get_color_u32_rgba(1.0, 0.2, 0.2, 1.0)

            draw_list.add_rect_filled(
                pos[0] + 2, y,
                pos[0] + 2 + bar_width, y + bar_height,
                color
            )

    def render_inventory_grid(self, slots: list[dict], selected_slot: int = -1):
        """
        Render inventory as a grid (like Minecraft's inventory screen).
        Args:
            slots: List of slot dicts with 'slot', 'item', etc.
            selected_slot: Currently selected hotbar slot (0-8)
        """
        # Create slot lookup
        slot_map = {s['slot']: s for s in slots}

        # Hotbar (0-8) - horizontal row
        imgui.text("Hotbar")
        for i in range(9):
            if i > 0:
                imgui.same_line()

            slot_data = slot_map.get(i)
            if slot_data:
                item = slot_data['item']
                self.render_slot(
                    item['id'],
                    item['count'],
                    item.get('durability')
                )
            else:
                # Empty slot
                imgui.dummy(self.slot_size, self.slot_size)

            # Highlight selected slot
            if i == selected_slot:
                draw_list = imgui.get_window_draw_list()
                pos_min = imgui.get_item_rect_min()
                pos_max = imgui.get_item_rect_max()
                draw_list.add_rect(
                    pos_min[0] - 2, pos_min[1] - 2,
                    pos_max[0] + 2, pos_max[1] + 2,
                    imgui.get_color_u32_rgba(1, 1, 1, 1),
                    thickness=2
                )

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Main inventory (9-35) - 3 rows of 9
        imgui.text("Inventory")
        for row in range(3):
            for col in range(9):
                slot_idx = 9 + row * 9 + col
                if col > 0:
                    imgui.same_line()

                slot_data = slot_map.get(slot_idx)
                if slot_data:
                    item = slot_data['item']
                    self.render_slot(
                        item['id'],
                        item['count'],
                        item.get('durability')
                    )
                else:
                    imgui.dummy(self.slot_size, self.slot_size)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Armor + Offhand (36-40) - vertical column
        imgui.text("Equipment")
        armor_names = {39: "Helmet", 38: "Chestplate", 37: "Leggings", 36: "Boots", 40: "Offhand"}
        for slot_idx in [39, 38, 37, 36, 40]:
            if slot_idx == 40:
                imgui.spacing()
                imgui.text("Offhand")

            slot_data = slot_map.get(slot_idx)
            if slot_data:
                item = slot_data['item']
                self.render_slot(
                    item['id'],
                    item['count'],
                    item.get('durability')
                )
            else:
                imgui.dummy(self.slot_size, self.slot_size)

            imgui.same_line()
            imgui.text(armor_names[slot_idx])

    def release(self):
        """Release all OpenGL textures."""
        for tex_glo, _ in self._texture_cache.values():
            # Note: texture objects are managed by moderngl context
            pass
        self._texture_cache.clear()
