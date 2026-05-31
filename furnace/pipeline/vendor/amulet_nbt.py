"""
amulet_nbt shim — replaces the amulet-nbt PyPI package.

Minecraft-Model-Reader (Amulet MC, MIT) uses amulet_nbt.TAG_String as a typed
string wrapper for block property values. This is the only type we need at
runtime; the others (TAG_Byte, TAG_Short, TAG_Int, TAG_Long) appear only as
type hints in block.py and are never instantiated by the Java baker path.

Original library: https://github.com/Amulet-Team/amulet-nbt (MIT licence)
"""


class TAG_String:
    """Typed string tag — wraps a plain str, exposes .py_data for comparison."""

    __slots__ = ("py_data",)

    def __init__(self, value: str = "") -> None:
        self.py_data = str(value)

    def to_snbt(self) -> str:
        escaped = self.py_data.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TAG_String):
            return self.py_data == other.py_data
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.py_data)

    def __repr__(self) -> str:
        return f"TAG_String({self.py_data!r})"

    def __str__(self) -> str:
        return self.py_data


# Stubs for the other integer tag types — present only as type hints in block.py,
# never instantiated in the Java baker path.
class _IntTag:
    def __init__(self, value: int = 0) -> None:
        self.py_data = int(value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _IntTag):
            return self.py_data == other.py_data
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.py_data)


TAG_Byte  = _IntTag
TAG_Short = _IntTag
TAG_Int   = _IntTag
TAG_Long  = _IntTag

# amulet_nbt.__major__ is checked by block.py to decide which read_snbt to use
__major__ = 2


def from_snbt(snbt: str):  # used by Block.from_snbt_blockstate, not called in baker
    raise NotImplementedError("from_snbt not needed in pipeline")
