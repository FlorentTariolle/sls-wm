from enum import IntEnum


class TileType(IntEnum):
    EMPTY = 0
    GROUND = 1
    SPIKE = 2
    PLATFORM = 3


# Physics
GRAVITY = 0.8
JUMP_VELOCITY = -10.0
PLAYER_SPEED = 4.0

# Tile / viewport
TILE_SIZE = 16
VIEWPORT_TILES_X = 16
VIEWPORT_TILES_Y = 12
INTERNAL_WIDTH = VIEWPORT_TILES_X * TILE_SIZE   # 256
INTERNAL_HEIGHT = VIEWPORT_TILES_Y * TILE_SIZE  # 192
OUTPUT_SIZE = 64  # 64x64 for VAE input

# Player
PLAYER_WIDTH = 14
PLAYER_HEIGHT = 14

# Colors (R, G, B)
COLOR_SKY = (135, 206, 235)
COLOR_GROUND = (139, 90, 43)
COLOR_SPIKE = (255, 50, 50)
COLOR_PLAYER = (0, 255, 100)
COLOR_PLATFORM = (100, 100, 100)

# Semantic colors
SEM_SKY = (0, 0, 0)
SEM_GROUND = (255, 255, 255)
SEM_SPIKE = (255, 0, 0)
SEM_PLAYER = (0, 255, 0)
SEM_PLATFORM = (255, 255, 255)

# Level generation
DEFAULT_LEVEL_LENGTH = 200  # columns
SAFE_ZONE_START = 10  # safe columns at start
SAFE_ZONE_END = 5    # safe columns at end
GROUND_ROW = 10      # ground is at row 10 (0-indexed from top)
