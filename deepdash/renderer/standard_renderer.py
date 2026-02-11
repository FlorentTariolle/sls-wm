import numpy as np
import pygame

from deepdash.core.constants import (
    TILE_SIZE, INTERNAL_WIDTH, INTERNAL_HEIGHT, OUTPUT_SIZE,
    TileType, COLOR_SKY, COLOR_GROUND, COLOR_SPIKE, COLOR_PLAYER, COLOR_PLATFORM,
)
from deepdash.core.game_state import GameState
from deepdash.renderer.base_renderer import BaseRenderer


class StandardRenderer(BaseRenderer):
    def __init__(self):
        if not pygame.get_init():
            pygame.init()
        self.surface = pygame.Surface((INTERNAL_WIDTH, INTERNAL_HEIGHT))

    def render_frame(self, game_state: GameState) -> np.ndarray:
        surf = self.surface
        surf.fill(COLOR_SKY)

        cam_x = game_state.camera_x
        level = game_state.level
        player = game_state.player

        # Determine visible tile columns
        col_start = max(0, int(cam_x // TILE_SIZE))
        col_end = min(level.cols, col_start + INTERNAL_WIDTH // TILE_SIZE + 2)

        for col in range(col_start, col_end):
            for row in range(level.rows):
                tile = level.get_tile(col, row)
                if tile == TileType.EMPTY:
                    continue

                sx = col * TILE_SIZE - int(cam_x)
                sy = row * TILE_SIZE

                if tile == TileType.GROUND:
                    pygame.draw.rect(surf, COLOR_GROUND, (sx, sy, TILE_SIZE, TILE_SIZE))
                elif tile == TileType.PLATFORM:
                    pygame.draw.rect(surf, COLOR_PLATFORM, (sx, sy, TILE_SIZE, TILE_SIZE))
                elif tile == TileType.SPIKE:
                    # Triangle pointing up
                    points = [
                        (sx + TILE_SIZE // 2, sy),
                        (sx, sy + TILE_SIZE),
                        (sx + TILE_SIZE, sy + TILE_SIZE),
                    ]
                    pygame.draw.polygon(surf, COLOR_SPIKE, points)

        # Draw player
        px = int(player.x - cam_x)
        py = int(player.y)
        pygame.draw.rect(surf, COLOR_PLAYER, (px, py, player.width, player.height))

        # Downscale to 64x64
        small = pygame.transform.smoothscale(surf, (OUTPUT_SIZE, OUTPUT_SIZE))
        # Convert to numpy array (H, W, 3)
        arr = pygame.surfarray.array3d(small)
        # pygame gives (W, H, 3), transpose to (H, W, 3)
        arr = np.transpose(arr, (1, 0, 2))
        return arr.astype(np.uint8)
