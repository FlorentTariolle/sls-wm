"""Human-playable mode for DeepDash.

Run with: python -m deepdash.play [--difficulty 0.5] [--seed 42] [--length 200]
"""
import argparse
import sys

import pygame

from deepdash.core.constants import INTERNAL_WIDTH, INTERNAL_HEIGHT, TILE_SIZE, TileType
from deepdash.core.constants import (
    COLOR_SKY, COLOR_GROUND, COLOR_SPIKE, COLOR_PLAYER, COLOR_PLATFORM,
)
from deepdash.core.generator import generate_level
from deepdash.core.game_state import GameState


DISPLAY_SCALE = 3  # 256x192 * 3 = 768x576
FPS = 30


def draw_game(screen: pygame.Surface, game_state: GameState):
    """Draw the game at full internal resolution onto the screen."""
    # Create internal surface
    internal = pygame.Surface((INTERNAL_WIDTH, INTERNAL_HEIGHT))
    internal.fill(COLOR_SKY)

    cam_x = game_state.camera_x
    level = game_state.level
    player = game_state.player

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
                pygame.draw.rect(internal, COLOR_GROUND, (sx, sy, TILE_SIZE, TILE_SIZE))
            elif tile == TileType.PLATFORM:
                pygame.draw.rect(internal, COLOR_PLATFORM, (sx, sy, TILE_SIZE, TILE_SIZE))
            elif tile == TileType.SPIKE:
                points = [
                    (sx + TILE_SIZE // 2, sy),
                    (sx, sy + TILE_SIZE),
                    (sx + TILE_SIZE, sy + TILE_SIZE),
                ]
                pygame.draw.polygon(internal, COLOR_SPIKE, points)

    # Draw player
    px = int(player.x - cam_x)
    py = int(player.y)
    pygame.draw.rect(internal, COLOR_PLAYER, (px, py, player.width, player.height))

    # Draw HUD
    font = pygame.font.SysFont(None, 16)
    tick_text = font.render(f"Tick: {game_state.tick}", True, (255, 255, 255))
    internal.blit(tick_text, (5, 5))

    # Scale up to display
    scaled = pygame.transform.scale(internal, screen.get_size())
    screen.blit(scaled, (0, 0))


def main():
    parser = argparse.ArgumentParser(description="Play DeepDash")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--length", type=int, default=200)
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((INTERNAL_WIDTH * DISPLAY_SCALE, INTERNAL_HEIGHT * DISPLAY_SCALE))
    pygame.display.set_caption("DeepDash - SPACE/UP to jump")
    clock = pygame.time.Clock()

    def new_game():
        level = generate_level(seed=args.seed, length=args.length, difficulty=args.difficulty)
        return GameState(level)

    game_state = new_game()
    running = True

    while running:
        action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game_state = new_game()

        # Continuous key press for jumping
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = 1

        if not game_state.done:
            reward, done = game_state.step(action)
            if done:
                if game_state.won:
                    print("YOU WIN! Press R to restart.")
                else:
                    print("GAME OVER! Press R to restart.")

        draw_game(screen, game_state)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
