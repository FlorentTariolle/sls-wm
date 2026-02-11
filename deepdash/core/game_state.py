from deepdash.core.constants import (
    TileType, TILE_SIZE, PLAYER_SPEED, VIEWPORT_TILES_X,
    INTERNAL_WIDTH, INTERNAL_HEIGHT,
)
from deepdash.core.player import Player
from deepdash.core.level import Level


class GameState:
    def __init__(self, level: Level):
        self.level = level
        # Spawn player on ground in the safe zone
        spawn_x = 3.0 * TILE_SIZE
        spawn_y = (level.grid.shape[0] - 1) * TILE_SIZE
        # Find the actual ground surface for spawn
        from deepdash.core.constants import GROUND_ROW
        spawn_y = float(GROUND_ROW * TILE_SIZE - 14)  # player height = 14
        self.player = Player(spawn_x, spawn_y)
        self.camera_x = 0.0
        self.tick = 0
        self.done = False
        self.won = False

    def step(self, action: int) -> tuple[float, bool]:
        """Advance one tick. Returns (reward, done)."""
        if self.done:
            return 0.0, True

        # 1. Apply action
        self.player.apply_action(action)

        # 2. Apply physics
        self.player.apply_physics()

        # 3. Collision resolution
        self._resolve_collisions()

        # 4. Update camera
        self._update_camera()

        # 5. Death checks
        if not self.player.alive:
            self.done = True
            return -100.0, True

        # Fell off screen
        if self.player.y > INTERNAL_HEIGHT + TILE_SIZE:
            self.player.alive = False
            self.done = True
            return -100.0, True

        # 6. Win check: reached end of level
        if self.player.x >= self.level.width_pixels - TILE_SIZE * 2:
            self.done = True
            self.won = True
            return 100.0, True

        self.tick += 1
        return 1.0, False

    def _resolve_collisions(self):
        """Resolve collisions: vertical landing → spikes → wall death.

        Vertical resolution happens first so the player snaps to the ground
        surface before spike checks — prevents false spike kills from the
        sub-pixel gravity sinking that happens each tick.
        """
        player = self.player

        # --- Pass 1: Vertical resolution (landing / ceiling bumps) ---
        tiles = self.level.get_tiles_in_rect(*player.hitbox)
        player.on_ground = False
        solids = sorted(
            [(c, r, t) for c, r, t in tiles if t in (TileType.GROUND, TileType.PLATFORM)],
            key=lambda x: x[1],
        )

        for col, row, tile_type in solids:
            tile_top = row * TILE_SIZE
            tile_bottom = tile_top + TILE_SIZE
            _, p_top, _, p_bottom = player.hitbox

            if p_bottom <= tile_top or p_top >= tile_bottom:
                continue

            if player.vy >= 0 and p_top < tile_top:
                # Falling and player's top is above tile → land on top
                player.y = tile_top - player.height
                player.vy = 0
                player.on_ground = True
            elif player.vy < 0 and p_bottom > tile_bottom:
                # Rising and player's bottom is below tile → bump ceiling
                player.y = tile_bottom
                player.vy = 0

        # --- Pass 2: Spike check (after vertical snap) ---
        tiles = self.level.get_tiles_in_rect(*player.hitbox)
        for col, row, tile_type in tiles:
            if tile_type == TileType.SPIKE:
                player.alive = False
                return

        # --- Pass 3: Wall collision (with resolved position) ---
        for col, row, tile_type in tiles:
            if tile_type in (TileType.GROUND, TileType.PLATFORM):
                tile_left = col * TILE_SIZE
                tile_top = row * TILE_SIZE
                tile_bottom = tile_top + TILE_SIZE
                p_left, p_top, p_right, p_bottom = player.hitbox
                # Must have real vertical overlap (not just touching)
                if p_bottom <= tile_top or p_top >= tile_bottom:
                    continue
                if p_left < tile_left and p_right > tile_left:
                    player.alive = False
                    return

    def _update_camera(self):
        """Keep player in left third of viewport."""
        target = self.player.x - INTERNAL_WIDTH // 3
        self.camera_x = max(0.0, target)
        max_cam = max(0.0, self.level.width_pixels - INTERNAL_WIDTH)
        self.camera_x = min(self.camera_x, max_cam)
