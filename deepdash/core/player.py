from deepdash.core.constants import (
    GRAVITY, JUMP_VELOCITY, PLAYER_SPEED, PLAYER_WIDTH, PLAYER_HEIGHT,
)


class Player:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = PLAYER_SPEED
        self.vy = 0.0
        self.on_ground = False
        self.alive = True

    def apply_action(self, action: int):
        """action: 0 = do nothing, 1 = jump (only if on ground)."""
        if action == 1 and self.on_ground:
            self.vy = JUMP_VELOCITY
            self.on_ground = False

    def apply_physics(self):
        """Apply gravity and velocity to position."""
        self.vy += GRAVITY
        self.x += self.vx
        self.y += self.vy

    @property
    def width(self) -> int:
        return PLAYER_WIDTH

    @property
    def height(self) -> int:
        return PLAYER_HEIGHT

    @property
    def hitbox(self) -> tuple[float, float, float, float]:
        """Return (left, top, right, bottom)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
