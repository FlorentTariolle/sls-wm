import numpy as np

from deepdash.core.constants import (
    TileType, VIEWPORT_TILES_Y, DEFAULT_LEVEL_LENGTH,
    SAFE_ZONE_START, SAFE_ZONE_END, GROUND_ROW,
)
from deepdash.core.level import Level


def generate_level(
    seed: int | None = None,
    length: int = DEFAULT_LEVEL_LENGTH,
    difficulty: float = 0.5,
) -> Level:
    """Generate a guaranteed-beatable procedural level.

    Solvability is enforced by:
    - Capping consecutive spikes (player can clear ~5 in one jump)
    - Enforcing safe landing zones after spikes and gaps
    - Requiring a clear run-up column before gaps
    - Limiting gap width to jumpable distances

    Args:
        seed: RNG seed for reproducibility.
        length: Number of columns.
        difficulty: 0.0 (easy) to 1.0 (hard).
    """
    rng = np.random.default_rng(seed)
    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    rows = VIEWPORT_TILES_Y  # 12
    grid = np.zeros((rows, length), dtype=np.int32)

    # Difficulty-scaled probabilities
    gap_chance = 0.04 + 0.12 * difficulty       # 4–16%
    spike_chance = 0.08 + 0.25 * difficulty      # 8–33%
    platform_chance = 0.03 + 0.10 * difficulty   # 3–13%

    # Solvability constraints (jump: 25 ticks, ~6 tiles horizontal)
    # Bot detects obstacles 2-4 tiles ahead, needs safe landing + detection margin
    max_consecutive_spikes = 2 + int(difficulty)       # 2–3
    max_gap_width = 2 + int(difficulty)                 # 2–3 tiles
    safe_after_spikes = 5     # clear columns after spike run
    safe_after_gap = 5        # clear columns after gap
    safe_before_gap = 1       # clear column before a gap for takeoff
    min_cols_between_gaps = 8  # minimum ground columns between gaps

    # State tracking
    safe_cooldown = 0          # forced safe ground columns remaining
    consecutive_spikes = 0
    in_gap = False
    gap_remaining = 0
    cols_since_gap_end = 100   # start high so first gap can appear early

    for col in range(length):
        # --- Safe zones at start/end ---
        if col < SAFE_ZONE_START or col >= length - SAFE_ZONE_END:
            _fill_ground(grid, col, rows)
            continue

        # --- Forced safe cooldown (landing zones) ---
        if safe_cooldown > 0:
            _fill_ground(grid, col, rows)
            safe_cooldown -= 1
            consecutive_spikes = 0
            cols_since_gap_end += 1
            continue

        # --- Inside a gap ---
        if in_gap:
            gap_remaining -= 1
            if gap_remaining <= 0:
                in_gap = False
                safe_cooldown = safe_after_gap  # guarantee landing zone
                cols_since_gap_end = 0
            else:
                # Optionally place a platform in wider gaps
                if gap_remaining >= 1 and rng.random() < platform_chance * 0.3:
                    plat_row = rng.integers(GROUND_ROW - 3, GROUND_ROW)
                    grid[plat_row, col] = TileType.PLATFORM
            continue

        cols_since_gap_end += 1

        # --- Try to start a gap ---
        can_gap = (
            cols_since_gap_end >= min_cols_between_gaps
            and consecutive_spikes == 0  # don't start gap right after spikes
            and col + max_gap_width + safe_after_gap < length - SAFE_ZONE_END
        )
        if can_gap and rng.random() < gap_chance:
            # Place ground on this column as run-up, gap starts next column
            if safe_before_gap > 0:
                _fill_ground(grid, col, rows)
                consecutive_spikes = 0
                in_gap = True
                gap_remaining = rng.integers(2, max_gap_width + 1)
            else:
                in_gap = True
                gap_remaining = rng.integers(2, max_gap_width + 1)
            continue

        # --- Normal ground column ---
        _fill_ground(grid, col, rows)

        # Try to place a spike (respect max consecutive)
        if consecutive_spikes < max_consecutive_spikes and rng.random() < spike_chance:
            grid[GROUND_ROW - 1, col] = TileType.SPIKE
            consecutive_spikes += 1
        else:
            # End of spike run — enforce landing zone
            if consecutive_spikes > 0:
                safe_cooldown = safe_after_spikes
            consecutive_spikes = 0

        # Optionally place a floating platform (not on spike columns)
        if grid[GROUND_ROW - 1, col] != TileType.SPIKE and rng.random() < platform_chance:
            plat_row = rng.integers(GROUND_ROW - 5, GROUND_ROW - 2)
            grid[plat_row, col] = TileType.PLATFORM

    return Level(grid)


def _fill_ground(grid: np.ndarray, col: int, rows: int):
    """Fill a column with ground from GROUND_ROW to bottom."""
    for r in range(GROUND_ROW, rows):
        grid[r, col] = TileType.GROUND
