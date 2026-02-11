import numpy as np

from deepdash.core.constants import TileType, TILE_SIZE, VIEWPORT_TILES_Y


class Level:
    def __init__(self, grid: np.ndarray):
        """grid: 2D array of TileType values, shape (rows, cols)."""
        self.grid = grid

    @property
    def rows(self) -> int:
        return self.grid.shape[0]

    @property
    def cols(self) -> int:
        return self.grid.shape[1]

    @property
    def width_pixels(self) -> int:
        return self.cols * TILE_SIZE

    @property
    def height_pixels(self) -> int:
        return self.rows * TILE_SIZE

    def get_tile(self, col: int, row: int) -> TileType:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return TileType(self.grid[row, col])
        return TileType.EMPTY

    def get_tile_at_pixel(self, px: float, py: float) -> TileType:
        col = int(px // TILE_SIZE)
        row = int(py // TILE_SIZE)
        return self.get_tile(col, row)

    def get_tiles_in_rect(self, left: float, top: float, right: float, bottom: float) -> list[tuple[int, int, TileType]]:
        """Return all tiles that overlap with the given pixel rect."""
        col_start = max(0, int(left // TILE_SIZE))
        col_end = min(self.cols - 1, int(right // TILE_SIZE))
        row_start = max(0, int(top // TILE_SIZE))
        row_end = min(self.rows - 1, int(bottom // TILE_SIZE))

        tiles = []
        for r in range(row_start, row_end + 1):
            for c in range(col_start, col_end + 1):
                t = TileType(self.grid[r, c])
                if t != TileType.EMPTY:
                    tiles.append((c, r, t))
        return tiles
