from abc import ABC, abstractmethod

import numpy as np

from deepdash.core.game_state import GameState


class BaseRenderer(ABC):
    @abstractmethod
    def render_frame(self, game_state: GameState) -> np.ndarray:
        """Render the current game state to a 64x64x3 uint8 numpy array."""
        ...
