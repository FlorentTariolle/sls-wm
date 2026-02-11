from typing import Any

import gymnasium as gym
import numpy as np

from deepdash.core.constants import OUTPUT_SIZE, DEFAULT_LEVEL_LENGTH
from deepdash.core.generator import generate_level
from deepdash.core.game_state import GameState
from deepdash.renderer.standard_renderer import StandardRenderer
from deepdash.renderer.semantic_renderer import SemanticRenderer


class DeepDashEnv(gym.Env):
    """Geometry Dash clone environment for RL training.

    Render modes:
        - "rgb_array": standard colorful rendering, returns 64x64x3 array
        - "semantic": semantic rendering (green=player, red=spike, etc.)
        - "human": opens a pygame window for human viewing
    """

    metadata = {
        "render_modes": ["rgb_array", "semantic", "human"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        difficulty: float = 0.5,
        level_length: int = DEFAULT_LEVEL_LENGTH,
    ):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.level_length = level_length

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(2)  # 0=no jump, 1=jump

        self._game_state: GameState | None = None
        self._standard_renderer: StandardRenderer | None = None
        self._semantic_renderer: SemanticRenderer | None = None
        self._human_display = None

    def _get_renderer(self):
        if self.render_mode == "semantic":
            if self._semantic_renderer is None:
                self._semantic_renderer = SemanticRenderer()
            return self._semantic_renderer
        else:
            if self._standard_renderer is None:
                self._standard_renderer = StandardRenderer()
            return self._standard_renderer

    def _get_obs(self) -> np.ndarray:
        renderer = self._get_renderer()
        return renderer.render_frame(self._game_state)

    def _get_info(self) -> dict[str, Any]:
        gs = self._game_state
        return {
            "tick": gs.tick,
            "player_x": gs.player.x,
            "player_y": gs.player.y,
            "alive": gs.player.alive,
            "won": gs.won,
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        level_seed = seed if seed is not None else self.np_random.integers(0, 2**31)
        level = generate_level(
            seed=int(level_seed),
            length=self.level_length,
            difficulty=self.difficulty,
        )
        self._game_state = GameState(level)

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_human(obs)

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward, done = self._game_state.step(action)

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_human(obs)

        return obs, reward, done, False, info

    def render(self) -> np.ndarray | None:
        if self.render_mode in ("rgb_array", "semantic"):
            return self._get_obs()
        elif self.render_mode == "human":
            obs = self._get_obs()
            self._render_human(obs)
            return None

    def _render_human(self, obs: np.ndarray):
        import pygame

        display_size = 512
        if self._human_display is None:
            pygame.display.init()
            self._human_display = pygame.display.set_mode((display_size, display_size))
            pygame.display.set_caption("DeepDash")

        # obs is (H, W, 3), pygame needs (W, H, 3)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (display_size, display_size))
        self._human_display.blit(surf, (0, 0))
        pygame.display.flip()

    def close(self):
        if self._human_display is not None:
            import pygame
            pygame.display.quit()
            self._human_display = None
