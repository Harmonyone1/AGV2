"""RL environment wrappers."""
from __future__ import annotations

import numpy as np
from gymnasium import ObservationWrapper


class ObservationStatsWrapper(ObservationWrapper):
    """Normalizes observations using running mean/variance."""

    def __init__(self, env, epsilon: float = 1e-6):
        super().__init__(env)
        shape = self.observation_space.shape
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def observation(self, observation):  # type: ignore[override]
        self._update_stats(observation)
        return (observation - self.mean) / (np.sqrt(self.var) + 1e-6)

    def _update_stats(self, observation):
        self.count += 1.0
        alpha = 1.0 / self.count
        delta = observation - self.mean
        self.mean += alpha * delta
        self.var += alpha * (delta ** 2 - self.var)


__all__ = ["ObservationStatsWrapper"]
