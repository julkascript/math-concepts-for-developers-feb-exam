"""
2D Perlin noise: coherent gradient noise for natural-looking procedural terrain.

Algorithm: grid of unit cells; at each vertex a random gradient vector;
for each point, dot products of gradients with distance vectors, then
smooth interpolation (ease curve s(t) = 3t^2 - 2t^3).
"""

import numpy as np
from typing import Union

from src.prng import seed_to_int, LCG


# Four gradient directions for 2D (unit vectors) — avoids diagonal bias
GRADIENTS_2D = np.array([
    [1, 0], [-1, 0], [0, 1], [0, -1],
    [1, 1], [-1, -1], [1, -1], [-1, 1]
], dtype=np.float64)
# Normalize the diagonals
for i in range(4, 8):
    GRADIENTS_2D[i] /= np.sqrt(2)


def _ease(t: np.ndarray) -> np.ndarray:
    """Smoothstep (ease) curve: s(t) = 3t^2 - 2t^3. Zero derivative at 0 and 1."""
    return t * t * (3.0 - 2.0 * t)


class PerlinNoise2D:
    """
    2D Perlin noise with seed-driven permutation table for deterministic output.
    """

    PERM_SIZE = 256

    def __init__(self, seed: Union[int, str]):
        """
        Build permutation table and gradient indices from seed. Same seed => same noise.
        """
        rng = LCG(seed)
        # Permutation of [0..255]
        perm = list(range(self.PERM_SIZE))
        for i in range(self.PERM_SIZE - 1, 0, -1):
            j = rng.next_int(i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        self._perm = np.array(perm, dtype=np.int32)
        # Gradient index per vertex (0..7)
        self._grad_idx = self._perm % len(GRADIENTS_2D)

    def _gradient_at(self, ix: int, iy: int) -> np.ndarray:
        """Get gradient vector at grid cell (ix, iy) using permutation."""
        i = (ix % self.PERM_SIZE) + (iy % self.PERM_SIZE) * 0  # avoid 2D perm if we want
        # Standard approach: combine ix, iy into single perm index
        idx = (self._perm[ix % self.PERM_SIZE] + iy) % self.PERM_SIZE
        g = self._grad_idx[idx]
        return GRADIENTS_2D[g]

    def noise_scalar(self, x: float, y: float) -> float:
        """
        Single noise value at (x, y). Returns value in approximately [-1, 1].
        """
        x0 = int(np.floor(x)) & 0xFF
        y0 = int(np.floor(y)) & 0xFF
        x1 = x0 + 1
        y1 = y0 + 1
        u = x - np.floor(x)
        v = y - np.floor(y)

        g00 = self._gradient_at(x0, y0)
        g10 = self._gradient_at(x1, y0)
        g01 = self._gradient_at(x0, y1)
        g11 = self._gradient_at(x1, y1)

        d00 = np.array([u, v], dtype=np.float64)
        d10 = np.array([u - 1, v], dtype=np.float64)
        d01 = np.array([u, v - 1], dtype=np.float64)
        d11 = np.array([u - 1, v - 1], dtype=np.float64)

        n00 = np.dot(g00, d00)
        n10 = np.dot(g10, d10)
        n01 = np.dot(g01, d01)
        n11 = np.dot(g11, d11)

        su = _ease(np.array([u]))[0]
        sv = _ease(np.array([v]))[0]

        nx0 = n00 * (1 - su) + n10 * su
        nx1 = n01 * (1 - su) + n11 * su
        return float(nx0 * (1 - sv) + nx1 * sv)

    def noise_grid(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Vectorized evaluation over 2D arrays x, y. Shapes must be broadcast-compatible.
        Returns array of same shape with noise values in approximately [-1, 1].
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shape = np.broadcast_shapes(x.shape, y.shape)
        x = np.broadcast_to(x, shape).copy()
        y = np.broadcast_to(y, shape).copy()

        x0 = np.floor(x).astype(np.int32) & (self.PERM_SIZE - 1)
        y0 = np.floor(y).astype(np.int32) & (self.PERM_SIZE - 1)
        x1 = (x0 + 1) & (self.PERM_SIZE - 1)
        y1 = (y0 + 1) & (self.PERM_SIZE - 1)
        u = x - np.floor(x)
        v = y - np.floor(y)

        def grad_dot(ix, iy, dx, dy):
            idx = (self._perm[ix] + iy) % self.PERM_SIZE
            g = GRADIENTS_2D[self._grad_idx[idx]]  # shape (..., 2)
            return g[..., 0] * dx + g[..., 1] * dy

        n00 = grad_dot(x0, y0, u, v)
        n10 = grad_dot(x1, y0, u - 1, v)
        n01 = grad_dot(x0, y1, u, v - 1)
        n11 = grad_dot(x1, y1, u - 1, v - 1)

        su = _ease(u)
        sv = _ease(v)

        nx0 = n00 * (1 - su) + n10 * su
        nx1 = n01 * (1 - su) + n11 * su
        return (nx0 * (1 - sv) + nx1 * sv).astype(np.float64)
