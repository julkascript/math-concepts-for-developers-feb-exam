"""
Terrain generation from seed: multi-octave Perlin noise as height map.

height(x,y) = sum over i of persistence^i * noise(frequency^i * x, frequency^i * y)
"""

import numpy as np
from typing import Union, Tuple, Optional

from src.perlin import PerlinNoise2D


def generate_terrain(
    seed: Union[int, str],
    size: Tuple[int, int] = (256, 256),
    octaves: int = 4,
    persistence: float = 0.5,
    frequency: float = 0.01,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Generate a 2D height map from a seed using multi-octave Perlin noise.

    Parameters
    ----------
    seed : int or str
        World seed; same seed yields identical terrain.
    size : (rows, cols)
        Output shape (height, width).
    octaves : int
        Number of noise layers; more => more detail.
    persistence : float
        Amplitude multiplier per octave (< 1 dampens high frequencies).
    frequency : float
        Base spatial frequency (higher => more zoomed-out / larger features).
    scale : float
        Overall height scale (multiplier on final sum).

    Returns
    -------
    height_map : np.ndarray
        Float array of shape `size`, values typically in a bounded range.
    """
    r, c = size
    yy, xx = np.mgrid[0:r, 0:c].astype(np.float64)
    perlin = PerlinNoise2D(seed)

    total = np.zeros((r, c), dtype=np.float64)
    amp = 1.0
    freq = 1.0
    max_val = 0.0  # for normalization (optional; we don't normalize by default)

    for _ in range(octaves):
        total += amp * perlin.noise_grid(xx * frequency * freq, yy * frequency * freq)
        max_val += amp
        amp *= persistence
        freq *= 2.0

    return total * scale


def heightmap_to_rgb(
    height_map: np.ndarray,
    sea_level: Optional[float] = None,
) -> np.ndarray:
    """
    Map height map to an RGB image (simple bands: water, low, mid, high).

    Parameters
    ----------
    height_map : np.ndarray
        2D float array.
    sea_level : float or None
        If set, values below are drawn as water (blue); else use 0.

    Returns
    -------
    rgb : np.ndarray
        (H, W, 3) uint8 array.
    """
    h = np.asarray(height_map)
    if sea_level is None:
        sea_level = np.percentile(h, 25)
    lo, hi = np.nanmin(h), np.nanmax(h)
    if hi <= lo:
        return np.zeros((*h.shape, 3), dtype=np.uint8)
    t = (h - lo) / (hi - lo)
    r = np.clip((t - 0.5) * 2, 0, 1)
    g = np.clip(1 - np.abs(t - 0.5) * 2, 0, 1)
    b = np.clip(t, 0, 1)
    # Water: below sea_level
    water = (h <= sea_level)
    norm_sea = (sea_level - lo) / (hi - lo) if hi > lo else 0
    r[water] = 0.2
    g[water] = 0.3
    b[water] = 0.9
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)
