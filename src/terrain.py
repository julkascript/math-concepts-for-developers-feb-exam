"""
Terrain generation from seed: multi-octave Perlin noise (fBm) as height map.

height(x,y) = sum_{i=0}^{k-1} persistence^i * noise(frequency * lacunarity^i * x,
                                                      frequency * lacunarity^i * y)

lacunarity is fixed at 2.0 (each octave doubles the spatial frequency).
persistence controls amplitude decay per octave (typically 0.5).
"""

import numpy as np
from typing import Union, Tuple, Optional

from src.perlin import PerlinNoise2D


LACUNARITY = 2.0  # frequency multiplier per octave (standard value)


def generate_terrain(
    seed: Union[int, str],
    size: Tuple[int, int] = (256, 256),
    octaves: int = 4,
    persistence: float = 0.5,
    frequency: float = 0.01,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Generate a 2D height map from a seed using multi-octave Perlin noise (fBm).

    Parameters
    ----------
    seed : int or str
        World seed; same seed yields identical terrain.
    size : (rows, cols)
        Output shape (height, width).
    octaves : int
        Number of noise layers. Each octave doubles the spatial frequency
        (lacunarity = 2.0) and halves the amplitude (when persistence = 0.5).
        More octaves add finer detail.
    persistence : float
        Amplitude multiplier per octave (typically < 1; 0.5 halves amplitude
        per octave, dampening high-frequency contributions).
    frequency : float
        Base spatial frequency. Higher values produce larger terrain features
        (more "zoomed out"). Typical range: 0.005 – 0.05.
    scale : float
        Overall height scale (multiplier on final sum).

    Returns
    -------
    height_map : np.ndarray
        Float array of shape `size`. Values are *not* normalized by default;
        the theoretical maximum is scale * sum(persistence^i, i=0..octaves-1)
        = scale * (1 - persistence^octaves) / (1 - persistence).
    """
    r, c = size
    yy, xx = np.mgrid[0:r, 0:c].astype(np.float64)
    perlin = PerlinNoise2D(seed)

    total = np.zeros((r, c), dtype=np.float64)
    amp = 1.0
    freq = 1.0

    for _ in range(octaves):
        total += amp * perlin.noise_grid(xx * frequency * freq, yy * frequency * freq)
        amp *= persistence
        freq *= LACUNARITY

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
