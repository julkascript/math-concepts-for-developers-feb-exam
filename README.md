# Minecraft-Style Seed-Based Procedural Terrain Generation

A mathematics and Python project that implements and explains a **seed-driven procedural world generation** system inspired by Minecraft: from a single integer seed, we produce deterministic, infinite-looking 2D terrain using pseudorandom number generators (PRNGs) and Perlin noise with octaves.

## Problem

**Given a seed (integer or string), produce a deterministic 2D terrain** that looks natural and is reproducible. Same seed always yields the same terrain; different seeds yield different worlds. This mirrors how games like Minecraft use seeds for level generation.

## Contents

- **Mathematical concepts:** Linear Congruential Generator (LCG), Perlin noise (gradients, dot products, smooth interpolation), and fractal octaves (persistence, frequency).
- **Code:** Reusable Python modules in `src/` (PRNG, Perlin noise, terrain generation) and a Jupyter notebook that walks through theory, implementation, and experiments.
- **Experiments:** Reproducibility checks, seed sensitivity, and parameter sensitivity (octaves, persistence).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── notebooks/          # Jupyter notebook(s)
│   └── seed_terrain_analysis.ipynb
└── src/                # Reusable Python modules
    ├── prng.py         # Seed → deterministic random sequence (LCG)
    ├── perlin.py       # 2D Perlin noise
    └── terrain.py     # Terrain from noise + octaves, visualization helpers
```

## How to run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the notebook:**
   ```bash
   jupyter notebook notebooks/seed_terrain_analysis.ipynb
   ```
   Or open `notebooks/seed_terrain_analysis.ipynb` in Jupyter Lab / VS Code.

3. **Use the library (optional):**
   ```python
   from src.prng import seed_to_int, LCG
   from src.perlin import PerlinNoise2D
   from src.terrain import generate_terrain
   ```

## References

- [Minecraft Wiki – Seed (level generation)](https://minecraft.fandom.com/wiki/Seed_(level_generation))
- [Perlin noise (Wikipedia)](https://en.wikipedia.org/wiki/Perlin_noise)
- [Perlin noise math FAQ / implementation](https://adrianb.io/2014/08/09/perlinnoise.html)
- Java `Random` / LCG (e.g. Oracle documentation)

## License and academic integrity

This project is for educational purposes. All sources are cited in the notebook. No plagiarism; comply with your institution’s academic integrity policy.
