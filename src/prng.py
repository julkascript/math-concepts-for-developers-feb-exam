"""
Deterministic pseudorandom number generation from a seed.

Uses a Linear Congruential Generator (LCG): x_{n+1} = (a*x_n + c) mod m.
Same seed always produces the same sequence, which is required for
reproducible procedural generation (e.g. Minecraft-style worlds).
"""

from typing import Union


def seed_to_int(seed: Union[int, str]) -> int:
    """
    Convert a seed to a non-negative integer suitable for the LCG.

    - int: normalized to 64-bit range; we use the lower 48 bits for the LCG state.
    - str: hashed similarly to Java's String.hashCode() for compatibility discussion;
      Python's hash() is not stable across runs, so we use a simple deterministic hash.

    Returns:
        Integer in range [0, 2**64 - 1] (as a Python int; we take mod 2**48 for LCG).
    """
    if isinstance(seed, int):
        # Normalize to 64-bit unsigned range
        return seed & ((1 << 64) - 1)
    # Deterministic string hash (polynomial rolling hash)
    h = 0
    for c in seed:
        h = (31 * h + ord(c)) & ((1 << 64) - 1)
    return h


class LCG:
    """
    Linear Congruential Generator: x_{n+1} = (a * x_n + c) mod m.

    Parameters (e.g. Java's Random uses a=0x5DEECE66D, c=11, m=2**48) give
    a full-period 48-bit generator. We use similar constants for clarity.
    """

    # Java Random constants (48-bit)
    A = 0x5DEECE66D
    C = 11
    M = 1 << 48
    MASK = M - 1

    def __init__(self, seed: Union[int, str]):
        """
        Initialize the LCG with a seed. Same seed yields same sequence.
        """
        s = seed_to_int(seed)
        # Java scrambles initial state with (seed ^ A) & MASK
        self._state = (s ^ self.A) & self.MASK

    def next_bits(self, bits: int) -> int:
        """
        Return the next `bits` bits (unsigned). Used to implement nextInt, nextLong, etc.
        """
        self._state = (self.A * self._state + self.C) & self.MASK
        return self._state >> (48 - bits)

    def next_int(self, bound: int = None) -> int:
        """
        Next random integer. If bound is None, return full 32 bits (signed).
        If bound is given, return value in [0, bound) (like random.randrange).
        """
        if bound is None:
            return self.next_bits(32)
        # Reject values that would cause non-uniform distribution
        if bound <= 0:
            raise ValueError("bound must be positive")
        r = self.next_bits(31)
        b = bound
        while True:
            if r < (1 << 31) - (1 << 31) % b:
                return r % b
            r = self.next_bits(31)

    def next_float(self) -> float:
        """Next float in [0.0, 1.0)."""
        return self.next_bits(24) / (1 << 24)

    def next_double(self) -> float:
        """Next double in [0.0, 1.0) (higher precision)."""
        return ((self.next_bits(26) << 27) + self.next_bits(27)) / (1 << 53)

    def __iter__(self):
        """Infinite stream of floats in [0, 1)."""
        while True:
            yield self.next_float()
