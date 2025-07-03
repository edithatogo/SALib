import math

import numpy as np
from scipy.stats.qmc import Sobol


def sample(N: int, D: int) -> np.ndarray:
    """Generate ``(N, D)`` array of Sobol sequence samples."""
    sampler = Sobol(d=D, scramble=False)
    m = int(math.log2(N)) if N > 0 else 0
    if 2 ** m == N:
        values = sampler.random_base2(m)
    else:
        values = sampler.random(N)
    return np.asarray(values)

