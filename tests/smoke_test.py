"""Minimal data-free check of the original BaySC implementation."""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.dahl import get_dahl
from src.core.sampler import run_sampler
from src.preprocessing.similarity import (
    compute_neighbor_matrix,
    compute_similarity_matrix,
)


def main():
    rng = np.random.default_rng(7)
    n = 12

    embedding = rng.normal(size=(n, 5))
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    coordinates = np.column_stack((np.arange(n), np.zeros(n)))

    similarity = compute_similarity_matrix(
        embedding,
        transform="fisher_z",
        keep_diag=True,
        method="cosine",
    )
    neighbors = compute_neighbor_matrix(coordinates, threshold=1.1)

    # The sampler expects log V_n values indexed by the active domain count.
    # Zero values are sufficient for this execution-only smoke test because
    # it is intended to validate imports, array shapes, and the MCMC path.
    log_vn = np.zeros(n + 20, dtype=float)

    result = run_sampler(
        A_list=[similarity],
        alpha_list=[1.0],
        neighbor_matrix=neighbors,
        lambda1=1.0,
        Vn=log_vn,
        n_iterations=4,
        init_K=3,
        seed=7,
        verbose=False,
        max_K=10,
    )
    summary = get_dahl(result["history"], burn_in=2)

    assert result["history"]["z"].shape == (4, n)
    assert summary["cluster_assign"].shape == (n,)
    assert np.all(summary["cluster_assign"] >= 1)
    print("BaySC smoke test passed.")


if __name__ == "__main__":
    main()
