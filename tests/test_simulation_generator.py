"""Basic reproducibility checks for the simulation data generator."""

import numpy as np

from simulations.generator import simulate_paired_counts


def test_generator_is_deterministic_and_has_expected_shapes():
    kwargs = dict(
        layout="irregular_compartments",
        log_fold_change=0.80,
        n_spots=40,
        n_domains=5,
        seed=12345,
        layout_seed=54321,
    )
    first = simulate_paired_counts(**kwargs)
    second = simulate_paired_counts(**kwargs)

    assert first.rna_counts.shape == (40, 2000)
    assert first.atac_counts.shape == (40, 4000)
    assert first.coordinates.shape == (40, 2)
    assert set(np.unique(first.labels)) == set(range(5))
    assert np.array_equal(first.rna_counts, second.rna_counts)
    assert np.array_equal(first.atac_counts, second.atac_counts)
    assert np.array_equal(first.coordinates, second.coordinates)
    assert np.array_equal(first.labels, second.labels)
