"""Command-line generator for the three BaySC simulation studies.

Examples
--------
Generate all 20 datasets from Simulation 1::

    python -m simulations.generate simulation1

Generate all 45 datasets from Simulation 2::

    python -m simulations.generate simulation2

Generate the seven data settings used in Simulation 3::

    python -m simulations.generate simulation3
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .generator import simulate_paired_counts


SIMULATION1_LAYOUTS = {
    "irregular_compartments": (0.70, 18_000_000),
    "nested_microenvironment": (0.84, 18_001_000),
    "disconnected_islands": (0.65, 18_002_000),
    "transition_boundaries": (0.65, 18_003_000),
}

SIMULATION2_SIGNAL = {
    "low": 0.60,
    "medium": 0.80,
    "high": 1.10,
}
SIMULATION2_QUALITY = {
    "balanced": (1.0, 1.0),
    "rna_dominant": (1.0, 0.75),
    "atac_dominant": (0.75, 1.0),
}

SIMULATION3_SETTINGS = (
    ("n-0500_k-05", 500, 5),
    ("n-1000_k-05", 1000, 5),
    ("n-1000_k-03", 1000, 3),
    ("n-1000_k-07", 1000, 7),
    ("n-1000_k-10", 1000, 10),
    ("n-2000_k-05", 2000, 5),
    ("n-4000_k-05", 4000, 5),
)


def _save_dataset(dataset, destination: Path, condition_metadata: dict):
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        destination / "dataset.npz",
        rna_counts=dataset.rna_counts,
        atac_counts=dataset.atac_counts,
        coordinates=dataset.coordinates,
        labels=dataset.labels,
        boundary_mask=dataset.boundary_mask,
        distance_to_boundary=dataset.distance_to_boundary,
    )
    metadata = dict(dataset.metadata)
    metadata.update(condition_metadata)
    with (destination / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def generate_simulation1(output_root: Path):
    for layout, (log_fold_change, seed_base) in SIMULATION1_LAYOUTS.items():
        for replicate in range(5):
            seed = seed_base + replicate
            destination = (
                output_root
                / "simulation1"
                / layout
                / "replicate-{:03d}".format(replicate + 1)
            )
            dataset = simulate_paired_counts(
                layout=layout,
                log_fold_change=log_fold_change,
                n_spots=500,
                n_domains=5,
                seed=seed,
            )
            _save_dataset(
                dataset,
                destination,
                {
                    "simulation": 1,
                    "replicate": replicate + 1,
                    "replicate_index_in_seed": replicate,
                },
            )
            print(destination)


def generate_simulation2(output_root: Path):
    signal_names = tuple(SIMULATION2_SIGNAL)
    quality_names = tuple(SIMULATION2_QUALITY)
    for signal_index, signal in enumerate(signal_names):
        for quality_index, quality in enumerate(quality_names):
            condition_index = signal_index * len(quality_names) + quality_index
            rna_quality, atac_quality = SIMULATION2_QUALITY[quality]
            for replicate in range(5):
                count_seed = 20_000_000 + 1_000 * condition_index + replicate
                layout_seed = 20_900_000 + replicate
                destination = (
                    output_root
                    / "simulation2"
                    / "signal-{}__quality-{}".format(signal, quality)
                    / "replicate-{:03d}".format(replicate + 1)
                )
                dataset = simulate_paired_counts(
                    layout="irregular_compartments",
                    log_fold_change=SIMULATION2_SIGNAL[signal],
                    n_spots=500,
                    n_domains=5,
                    seed=count_seed,
                    layout_seed=layout_seed,
                    rna_quality_multiplier=rna_quality,
                    atac_quality_multiplier=atac_quality,
                )
                _save_dataset(
                    dataset,
                    destination,
                    {
                        "simulation": 2,
                        "signal_level": signal,
                        "modality_quality": quality,
                        "replicate": replicate + 1,
                        "replicate_index_in_seed": replicate,
                    },
                )
                print(destination)


def generate_simulation3(output_root: Path):
    for setting_index, (setting, n_spots, n_domains) in enumerate(
        SIMULATION3_SETTINGS
    ):
        count_seed = 21_000_000 + 1_000 * setting_index
        layout_seed = 21_900_000 + 1_000 * setting_index
        destination = output_root / "simulation3" / setting
        dataset = simulate_paired_counts(
            layout="irregular_compartments",
            log_fold_change=0.80,
            n_spots=n_spots,
            n_domains=n_domains,
            seed=count_seed,
            layout_seed=layout_seed,
        )
        _save_dataset(
            dataset,
            destination,
            {
                "simulation": 3,
                "setting": setting,
                "scaling_variable": (
                    "n" if n_domains == 5 else "K"
                ),
            },
        )
        print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study",
        choices=("simulation1", "simulation2", "simulation3", "all"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "simulations",
    )
    args = parser.parse_args()

    generators = {
        "simulation1": generate_simulation1,
        "simulation2": generate_simulation2,
        "simulation3": generate_simulation3,
    }
    selected = (
        tuple(generators)
        if args.study == "all"
        else (args.study,)
    )
    for study in selected:
        generators[study](args.output_root)


if __name__ == "__main__":
    main()
