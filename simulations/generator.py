"""Generate paired RNA/ATAC counts with known spatial-domain labels.

This module contains only the data-generating mechanism used by the three
simulation studies.  It is independent of BaySC fitting, parameter selection,
and evaluation code.  The labels returned here are the simulation truth and
are not used by the BaySC fitting procedure.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree


GENERATOR_VERSION = "bridge_sparse_complementary_counts_v1"
SIMULATION2_GENERATOR_VERSION = "bridge_sparse_complementary_counts_simulation2_v1"

RNA_FEATURES = 2000
ATAC_FEATURES = 4000
RNA_MARKER_FRACTION = 0.20
ATAC_MARKER_FRACTION = 0.16
RNA_BASE_SHAPE = 1.5
RNA_BASE_SCALE = 0.45
RNA_BASE_OFFSET = 0.015
ATAC_TARGET_SPARSITY = 0.90
ATAC_FEATURE_OFFSET_SD = 0.55
ATAC_EFFECT_MULTIPLIER = 2.2
WEAK_MODALITY_STRENGTH = 0.50
JOINT_MODERATE_STRENGTH = 0.70


@dataclass(frozen=True)
class SpatialLayout:
    coordinates: np.ndarray
    labels: np.ndarray
    boundary_mask: np.ndarray
    distance_to_boundary: np.ndarray


@dataclass(frozen=True)
class SimulatedDataset:
    rna_counts: np.ndarray
    atac_counts: np.ndarray
    coordinates: np.ndarray
    labels: np.ndarray
    boundary_mask: np.ndarray
    distance_to_boundary: np.ndarray
    metadata: Dict[str, object]


def _jittered_grid(n_spots: int, rng: np.random.Generator) -> np.ndarray:
    side = int(np.ceil(np.sqrt(n_spots)))
    axis = np.linspace(0.0, 1.0, side)
    xx, yy = np.meshgrid(axis, axis)
    coordinates = np.column_stack((xx.ravel(), yy.ravel()))[:n_spots]
    jitter = rng.normal(
        0.0,
        0.10 / max(side - 1, 1),
        size=coordinates.shape,
    )
    return np.clip(coordinates + jitter, 0.0, 1.0)


def _irregular_labels(
    coordinates: np.ndarray,
    n_domains: int,
    rng: np.random.Generator,
) -> np.ndarray:
    angle = 2.0 * np.pi * (np.arange(n_domains) + 0.35) / n_domains
    radius = 0.28 + 0.08 * rng.uniform(-1.0, 1.0, size=n_domains)
    centers = np.column_stack(
        (0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle))
    )
    centers += rng.normal(0.0, 0.025, size=centers.shape)

    x_coord, y_coord = coordinates[:, 0], coordinates[:, 1]
    warped = np.column_stack(
        (
            x_coord + 0.055 * np.sin(4.0 * np.pi * y_coord),
            y_coord + 0.045 * np.sin(3.0 * np.pi * x_coord + 0.4),
        )
    )
    scale = rng.uniform(0.85, 1.20, size=(n_domains, 2))
    distances = np.empty((len(coordinates), n_domains), dtype=np.float64)
    for domain in range(n_domains):
        distances[:, domain] = np.sum(
            ((warped - centers[domain]) / scale[domain]) ** 2,
            axis=1,
        )
    return np.argmin(distances, axis=1).astype(np.int32)


def _transition_labels(coordinates: np.ndarray, n_domains: int) -> np.ndarray:
    x_coord, y_coord = coordinates[:, 0], coordinates[:, 1]
    score = (
        x_coord
        + 0.13 * np.sin(2.0 * np.pi * y_coord + 0.25)
        + 0.035 * np.sin(6.0 * np.pi * y_coord - 0.40)
    )
    cuts = np.quantile(
        score,
        np.linspace(0.0, 1.0, n_domains + 1)[1:-1],
    )
    return np.digitize(score, cuts, right=False).astype(np.int32)


def _nested_labels(coordinates: np.ndarray, n_domains: int) -> np.ndarray:
    x_coord, y_coord = coordinates[:, 0], coordinates[:, 1]
    radius = np.sqrt(
        ((x_coord - 0.47) / 1.05) ** 2
        + ((y_coord - 0.52) / 0.86) ** 2
    )
    radius += 0.025 * np.sin(
        8.0 * np.arctan2(y_coord - 0.52, x_coord - 0.47)
    )
    labels = np.full(
        len(coordinates), min(n_domains - 1, 4), dtype=np.int32
    )
    labels[radius < 0.48] = 1 % n_domains
    labels[radius < 0.25] = 0
    if n_domains >= 4:
        vascular_band = (
            np.abs(y_coord - (0.20 + 0.48 * x_coord)) < 0.150
        ) & (radius > 0.27)
        labels[vascular_band] = 3
        immune_niche = (
            (x_coord - 0.72) ** 2 + (y_coord - 0.69) ** 2 < 0.160**2
        ) | (
            (x_coord - 0.29) ** 2 + (y_coord - 0.73) ** 2 < 0.150**2
        )
        labels[immune_niche] = 2
    if n_domains > 5:
        exterior = radius >= 0.39
        theta = (
            np.arctan2(y_coord - 0.52, x_coord - 0.47) + np.pi
        ) / (2.0 * np.pi)
        extra = 4 + np.floor(theta * (n_domains - 4)).astype(int)
        labels[exterior] = np.minimum(extra[exterior], n_domains - 1)
    return labels


def _island_labels(coordinates: np.ndarray, n_domains: int) -> np.ndarray:
    x_coord, y_coord = coordinates[:, 0], coordinates[:, 1]
    island = (
        (x_coord - 0.25) ** 2 + (y_coord - 0.28) ** 2 < 0.15**2
    ) | (
        (x_coord - 0.75) ** 2 + (y_coord - 0.72) ** 2 < 0.15**2
    )
    labels = np.zeros(len(coordinates), dtype=np.int32)
    theta = (
        np.arctan2(y_coord - 0.5, x_coord - 0.5) + np.pi
    ) / (2.0 * np.pi)
    labels[~island] = 1 + np.minimum(
        np.floor(theta[~island] * max(n_domains - 1, 1)).astype(int),
        n_domains - 2,
    )
    return labels


def _ensure_all_domains(
    labels: np.ndarray,
    coordinates: np.ndarray,
    n_domains: int,
) -> np.ndarray:
    labels = labels.copy()
    missing = sorted(
        set(range(n_domains)) - set(np.unique(labels).tolist())
    )
    if not missing:
        return labels
    order = np.argsort(coordinates[:, 0] + 1.7 * coordinates[:, 1])
    chunks = np.array_split(order, n_domains)
    for domain in missing:
        labels[chunks[domain]] = domain
    return labels


def _boundary_summary(
    coordinates: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(coordinates)
    _, neighbors = tree.query(coordinates, k=min(7, len(coordinates)))
    neighbors = np.atleast_2d(neighbors)
    boundary = np.array(
        [
            np.any(labels[row[1:]] != labels[index])
            for index, row in enumerate(neighbors)
        ],
        dtype=bool,
    )
    if not np.any(boundary):
        return boundary, np.full(len(coordinates), np.inf)
    boundary_tree = cKDTree(coordinates[boundary])
    distance, _ = boundary_tree.query(coordinates, k=1)
    return boundary, distance


def generate_layout(
    layout: str,
    n_spots: int,
    n_domains: int,
    seed: int,
) -> SpatialLayout:
    """Generate one of the four prespecified spatial layouts."""
    if n_domains < 2:
        raise ValueError("n_domains must be at least two")
    rng = np.random.default_rng(seed)
    coordinates = _jittered_grid(n_spots, rng)
    if layout == "irregular_compartments":
        labels = _irregular_labels(coordinates, n_domains, rng)
    elif layout == "transition_boundaries":
        labels = _transition_labels(coordinates, n_domains)
    elif layout == "nested_microenvironment":
        labels = _nested_labels(coordinates, n_domains)
    elif layout == "disconnected_islands":
        labels = _island_labels(coordinates, n_domains)
    else:
        raise ValueError("unknown spatial layout: {}".format(layout))
    labels = _ensure_all_domains(labels, coordinates, n_domains)
    boundary, distance = _boundary_summary(coordinates, labels)
    return SpatialLayout(coordinates, labels, boundary, distance)


def _marker_blocks(
    n_features: int,
    n_domains: int,
    total_fraction: float,
) -> Tuple[np.ndarray, ...]:
    markers_per_domain = max(
        5, int(n_features * float(total_fraction) / n_domains)
    )
    usable = min(n_features, markers_per_domain * n_domains)
    return tuple(
        np.asarray(block, dtype=np.int32)
        for block in np.array_split(
            np.arange(usable, dtype=np.int32), n_domains
        )
    )


def _domain_modality_strengths(
    n_domains: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rna = np.ones(n_domains, dtype=np.float64)
    atac = np.ones(n_domains, dtype=np.float64)
    if n_domains >= 2:
        atac[1] = WEAK_MODALITY_STRENGTH
    if n_domains >= 3:
        rna[2] = WEAK_MODALITY_STRENGTH
    if n_domains >= 5:
        rna[4] = JOINT_MODERATE_STRENGTH
        atac[4] = JOINT_MODERATE_STRENGTH
    return rna, atac


def _logit(probability: float) -> float:
    probability = float(np.clip(probability, 1e-8, 1.0 - 1e-8))
    return float(np.log(probability / (1.0 - probability)))


def simulate_paired_counts(
    layout: str,
    log_fold_change: float,
    n_spots: int = 500,
    n_domains: int = 5,
    seed: int = 18_000_000,
    *,
    rna_quality_multiplier: float = 1.0,
    atac_quality_multiplier: float = 1.0,
    layout_seed=None,
) -> SimulatedDataset:
    """Generate paired RNA and ATAC observations for one dataset."""
    if n_spots < n_domains:
        raise ValueError("n_spots must be at least n_domains")
    if n_domains < 2:
        raise ValueError("n_domains must be at least two")
    if log_fold_change <= 0.0:
        raise ValueError("log_fold_change must be positive")
    if rna_quality_multiplier <= 0.0 or atac_quality_multiplier <= 0.0:
        raise ValueError("modality quality multipliers must be positive")

    effective_layout_seed = (
        int(seed) + 17 if layout_seed is None else int(layout_seed)
    )
    separate_modality_streams = (
        layout_seed is not None
        or not np.isclose(rna_quality_multiplier, 1.0)
        or not np.isclose(atac_quality_multiplier, 1.0)
    )
    spatial = generate_layout(
        layout, n_spots, n_domains, effective_layout_seed
    )
    common_rng = np.random.default_rng(seed)
    rna_rng = (
        np.random.default_rng(int(seed) + 10_019)
        if separate_modality_streams
        else common_rng
    )
    atac_rng = (
        np.random.default_rng(int(seed) + 20_033)
        if separate_modality_streams
        else common_rng
    )
    rna_strength, atac_strength = _domain_modality_strengths(n_domains)

    rna_baseline = (
        rna_rng.gamma(
            shape=RNA_BASE_SHAPE,
            scale=RNA_BASE_SCALE,
            size=RNA_FEATURES,
        )
        + RNA_BASE_OFFSET
    )
    rna_effects = np.zeros((n_domains, RNA_FEATURES), dtype=np.float64)
    for domain, marker_ids in enumerate(
        _marker_blocks(RNA_FEATURES, n_domains, RNA_MARKER_FRACTION)
    ):
        rna_effects[domain, marker_ids] = (
            log_fold_change
            * rna_strength[domain]
            * rna_quality_multiplier
        )
    rna_means = rna_baseline[None, :] * np.exp(
        rna_effects[spatial.labels]
    )
    rna_counts = rna_rng.poisson(rna_means).astype(np.int32)

    atac_intercept = _logit(1.0 - ATAC_TARGET_SPARSITY)
    atac_feature_offsets = atac_rng.normal(
        0.0, ATAC_FEATURE_OFFSET_SD, size=ATAC_FEATURES
    )
    atac_effects = np.zeros((n_domains, ATAC_FEATURES), dtype=np.float64)
    for domain, marker_ids in enumerate(
        _marker_blocks(ATAC_FEATURES, n_domains, ATAC_MARKER_FRACTION)
    ):
        atac_effects[domain, marker_ids] = (
            ATAC_EFFECT_MULTIPLIER
            * log_fold_change
            * atac_strength[domain]
            * atac_quality_multiplier
        )
    atac_logits = (
        atac_intercept
        + atac_feature_offsets[None, :]
        + atac_effects[spatial.labels]
    )
    atac_probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(atac_logits, -20.0, 20.0))
    )
    atac_counts = (
        atac_rng.random(atac_probabilities.shape) < atac_probabilities
    ).astype(np.int8)

    metadata = {
        "generator_version": (
            SIMULATION2_GENERATOR_VERSION
            if separate_modality_streams
            else GENERATOR_VERSION
        ),
        "layout": layout,
        "seed": int(seed),
        "effective_layout_seed": effective_layout_seed,
        "n_spots": int(n_spots),
        "n_domains": int(n_domains),
        "log_fold_change": float(log_fold_change),
        "rna_features": RNA_FEATURES,
        "atac_features": ATAC_FEATURES,
        "rna_distribution": "Poisson",
        "atac_distribution": "Bernoulli",
        "rna_marker_fraction": RNA_MARKER_FRACTION,
        "atac_marker_fraction": ATAC_MARKER_FRACTION,
        "rna_base_shape": RNA_BASE_SHAPE,
        "rna_base_scale": RNA_BASE_SCALE,
        "rna_base_offset": RNA_BASE_OFFSET,
        "atac_target_sparsity": ATAC_TARGET_SPARSITY,
        "atac_feature_offset_sd": ATAC_FEATURE_OFFSET_SD,
        "atac_effect_multiplier": ATAC_EFFECT_MULTIPLIER,
        "weak_modality_strength": WEAK_MODALITY_STRENGTH,
        "joint_moderate_strength": JOINT_MODERATE_STRENGTH,
        "rna_quality_multiplier": float(rna_quality_multiplier),
        "atac_quality_multiplier": float(atac_quality_multiplier),
        "rna_domain_strengths": rna_strength.tolist(),
        "atac_domain_strengths": atac_strength.tolist(),
        "spot_library_factor": "fixed at one",
        "spot_atac_depth_offset": "fixed at zero",
        "dropout": 0.0,
        "boundary_mixing": 0.0,
        "rna_zero_fraction": float(np.mean(rna_counts == 0)),
        "atac_zero_fraction": float(np.mean(atac_counts == 0)),
        "truth_used_for_data_generation": True,
        "truth_used_for_parameter_selection": False,
    }
    return SimulatedDataset(
        rna_counts=rna_counts,
        atac_counts=atac_counts,
        coordinates=spatial.coordinates,
        labels=spatial.labels,
        boundary_mask=spatial.boundary_mask,
        distance_to_boundary=spatial.distance_to_boundary,
        metadata=metadata,
    )


__all__ = [
    "GENERATOR_VERSION",
    "SIMULATION2_GENERATOR_VERSION",
    "SimulatedDataset",
    "SpatialLayout",
    "generate_layout",
    "simulate_paired_counts",
]
