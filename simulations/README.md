# Simulation data generation

This directory reproduces the paired RNA--ATAC data used in the three BaySC
simulation studies. Data generation is independent of BaySC fitting and model
selection. True labels are written to each generated dataset for evaluation,
but are not supplied to the clustering algorithm.

## Output format

Each simulated dataset is stored in a separate directory containing:

- `dataset.npz`: RNA counts, ATAC counts, coordinates, true labels, the true
  boundary mask, and distance to the nearest true boundary;
- `metadata.json`: the generating parameters, condition name, and random seed.

The arrays in `dataset.npz` are named `rna_counts`, `atac_counts`,
`coordinates`, `labels`, `boundary_mask`, and `distance_to_boundary`.

## Simulation 1

Simulation 1 contains four layouts and five independent replicates per layout
(`n=500`, `K=5`; 20 datasets in total):

- irregular contiguous compartments;
- a nested microenvironment;
- disconnected regions sharing the same label;
- gradual/curved transition boundaries represented by ordered spatial layers.

Generate all Simulation 1 datasets from the repository root with:

```bash
python -m simulations.generate simulation1
```

## Simulation 2

Simulation 2 uses the irregular layout and combines three signal levels with
three modality-quality settings. Each of the nine conditions has five
independent count realizations. Within a replicate, the same spatial layout is
used across the nine conditions.

```bash
python -m simulations.generate simulation2
```

The signal levels are `low=0.60`, `medium=0.80`, and `high=1.10`. The RNA/ATAC
quality multipliers are `(1,1)`, `(1,0.75)`, and `(0.75,1)` for balanced,
RNA-dominant, and ATAC-dominant conditions, respectively.

## Simulation 3

Simulation 3 generates the seven datasets used to evaluate scaling with the
number of spatial locations and the number of true domains:

```bash
python -m simulations.generate simulation3
```

The settings are `n=500,1000,2000,4000` with `K=5`, and `K=3,5,7,10` with
`n=1000`. The shared `n=1000, K=5` setting is generated once.

To generate all three studies, run:

```bash
python -m simulations.generate all
```

The default output location is `data/simulations`. The command refuses to
overwrite an existing dataset directory so that a completed simulation cannot
be replaced silently. Use `--output-root PATH` to write to another location.

## Data-generating mechanism

RNA counts follow feature-specific Poisson distributions and ATAC counts
follow feature-specific Bernoulli distributions. Each true domain has disjoint
RNA and ATAC marker sets. The modality-specific domain effects are deliberately
complementary: one domain is more informative in RNA, one in ATAC, and one has
moderate evidence in both modalities. Spot-specific depth factors, artificial
dropout, and boundary mixing are not added.

All seeds and condition-specific settings are fixed in `generate.py`.
