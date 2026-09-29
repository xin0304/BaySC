# Data layout

The raw datasets are not redistributed with this code package. Obtain them
from the sources cited in the accompanying manuscript and place them under
`data/raw/` using the layout below. Manual annotations are used only for
post-fitting evaluation in the supplied scripts.

## Human breast cancer (10x Visium)

```text
data/raw/Breast cancer/
  V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5
  metadata.tsv
  spatial/
    tissue_positions_list.csv
```

## HER2-positive breast cancer

For each section `S` in `A1`, `B1`, `C1`, `D1`, `E1`, `F1`, `G2`, and `H1`:

```text
data/raw/S_sample/
  S.tsv.gz
  S_labeled_coordinates.tsv
  S_selection.tsv
```

## STARmap mouse visual cortex

```text
data/raw/STARmap/
  STARmap_20180505_BY3_1k.h5ad
```

## MISAR-seq mouse E15 brain

```text
data/raw/
  MISAR_seq_mouse_E15_brain_mRNA_data.h5
  MISAR_seq_mouse_E15_brain_ATAC_data.h5
```

## Human lymph node spatial CITE-seq

```text
data/raw/Dataset11_Human_Lymph_Node_A1/
  adata_RNA.h5ad
  adata_ADT.h5ad
  annotation.csv
```

The preprocessing scripts write derived embeddings and metadata to
`data/processed/<dataset>/`. These generated files are ignored by Git.
