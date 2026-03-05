"""
Data loading utilities.

Supports:
  1. Simulated gut scRNA-seq data (built-in, no download needed)
  2. Real 10x CellRanger output directory (barcodes.tsv.gz / features.tsv.gz / matrix.mtx.gz)
  3. Existing AnnData (.h5ad) file
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from .simulator import simulate_counts, MITO_GENES
from .markers import CELL_TYPES, ALL_MARKER_GENES

logger = logging.getLogger(__name__)


def load_simulated(
    n_cells: int = 3000,
    n_genes: int = 2000,
    random_seed: int = 42,
) -> ad.AnnData:
    """
    Generate a realistic simulated gut scRNA-seq AnnData object.

    The count data is parameterised by real published gut cell type marker
    genes from Smillie et al. 2019 (Nature) and Elmentaite et al. 2021 (Nature).

    Returns
    -------
    AnnData with:
      .X          — raw count matrix (CSR sparse)
      .obs['cell_type'] — true cell type label
      .obs['compartment'] — Epithelial / Immune / Stromal
      .var['is_marker'] — bool: is this a published marker gene?
      .var['mt'] — bool: is this a mitochondrial gene?
    """
    logger.info("Generating simulated gut scRNA-seq data …")
    counts, cell_ids, gene_names, cell_labels = simulate_counts(
        n_cells=n_cells, n_genes=n_genes, random_seed=random_seed
    )

    # Compartment mapping
    compartment_map = {}
    from .markers import COMPARTMENTS
    for comp, cts in COMPARTMENTS.items():
        for ct in cts:
            compartment_map[ct] = comp

    obs = pd.DataFrame(
        {
            "cell_type":   cell_labels,
            "compartment": [compartment_map.get(l, "Unknown") for l in cell_labels],
        },
        index=cell_ids,
    )
    var = pd.DataFrame(
        {
            "is_marker": [g in ALL_MARKER_GENES for g in gene_names],
            "mt":        [g in MITO_GENES or g.startswith("MT-") for g in gene_names],
        },
        index=gene_names,
    )

    adata = ad.AnnData(
        X   = csr_matrix(counts),
        obs = obs,
        var = var,
    )
    adata.uns["data_source"] = "simulated — Smillie 2019 + Elmentaite 2021 marker genes"
    logger.info("AnnData: %s", adata)
    return adata


def load_10x_cellranger(directory: str | Path) -> ad.AnnData:
    """
    Load a 10x CellRanger output directory.

    Expected structure:
        <directory>/
            barcodes.tsv.gz   (or barcodes.tsv)
            features.tsv.gz   (or genes.tsv)
            matrix.mtx.gz     (or matrix.mtx)

    Parameters
    ----------
    directory : Path to CellRanger filtered_feature_bc_matrix directory.

    Returns
    -------
    AnnData with raw counts in .X
    """
    directory = Path(directory)
    logger.info("Loading 10x CellRanger data from: %s", directory)
    adata = sc.read_10x_mtx(str(directory), var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    logger.info("Loaded: %s", adata)
    return adata


def load_h5ad(path: str | Path) -> ad.AnnData:
    """
    Load an existing AnnData .h5ad file.

    Parameters
    ----------
    path : Path to .h5ad file.
    """
    path = Path(path)
    logger.info("Loading AnnData from: %s", path)
    adata = ad.read_h5ad(str(path))
    if "mt" not in adata.var.columns:
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
    logger.info("Loaded: %s", adata)
    return adata
