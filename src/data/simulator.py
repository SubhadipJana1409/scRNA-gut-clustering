"""
Realistic scRNA-seq count data simulator.

Generates synthetic single-cell gene expression data that mirrors the
statistical properties of real gut mucosal scRNA-seq datasets:

  - Negative Binomial (NB) count distribution  (standard for scRNA-seq)
  - Sparse expression (95–99% zeros for most genes)
  - Cell-type-specific marker gene enrichment (from published markers)
  - Realistic per-cell library size variation (~500 – 10,000 UMIs)
  - Dropout (gene not detected even if expressed) — key scRNA artifact

Data biology: Smillie et al. 2019 (Nature), Elmentaite et al. 2021 (Nature)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .markers import CELL_TYPES, CELL_TYPE_PROPORTIONS, ALL_MARKER_GENES

logger = logging.getLogger(__name__)

# ── Background gene universe ─────────────────────────────────────────────────
# Additional non-marker genes to pad to realistic panel size
_BACKGROUND_GENES = [
    # Housekeeping
    "ACTB", "GAPDH", "B2M", "MALAT1", "NEAT1", "RPS27A", "RPL13A",
    "RPS3", "RPL10", "RPS18", "RPL32", "RPS14", "RPL26", "EEF1A1",
    "TUBA1B", "TUBB", "NPM1", "HSP90AB1", "HSPA8", "HSPD1",
    # Transcription factors (pan-expressed)
    "KLF6", "EGR1", "FOS", "JUN", "ATF3", "CEBPB", "STAT3", "NFKB1",
    # Mitochondrial genes (QC marker)
    "MT-CO1", "MT-CO2", "MT-CO3", "MT-ND1", "MT-ND2", "MT-ND4",
    "MT-ND5", "MT-ATP6", "MT-CYB", "MT-RNR2",
    # Cell cycle
    "CCNA2", "CCNB2", "CCND1", "CCNE1", "CDK2", "CDK4", "CDK6",
    "E2F1", "RB1", "TP53",
    # Ribosomal
    *[f"RPS{i}" for i in range(4, 30)],
    *[f"RPL{i}" for i in range(3, 40)],
    # Random filler
    *[f"GENE{i:04d}" for i in range(1, 200)],
]

MITO_GENES = [g for g in _BACKGROUND_GENES if g.startswith("MT-")]


def _neg_binom_sample(mean: float, dispersion: float, size: int) -> np.ndarray:
    """Sample from a Negative Binomial distribution (NB parameterisation)."""
    p = dispersion / (dispersion + mean) if (dispersion + mean) > 0 else 1.0
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.random.negative_binomial(dispersion, p, size=size).astype(np.float32)


def build_gene_panel(n_total_genes: int = 2000) -> list[str]:
    """
    Build gene panel = all marker genes + background genes up to n_total_genes.
    """
    panel = list(ALL_MARKER_GENES)
    for g in _BACKGROUND_GENES:
        if g not in panel:
            panel.append(g)
        if len(panel) >= n_total_genes:
            break
    # Pad with generic genes if still short
    i = 0
    while len(panel) < n_total_genes:
        g = f"NOVEL{i:05d}"
        panel.append(g)
        i += 1
    return panel[:n_total_genes]


def simulate_counts(
    n_cells: int = 3000,
    n_genes: int = 2000,
    random_seed: int = 42,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """
    Generate realistic scRNA-seq count matrix.

    Parameters
    ----------
    n_cells     : Total number of cells.
    n_genes     : Number of genes in the panel.
    random_seed : NumPy random seed.

    Returns
    -------
    counts      : (n_cells, n_genes) integer count matrix (dense).
    cell_ids    : List of cell barcodes.
    gene_names  : List of gene names.
    cell_labels : True cell type label per cell.
    """
    np.random.seed(random_seed)
    gene_names = build_gene_panel(n_genes)
    gene_idx   = {g: i for i, g in enumerate(gene_names)}

    # Assign cells to cell types by proportion
    cell_type_names = list(CELL_TYPE_PROPORTIONS.keys())
    props = np.array([CELL_TYPE_PROPORTIONS[ct] for ct in cell_type_names])
    props = props / props.sum()
    cell_counts_per_type = np.floor(props * n_cells).astype(int)
    diff = n_cells - cell_counts_per_type.sum()
    cell_counts_per_type[0] += diff   # put remainder in stem cells

    logger.info("Cell type distribution:")
    for ct, n in zip(cell_type_names, cell_counts_per_type):
        logger.info("  %-30s %4d cells", ct, n)

    # Background expression: low NB for all genes
    BG_MEAN  = 0.05
    BG_DISP  = 0.1
    # Library size: log-normal ~ 2000 UMIs median
    lib_sizes = np.random.lognormal(mean=7.5, sigma=0.5, size=n_cells).astype(int)
    lib_sizes = np.clip(lib_sizes, 300, 15000)

    count_matrix  = np.zeros((n_cells, n_genes), dtype=np.int32)
    cell_labels   = []
    cell_ids      = []
    cell_pointer  = 0

    for ct_name, n_ct in zip(cell_type_names, cell_counts_per_type):
        if n_ct == 0:
            continue
        ct_info = CELL_TYPES[ct_name]

        for _ in range(n_ct):
            cell_id = f"CELL_{cell_pointer:05d}"
            cell_ids.append(cell_id)
            cell_labels.append(ct_name)

            # Background expression for all genes (sparse)
            row = _neg_binom_sample(BG_MEAN, BG_DISP, n_genes)

            # Marker gene enrichment — draw from higher-mean NB
            for gene in ct_info["markers_up"]:
                if gene in gene_idx:
                    g_idx = gene_idx[gene]
                    marker_mean = np.random.uniform(3.0, 12.0)
                    row[g_idx] = _neg_binom_sample(marker_mean, 0.5, 1)[0]

            # Suppressed genes
            for gene in ct_info.get("markers_down", []):
                if gene in gene_idx:
                    row[gene_idx[gene]] = 0.0

            # Scale to realistic library size
            row_sum = row.sum()
            if row_sum > 0:
                pvals = row.astype(np.float64) / float(row_sum)
                pvals = pvals / pvals.sum()       # renormalise to sum=1
                pvals = np.clip(pvals, 0.0, 1.0)
                pvals[-1] = max(0.0, 1.0 - float(pvals[:-1].sum()))
                row = np.random.multinomial(int(lib_sizes[cell_pointer]), pvals).astype(np.int32)

            # Apply dropout (cells randomly fail to capture low-expressed genes)
            # P(dropout) increases as expression decreases (logistic model)
            dropout_prob = np.exp(-0.5 * row.astype(float))
            dropout_mask = np.random.random(n_genes) < dropout_prob
            row[dropout_mask] = 0

            count_matrix[cell_pointer] = row
            cell_pointer += 1

    logger.info(
        "Simulated %d cells × %d genes | Sparsity: %.1f%%",
        n_cells, n_genes,
        100.0 * (count_matrix == 0).sum() / count_matrix.size,
    )
    return count_matrix, cell_ids, gene_names, cell_labels
