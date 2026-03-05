"""
main.py — scRNA-seq Gut Cell Clustering Pipeline
=================================================

Usage
-----
# Simulated data (built-in, no download):
    python -m src.main

# Real 10x CellRanger data:
    python -m src.main --cellranger-dir /path/to/filtered_feature_bc_matrix

# Existing .h5ad:
    python -m src.main --h5ad my_data.h5ad
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import scanpy as sc

from src.data       import load_simulated, load_10x_cellranger, load_h5ad
from src.models     import (run_qc, preprocess, build_neighbors_and_umap,
                             cluster, annotate_cell_types, run_marker_genes)
from src.visualization import (
    set_style, plot_qc_violin, plot_pca_variance, plot_pca_scatter,
    plot_umap, plot_umap_markers, plot_cluster_composition,
    plot_marker_heatmap, plot_cell_annotation_umap, plot_compartment_summary,
)
from src.utils import load_config, setup_logging
from src.models.model_io import save_models, load_models

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="scRNA-seq Gut Cell Clustering")
    p.add_argument("--config",         default="configs/config.yaml")
    p.add_argument("--cellranger-dir", default=None,
                   help="Path to 10x CellRanger output directory")
    p.add_argument("--h5ad",           default=None,
                   help="Path to existing .h5ad file")
    p.add_argument("--out-dir",        default=None)
    return p.parse_args(argv)


def main(argv=None) -> ad.AnnData:
    args = parse_args(argv)
    cfg  = load_config(args.config)
    setup_logging(**cfg.get("logging", {}))
    set_style()

    out_dir = Path(args.out_dir or cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # ── 1. LOAD DATA ─────────────────────────────────────────────────────────
    if args.cellranger_dir:
        logger.info("Loading real 10x CellRanger data …")
        adata = load_10x_cellranger(args.cellranger_dir)
    elif args.h5ad:
        logger.info("Loading .h5ad file …")
        adata = load_h5ad(args.h5ad)
    else:
        logger.info("Using built-in simulated gut scRNA-seq data …")
        adata = load_simulated(
            n_cells     = cfg["data"]["n_cells"],
            n_genes     = cfg["data"]["n_genes"],
            random_seed = cfg["data"]["random_seed"],
        )
    logger.info("Loaded: %s", adata)

    # ── 2. QC ────────────────────────────────────────────────────────────────
    adata = run_qc(adata, cfg)
    plot_qc_violin(adata, out_dir)

    # ── 3. PREPROCESSING ─────────────────────────────────────────────────────
    adata = preprocess(adata, cfg)
    plot_pca_variance(adata, out_dir)
    plot_pca_scatter(adata, out_dir)

    # ── 4. NEIGHBOURS + UMAP ─────────────────────────────────────────────────
    adata = build_neighbors_and_umap(adata, cfg)

    # ── 5. CLUSTERING ────────────────────────────────────────────────────────
    adata = cluster(adata, cfg)

    # ── 6. CELL TYPE ANNOTATION ───────────────────────────────────────────────
    adata = annotate_cell_types(adata, cfg)

    # ── 7. MARKER GENE DISCOVERY ─────────────────────────────────────────────
    # Pop sklearn objects NOW — before rank_genes_groups writes to adata.uns,
    # and before h5ad save (h5py cannot serialise sklearn objects).
    _kmeans_to_save = adata.uns.pop("_kmeans_model", None)
    _pca_to_save    = adata.uns.pop("_pca_model", None)

    marker_df = run_marker_genes(adata)
    marker_df.to_csv(out_dir / "marker_genes_per_cluster.csv", index=False)
    logger.info("Marker genes saved: %s", out_dir / "marker_genes_per_cluster.csv")

    # ── 8. VISUALISATIONS ────────────────────────────────────────────────────
    plot_umap(adata, out_dir)
    plot_umap_markers(adata, out_dir)
    plot_cluster_composition(adata, out_dir)
    plot_marker_heatmap(adata, out_dir)
    plot_cell_annotation_umap(adata, out_dir)
    plot_compartment_summary(adata, out_dir)

    # ── 9. SAVE TRAINED MODELS ────────────────────────────────────────────────
    if _kmeans_to_save is not None and _pca_to_save is not None:
        save_models(adata, _kmeans_to_save, _pca_to_save, cfg, out_dir)
        logger.info("Models saved → %s/models/", out_dir)
    else:
        logger.warning("sklearn models not found — skipping model save")

    # ── 10. SAVE ANNDATA ──────────────────────────────────────────────────────
    if cfg["output"].get("save_h5ad", True):
        h5ad_path = out_dir / "gut_scrna_clustered.h5ad"
        adata.__dict__["_uns"].pop("log1p", None)
        adata.write_h5ad(str(h5ad_path))
        logger.info("AnnData saved: %s", h5ad_path)

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    n_leiden   = adata.obs["leiden"].nunique()
    ari        = adata.uns.get("ARI_leiden", float("nan"))
    sil_leiden = adata.uns.get("silhouette_leiden", float("nan"))
    sil_km     = adata.uns.get("silhouette_kmeans", float("nan"))

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  Day 20 — scRNA-seq Gut Cell Clustering  ✅  COMPLETE           ║
╠══════════════════════════════════════════════════════════════════╣
║  Data Source : Smillie 2019 + Elmentaite 2021 marker genes      ║
║  Cells       : {adata.n_obs:<49} ║
║  Genes       : {adata.n_vars:<49} ║
║  PCA comps   : {adata.obsm["X_pca"].shape[1]:<49} ║
╠══════════════════════════════════════════════════════════════════╣
║  Leiden clusters       : {n_leiden:<40} ║
║  Adj. Rand Index       : {ari:<40.3f} ║
║  Silhouette (Leiden)   : {sil_leiden:<40.3f} ║
║  Silhouette (K-Means)  : {sil_km:<40.3f} ║
╠══════════════════════════════════════════════════════════════════╣
║  Outputs saved to: {str(out_dir):<45} ║
╚══════════════════════════════════════════════════════════════════╝
""")
    return adata


if __name__ == "__main__":
    main()
