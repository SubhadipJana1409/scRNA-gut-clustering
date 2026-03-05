"""
scRNA-seq preprocessing and clustering pipeline.

Steps
-----
1. Quality Control  (per-cell: n_genes, n_counts, pct_mito)
2. Normalisation    (CPM-style, log1p)
3. HVG Selection    (highly variable genes)
4. Scaling          (zero-mean, unit variance — optional, standard for PCA)
5. PCA              (linear dimensionality reduction)
6. KNN Graph        (neighbourhood graph for Leiden/UMAP)
7. Clustering       (Leiden community detection + K-Means for comparison)
8. UMAP             (non-linear 2-D embedding for visualisation)
9. Cell Annotation  (marker gene scoring → automatic cell type labels)
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ..data.markers import CELL_TYPES, CELL_TYPE_COLORS

logger = logging.getLogger(__name__)
sc.settings.verbosity = 1


def run_qc(adata: ad.AnnData, cfg: dict) -> ad.AnnData:
    """
    Compute and apply quality-control filters.

    Filters cells by:
      - n_genes_by_counts: min / max number of genes detected
      - pct_counts_mt: maximum mitochondrial gene fraction

    Filters genes by:
      - n_cells: minimum number of cells expressing the gene
    """
    qc = cfg.get("qc", {})

    # Compute QC metrics
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    n_before = adata.n_obs
    # Cell filters
    min_genes = qc.get("min_genes_per_cell", 200)
    max_genes = qc.get("max_genes_per_cell", 5000)
    max_mito  = qc.get("max_pct_mito", 20.0)

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, max_genes=max_genes)
    adata = adata[adata.obs["pct_counts_mt"] <= max_mito].copy()

    # Gene filter
    min_cells = qc.get("min_cells_per_gene", 3)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    logger.info(
        "QC: %d → %d cells | %d genes retained",
        n_before, adata.n_obs, adata.n_vars,
    )
    return adata


def preprocess(adata: ad.AnnData, cfg: dict) -> ad.AnnData:
    """
    Normalise, log-transform, select HVGs, scale, and run PCA.
    """
    pp = cfg.get("preprocessing", {})

    # Store raw counts
    adata.layers["counts"] = adata.X.copy()

    # Normalise to target sum (CPM-like)
    sc.pp.normalize_total(adata, target_sum=pp.get("target_sum", 10_000))
    logger.info("Normalised to %d counts per cell", pp.get("target_sum", 10_000))

    # Log1p transform
    if pp.get("log1p", True):
        sc.pp.log1p(adata)
        logger.info("log1p transform applied")

    # HVG selection
    n_top = pp.get("n_top_genes", 2000)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat")
    n_hvg = adata.var["highly_variable"].sum()
    logger.info("Highly variable genes: %d / %d", n_hvg, adata.n_vars)

    # Scale to zero mean, unit variance (clipped at 10)
    sc.pp.scale(adata, max_value=10)

    # PCA
    n_pcs = pp.get("n_pcs", 50)
    sc.tl.pca(adata, n_comps=min(n_pcs, adata.n_obs - 1, adata.n_vars - 1))
    logger.info("PCA: %d components computed", adata.obsm["X_pca"].shape[1])

    # Build and store sklearn PCA for later serialisation via model_io
    from sklearn.decomposition import PCA as _PCA
    hvg_mask = adata.var["highly_variable"].values
    X_hvg    = adata.X[:, hvg_mask]
    if hasattr(X_hvg, "toarray"):
        X_hvg = X_hvg.toarray()
    n_fit = min(n_pcs, X_hvg.shape[0] - 1, X_hvg.shape[1] - 1)
    sklearn_pca = _PCA(n_components=n_fit, random_state=42)
    sklearn_pca.fit(X_hvg)
    adata.uns["_pca_model"] = sklearn_pca   # stash for model_io

    return adata


def build_neighbors_and_umap(adata: ad.AnnData, cfg: dict) -> ad.AnnData:
    """Compute KNN graph and UMAP embedding."""
    cl = cfg.get("clustering", {})
    n_neighbors = cl.get("n_neighbors", 15)
    n_umap      = cl.get("n_umap_components", 2)

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=40)
    sc.tl.umap(adata, n_components=n_umap)
    logger.info("KNN graph + UMAP computed (n_neighbors=%d)", n_neighbors)
    return adata


def cluster(adata: ad.AnnData, cfg: dict) -> ad.AnnData:
    """
    Cluster cells using Leiden (graph-based) and K-Means (centroid-based).
    """
    cl  = cfg.get("clustering", {})
    res = cl.get("leiden_resolution", 0.5)
    k   = cl.get("kmeans_k", 8)

    # Leiden clustering
    sc.tl.leiden(adata, resolution=res, key_added="leiden")
    n_leiden = adata.obs["leiden"].nunique()
    logger.info("Leiden clustering: %d clusters (resolution=%.2f)", n_leiden, res)

    # K-Means on PCA embedding
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    adata.obs["kmeans"] = km.fit_predict(adata.obsm["X_pca"]).astype(str)
    adata.uns["_kmeans_model"] = km   # stash for model_io
    logger.info("K-Means clustering: %d clusters", k)

    # Cluster quality metrics
    X_pca = adata.obsm["X_pca"]
    leiden_labels = adata.obs["leiden"].values
    km_labels     = adata.obs["kmeans"].values

    if len(set(leiden_labels)) > 1:
        sil_leiden = silhouette_score(X_pca, leiden_labels, sample_size=min(2000, len(X_pca)))
        sil_km     = silhouette_score(X_pca, km_labels,     sample_size=min(2000, len(X_pca)))
        logger.info("Silhouette score — Leiden: %.3f | K-Means: %.3f", sil_leiden, sil_km)
        adata.uns["silhouette_leiden"] = float(sil_leiden)
        adata.uns["silhouette_kmeans"] = float(sil_km)

    # If true labels available, compute ARI
    if "cell_type" in adata.obs.columns:
        ari = adjusted_rand_score(adata.obs["cell_type"], adata.obs["leiden"])
        logger.info("Adjusted Rand Index (Leiden vs true labels): %.3f", ari)
        adata.uns["ARI_leiden"] = float(ari)

    return adata


def annotate_cell_types(adata: ad.AnnData, cfg: dict) -> ad.AnnData:
    """
    Score each cell for each cell type using published marker genes,
    then assign a predicted cell type label.

    Uses scanpy's score_genes() which computes a z-score-based enrichment
    for each gene set vs. a random background set.
    """
    ann_cfg   = cfg.get("annotation", {})
    threshold = ann_cfg.get("min_score_threshold", 0.2)
    scores    = {}

    for ct_name, ct_info in CELL_TYPES.items():
        markers_present = [g for g in ct_info["markers_up"] if g in adata.var_names]
        if len(markers_present) < 2:
            continue
        score_key = f"score_{ct_name.replace(' ', '_').replace('/', '_').replace('+', 'pos')}"
        try:
            sc.tl.score_genes(adata, gene_list=markers_present, score_name=score_key)
            scores[ct_name] = score_key
        except Exception as e:
            logger.warning("Could not score %s: %s", ct_name, e)

    if scores:
        # Assign predicted label as the highest-scoring cell type
        score_matrix = adata.obs[[v for v in scores.values()]].values
        best_idx     = score_matrix.argmax(axis=1)
        best_score   = score_matrix.max(axis=1)
        ct_names     = list(scores.keys())

        adata.obs["predicted_cell_type"] = [
            ct_names[i] if best_score[j] > threshold else "Unassigned"
            for j, i in enumerate(best_idx)
        ]
        logger.info(
            "Predicted cell type distribution:\n%s",
            adata.obs["predicted_cell_type"].value_counts().to_string(),
        )

    return adata


def run_marker_genes(adata: ad.AnnData) -> pd.DataFrame:
    """
    Run Wilcoxon rank-sum test to find differentially expressed genes
    per Leiden cluster (de-novo marker discovery).

    Returns
    -------
    pd.DataFrame with top DEGs per cluster.
    """
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", pts=True)
    result   = adata.uns["rank_genes_groups"]
    clusters = list(result["names"].dtype.names)
    rows     = []
    for cl in clusters:
        for rank in range(min(10, len(result["names"][cl]))):
            rows.append({
                "cluster":  cl,
                "rank":     rank + 1,
                "gene":     result["names"][cl][rank],
                "score":    result["scores"][cl][rank],
                "pval_adj": result["pvals_adj"][cl][rank],
                "log2fc":   result["logfoldchanges"][cl][rank],
            })
    df = pd.DataFrame(rows)
    logger.info("Marker gene analysis done: %d clusters × top-10 genes", len(clusters))
    return df
