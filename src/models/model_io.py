"""
Model serialisation — save and reload all trained pipeline objects.

Saved artefacts
---------------
models/
├── kmeans_model.joblib          ← sklearn KMeans (fit on PCA embedding)
├── pca_model.joblib             ← sklearn PCA (fit on HVG expression matrix)
├── pipeline_config.json         ← full config + cluster labels + gene list
└── model_card.md                ← human-readable model card
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def save_models(
    adata: ad.AnnData,
    kmeans_model: KMeans,
    pca_model: PCA,
    cfg: dict,
    out_dir: Path,
) -> dict[str, Path]:
    """
    Persist all trained model objects and metadata.

    Parameters
    ----------
    adata       : Fully processed AnnData (post-cluster, post-annotation).
    kmeans_model: Trained sklearn KMeans instance.
    pca_model   : Trained sklearn PCA instance.
    cfg         : Pipeline config dict.
    out_dir     : Directory to save into (models/ sub-folder created automatically).

    Returns
    -------
    dict mapping artefact name → saved Path.
    """
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    # ── 1. KMeans ────────────────────────────────────────────────────────────
    km_path = model_dir / "kmeans_model.joblib"
    joblib.dump(kmeans_model, km_path)
    saved["kmeans"] = km_path
    logger.info("KMeans saved: %s", km_path)

    # ── 2. PCA ───────────────────────────────────────────────────────────────
    pca_path = model_dir / "pca_model.joblib"
    joblib.dump(pca_model, pca_path)
    saved["pca"] = pca_path
    logger.info("PCA saved: %s", pca_path)

    # ── 3. Pipeline config + cluster labels + HVG list ───────────────────────
    leiden_dist = adata.obs["leiden"].value_counts().to_dict()
    pred_dist   = {}
    if "predicted_cell_type" in adata.obs.columns:
        pred_dist = adata.obs["predicted_cell_type"].value_counts().to_dict()

    meta = {
        "saved_at":         datetime.utcnow().isoformat() + "Z",
        "data_source":      adata.uns.get("data_source", "unknown"),
        "n_cells":          int(adata.n_obs),
        "n_genes_total":    int(adata.n_vars),
        "n_hvgs":           int(adata.var.get("highly_variable", pd.Series(dtype=bool)).sum()),
        "n_pca_components": int(adata.obsm["X_pca"].shape[1]),
        "n_leiden_clusters":adata.obs["leiden"].nunique(),
        "ARI_leiden":       float(adata.uns.get("ARI_leiden", float("nan"))),
        "silhouette_leiden":float(adata.uns.get("silhouette_leiden", float("nan"))),
        "silhouette_kmeans":float(adata.uns.get("silhouette_kmeans", float("nan"))),
        "leiden_distribution":  {str(k): int(v) for k, v in leiden_dist.items()},
        "predicted_cell_types": {str(k): int(v) for k, v in pred_dist.items()},
        "kmeans_k":         int(kmeans_model.n_clusters),
        "kmeans_inertia":   float(kmeans_model.inertia_),
        "pipeline_config":  cfg,
        "hvg_list":         [g for g, hv in zip(
                                 adata.var_names,
                                 adata.var.get("highly_variable", [False] * adata.n_vars)
                             ) if hv],
    }

    config_path = model_dir / "pipeline_config.json"
    with open(config_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    saved["config"] = config_path
    logger.info("Pipeline config saved: %s", config_path)

    # ── 4. Model card (Markdown) ──────────────────────────────────────────────
    card = _build_model_card(meta, kmeans_model, pca_model)
    card_path = model_dir / "model_card.md"
    card_path.write_text(card)
    saved["model_card"] = card_path
    logger.info("Model card saved: %s", card_path)

    return saved


def load_models(model_dir: str | Path) -> dict:
    """
    Reload all saved pipeline artefacts.

    Parameters
    ----------
    model_dir : Path to the models/ directory.

    Returns
    -------
    dict with keys: kmeans, pca, config, hvg_list

    Example
    -------
    >>> artefacts = load_models("outputs/models")
    >>> kmeans = artefacts["kmeans"]
    >>> pca    = artefacts["pca"]
    >>> # Predict cluster for new cells (already PCA-embedded):
    >>> labels = kmeans.predict(new_X_pca)
    """
    model_dir = Path(model_dir)

    km_path  = model_dir / "kmeans_model.joblib"
    pca_path = model_dir / "pca_model.joblib"
    cfg_path = model_dir / "pipeline_config.json"

    missing = [p for p in [km_path, pca_path, cfg_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")

    kmeans_model = joblib.load(km_path)
    pca_model    = joblib.load(pca_path)

    with open(cfg_path) as f:
        config = json.load(f)

    logger.info("Loaded KMeans (k=%d) from %s", kmeans_model.n_clusters, km_path)
    logger.info("Loaded PCA (%d components) from %s", pca_model.n_components_, pca_path)

    return {
        "kmeans":   kmeans_model,
        "pca":      pca_model,
        "config":   config,
        "hvg_list": config.get("hvg_list", []),
    }


def predict_new_cells(
    new_counts: np.ndarray,
    gene_names: list[str],
    artefacts: dict,
) -> np.ndarray:
    """
    Assign cluster labels to new cells using saved KMeans + PCA models.

    Parameters
    ----------
    new_counts  : (n_new_cells, n_genes) raw count matrix.
    gene_names  : Gene names corresponding to columns of new_counts.
    artefacts   : Output of load_models().

    Returns
    -------
    cluster_labels : (n_new_cells,) integer cluster assignments.

    Notes
    -----
    Applies the same HVG selection, normalisation, and PCA projection
    that were used during training. The counts must cover the HVG gene
    set used during training (missing genes are zero-filled).
    """
    hvg_list = artefacts["hvg_list"]
    pca      = artefacts["pca"]
    kmeans   = artefacts["kmeans"]

    gene_to_col = {g: i for i, g in enumerate(gene_names)}

    # Select / zero-fill HVG columns
    X = np.zeros((new_counts.shape[0], len(hvg_list)), dtype=np.float32)
    for j, gene in enumerate(hvg_list):
        if gene in gene_to_col:
            X[:, j] = new_counts[:, gene_to_col[gene]]

    # Normalise (CPM) + log1p
    X        = X.astype(np.float64)
    row_sums = X.sum(axis=1, keepdims=True) + 1e-6
    X        = X / row_sums * 10_000
    X        = np.log1p(X)

    # PCA projection — cast to match KMeans centroid dtype
    X_pca = pca.transform(X)
    centers_dtype = kmeans.cluster_centers_.dtype
    X_pca = X_pca.astype(centers_dtype)

    # KMeans prediction
    labels = kmeans.predict(X_pca)
    logger.info("Predicted %d cells → %d unique clusters", len(labels), len(set(labels)))
    return labels


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_model_card(meta: dict, kmeans: KMeans, pca: PCA) -> str:
    pred_table = "\n".join(
        f"| {ct} | {n} |"
        for ct, n in sorted(meta["predicted_cell_types"].items(), key=lambda x: -x[1])
    )
    leiden_table = "\n".join(
        f"| {cl} | {n} |"
        for cl, n in sorted(meta["leiden_distribution"].items(), key=lambda x: int(x[0]))
    )
    return f"""# Model Card — scRNA-seq Gut Cell Clustering

**Saved:** {meta["saved_at"]}
**Data source:** {meta["data_source"]}

---

## Model Summary

| Property | Value |
|----------|-------|
| Cells | {meta["n_cells"]:,} |
| Total genes | {meta["n_genes_total"]:,} |
| Highly variable genes | {meta["n_hvgs"]:,} |
| PCA components | {meta["n_pca_components"]} |
| Leiden clusters | {meta["n_leiden_clusters"]} |
| K-Means k | {meta["kmeans_k"]} |
| KMeans inertia | {meta["kmeans_inertia"]:.1f} |
| Adjusted Rand Index | {meta["ARI_leiden"]:.3f} |
| Silhouette (Leiden) | {meta["silhouette_leiden"]:.3f} |
| Silhouette (K-Means) | {meta["silhouette_kmeans"]:.3f} |

---

## Trained Models

| File | Type | Description |
|------|------|-------------|
| `kmeans_model.joblib` | sklearn KMeans | Cluster centroids in PCA space |
| `pca_model.joblib` | sklearn PCA | Fitted PCA transformation |
| `pipeline_config.json` | JSON | Full config + HVG list + cluster labels |

---

## Leiden Cluster Distribution

| Cluster | Cells |
|---------|-------|
{leiden_table}

---

## Predicted Cell Types

| Cell Type | Cells |
|-----------|-------|
{pred_table}

---

## Intended Use

- Cluster gut mucosal scRNA-seq data into biologically meaningful cell populations.
- Use `predict_new_cells()` to assign cluster labels to new unseen cells.
- Load in any scanpy session via `load_models("outputs/models")`.

## Limitations

- Trained on simulated data parameterised by Smillie et al. 2019 marker genes.
- Performance may differ on real 10x data (batch effects, different protocols).
- Re-train by running `python -m src.main` on your own data.
"""
