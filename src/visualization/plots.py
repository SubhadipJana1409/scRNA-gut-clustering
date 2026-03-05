"""
Visualisation functions for the gut scRNA-seq clustering pipeline.

All figures use a consistent dark publication theme.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Dark theme ────────────────────────────────────────────────────────────────
BG   = "#0f0f1a"
AX   = "#1a1a2e"
TEXT = "#e8e8f0"
ACC  = "#7c6af7"
GRID = "#2e2e4e"

_CMAP_SEQ  = "YlOrRd"
_CMAP_DIVG = "RdBu_r"


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    AX,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "text.color":        TEXT,
        "legend.facecolor":  AX,
        "legend.edgecolor":  GRID,
        "grid.color":        GRID,
        "grid.alpha":        0.4,
        "font.size":         11,
        "savefig.facecolor": BG,
    })
    sc.set_figure_params(dpi=120, facecolor=BG, frameon=False)
    sc.settings.figsize = (6, 5)


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor=BG, dpi=dpi)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ──────────────────────────────────────────────────────────────────────────────
def plot_qc_violin(adata: ad.AnnData, out_dir: Path) -> None:
    """Fig 1 — QC violin plots (n_genes, total_counts, pct_mito)."""
    metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    labels  = ["Genes / Cell", "UMI Counts / Cell", "Mitochondrial %"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Quality Control Metrics — Gut Mucosal Cells", fontsize=14, fontweight="bold")

    for ax, metric, label in zip(axes, metrics, labels):
        data = adata.obs[metric].values
        ax.violinplot(data, showmedians=True, showextrema=True)
        ax.scatter(np.ones(len(data)) + np.random.normal(0, 0.02, len(data)),
                   data, alpha=0.15, s=2, color=ACC)
        ax.set_ylabel(label)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"Median: {np.median(data):.0f}")

    plt.tight_layout()
    _save(fig, out_dir / "fig1_qc_violin.png")


# ──────────────────────────────────────────────────────────────────────────────
def plot_pca_variance(adata: ad.AnnData, out_dir: Path) -> None:
    """Fig 2 — PCA variance explained elbow plot."""
    var_ratio = adata.uns["pca"]["variance_ratio"]
    n_show    = min(40, len(var_ratio))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("PCA — Dimensionality Reduction", fontsize=14, fontweight="bold")

    # Elbow plot
    pcs = np.arange(1, n_show + 1)
    axes[0].plot(pcs, var_ratio[:n_show] * 100, "o-", color=ACC, lw=2, ms=5)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance Explained (%)")
    axes[0].set_title("Scree Plot")
    axes[0].grid(True, alpha=0.3)

    # Cumulative variance
    cumvar = np.cumsum(var_ratio[:n_show]) * 100
    axes[1].plot(pcs, cumvar, "s-", color="#e74c3c", lw=2, ms=5)
    axes[1].axhline(80, color="#7c6af7", linestyle="--", lw=1.5, label="80%")
    axes[1].axhline(90, color="#2ecc71", linestyle="--", lw=1.5, label="90%")
    axes[1].set_xlabel("Principal Component")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_title("Cumulative Variance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir / "fig2_pca_variance.png")


# ──────────────────────────────────────────────────────────────────────────────
def plot_pca_scatter(adata: ad.AnnData, out_dir: Path) -> None:
    """Fig 3 — PCA scatter coloured by cell type and compartment."""
    from ..data.markers import CELL_TYPE_COLORS

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("PCA Embedding — Gut Mucosal Cells", fontsize=14, fontweight="bold")

    X_pca = adata.obsm["X_pca"]

    # By cell type
    if "cell_type" in adata.obs.columns:
        cell_types = adata.obs["cell_type"].unique()
        for ct in cell_types:
            mask = adata.obs["cell_type"] == ct
            color = CELL_TYPE_COLORS.get(ct, "#aaaaaa")
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=color, s=4, alpha=0.5, label=ct, rasterized=True)
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].set_title("Cell Types")
        axes[0].legend(
            markerscale=3, fontsize=6, loc="upper right",
            ncol=2, framealpha=0.5)

    # By compartment
    if "compartment" in adata.obs.columns:
        comp_colors = {"Epithelial": "#3498db", "Immune": "#e74c3c", "Stromal": "#2ecc71", "Unknown": "#7f8c8d"}
        for comp in adata.obs["compartment"].unique():
            mask = adata.obs["compartment"] == comp
            axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=comp_colors.get(comp, "#aaaaaa"), s=4, alpha=0.5, label=comp, rasterized=True)
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].set_title("Compartment")
        axes[1].legend(markerscale=3, fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir / "fig3_pca_scatter.png")


# ──────────────────────────────────────────────────────────────────────────────
def plot_umap(adata: ad.AnnData, out_dir: Path) -> None:
    """Fig 4 — UMAP coloured by Leiden cluster, cell type, and key markers."""
    from ..data.markers import CELL_TYPE_COLORS

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("UMAP Embedding — Gut Mucosal Cells", fontsize=14, fontweight="bold")

    X_umap = adata.obsm["X_umap"]

    # Panel A: Leiden clusters
    leiden_cats = sorted(adata.obs["leiden"].unique(), key=int)
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(leiden_cats), 1)))
    for i, cl in enumerate(leiden_cats):
        mask = adata.obs["leiden"] == cl
        axes[0].scatter(X_umap[mask, 0], X_umap[mask, 1],
                        c=[palette[i % len(palette)]], s=4, alpha=0.6,
                        label=f"C{cl}", rasterized=True)
    axes[0].set_title("Leiden Clusters")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].legend(markerscale=3, fontsize=7, ncol=3, loc="upper right", framealpha=0.4)

    # Panel B: True cell types
    if "cell_type" in adata.obs.columns:
        for ct in adata.obs["cell_type"].unique():
            mask  = adata.obs["cell_type"] == ct
            color = CELL_TYPE_COLORS.get(ct, "#aaaaaa")
            axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1],
                            c=color, s=4, alpha=0.6, label=ct, rasterized=True)
        axes[1].set_title("True Cell Types")
        axes[1].set_xlabel("UMAP 1")
        axes[1].set_ylabel("")
        axes[1].legend(markerscale=3, fontsize=6, ncol=2, loc="upper right", framealpha=0.4)
    else:
        axes[1].set_visible(False)

    # Panel C: Key marker gene expression overlay (FABP1 = enterocyte marker)
    marker_gene = "FABP1"
    if marker_gene in adata.var_names:
        gene_idx = list(adata.var_names).index(marker_gene)
        expr = np.array(adata.X[:, gene_idx].todense()).flatten() if hasattr(adata.X, "todense") else adata.X[:, gene_idx]
        # Use normalised expression
        if "log1p" in adata.uns.get("log1p", {}) or adata.X.max() < 30:
            pass  # already log-normalised
        sc2 = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=expr,
                               cmap="YlOrRd", s=4, alpha=0.7, rasterized=True)
        plt.colorbar(sc2, ax=axes[2], label="log1p Expression")
        axes[2].set_title(f"{marker_gene} Expression")
        axes[2].set_xlabel("UMAP 1")
        axes[2].set_ylabel("")
    else:
        axes[2].set_title("No FABP1 in panel")
        axes[2].set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir / "fig4_umap.png")


# ──────────────────────────────────────────────────────────────────────────────
def plot_umap_markers(adata: ad.AnnData, out_dir: Path) -> None:
    """Fig 5 — UMAP multi-panel: expression of 8 canonical gut markers."""

    key_markers = ["LGR5", "MKI67", "FABP1", "MUC2", "CHGA", "CD3D", "CD68", "COL1A1"]
    available   = [g for g in key_markers if g in adata.var_names][:8]
    n           = len(available)
    if n == 0:
        logger.warning("No key markers found in data — skipping fig5")
        return

    ncols = 4
    nrows = (n + 3) // 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Canonical Gut Marker Gene Expression on UMAP", fontsize=14, fontweight="bold")
    axes_flat = axes.flat if nrows > 1 else [axes] if ncols == 1 else list(axes)

    X_umap = adata.obsm["X_umap"]
    for ax, gene in zip(axes_flat, available):
        gene_idx = list(adata.var_names).index(gene)
        expr = np.array(adata.X[:, gene_idx].todense()).flatten() if hasattr(adata.X, "todense") else adata.X[:, gene_idx]
        sc2  = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=expr,
                          cmap="YlOrRd", s=3, alpha=0.6, vmin=0, rasterized=True)
        plt.colorbar(sc2, ax=ax, shrink=0.8)
        ax.set_title(f"*{gene}*", style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(AX)

    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir / "fig5_umap_markers.png")


# ──────────────────────────────────────────────────────────────────────────────
def plot_cluster_composition(adata: ad.AnnData, out_dir: Path) -> None:
    """Fig 6 — Stacked bar chart: cell type composition per Leiden cluster."""
    if "cell_type" not in adata.obs.columns:
        return

    from ..data.markers import CELL_TYPE_COLORS

    ct_per_cluster = pd.crosstab(adata.obs["leiden"], adata.obs["cell_type"], normalize="index") * 100
    ct_per_cluster = ct_per_cluster.sort_index(key=lambda x: x.astype(int))
    cell_types     = ct_per_cluster.columns.tolist()
    colors         = [CELL_TYPE_COLORS.get(ct, "#aaaaaa") for ct in cell_types]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    bottom = np.zeros(len(ct_per_cluster))
    for ct, color in zip(cell_types, colors):
        values = ct_per_cluster[ct].values
        ax.bar(ct_per_cluster.index, values, bottom=bottom, label=ct, color=color,
               edgecolor=BG, linewidth=0.5)
        bottom += values

    ax.set_xlabel("Leiden Cluster")
    ax.set_ylabel("Cell Type Fraction (%)")
    ax.set_title("Cell Type Composition per Leiden Cluster", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.7)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    _save(fig, out_dir / "fig6_cluster_composition.png")


# ──────────────────────────────────────────────────────────────────────────────
def plot_marker_heatmap(adata: ad.AnnData, out_dir: Path) -> None:
    """Fig 7 — Heatmap of top marker genes per cluster (dotplot style)."""
    if "rank_genes_groups" not in adata.uns:
        logger.warning("run_marker_genes() not called — skipping fig7")
        return

    result   = adata.uns["rank_genes_groups"]
    clusters = list(result["names"].dtype.names)
    # Top 3 genes per cluster
    top_genes = []
    for cl in clusters:
        for i in range(min(3, len(result["names"][cl]))):
            g = result["names"][cl][i]
            if g not in top_genes and g in adata.var_names:
                top_genes.append(g)

    if not top_genes:
        return

    # Build mean expression matrix (per cluster)
    mean_expr = {}
    for cl in clusters:
        mask = adata.obs["leiden"] == cl
        X_cl = adata[mask, top_genes].X
        if hasattr(X_cl, "toarray"):
            X_cl = X_cl.toarray()
        mean_expr[cl] = X_cl.mean(axis=0)
    df = pd.DataFrame(mean_expr, index=top_genes).T

    fig, ax = plt.subplots(figsize=(max(12, len(top_genes) * 0.4), 6))
    fig.patch.set_facecolor(BG)
    sns.heatmap(df, cmap="YlOrRd", ax=ax, linewidths=0.3, linecolor=BG,
                cbar_kws={"label": "Mean log1p expression"},
                xticklabels=True, yticklabels=True)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Leiden Cluster")
    ax.set_title("Top Marker Genes per Cluster", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    _save(fig, out_dir / "fig7_marker_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
def plot_cell_annotation_umap(adata: ad.AnnData, out_dir: Path) -> None:
    """Fig 8 — UMAP side-by-side: Leiden clusters vs predicted cell types."""
    from ..data.markers import CELL_TYPE_COLORS

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Leiden Clusters → Predicted Cell Type Annotation",
                 fontsize=14, fontweight="bold")
    X_umap = adata.obsm["X_umap"]

    # Leiden
    leiden_cats = sorted(adata.obs["leiden"].unique(), key=int)
    pal = plt.cm.tab20(np.linspace(0, 1, max(len(leiden_cats), 1)))
    for i, cl in enumerate(leiden_cats):
        mask = adata.obs["leiden"] == cl
        axes[0].scatter(X_umap[mask, 0], X_umap[mask, 1],
                        c=[pal[i % len(pal)]], s=5, alpha=0.6, label=f"C{cl}", rasterized=True)
        # Cluster centre label
        cx, cy = X_umap[mask, 0].mean(), X_umap[mask, 1].mean()
        axes[0].text(cx, cy, cl, fontsize=8, ha="center", va="center", color="white",
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", fc=pal[i % len(pal)], alpha=0.7))
    axes[0].set_title("Leiden Clusters", fontsize=12)
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")

    # Predicted cell types
    if "predicted_cell_type" in adata.obs.columns:
        pred_cts = adata.obs["predicted_cell_type"].unique()
        for ct in pred_cts:
            mask  = adata.obs["predicted_cell_type"] == ct
            color = CELL_TYPE_COLORS.get(ct, "#aaaaaa")
            axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1],
                            c=color, s=5, alpha=0.6, label=ct, rasterized=True)
        axes[1].set_title("Predicted Cell Types", fontsize=12)
        axes[1].set_xlabel("UMAP 1")
        axes[1].set_ylabel("")
        axes[1].legend(markerscale=3, fontsize=7, ncol=2, loc="upper right", framealpha=0.5)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir / "fig8_annotated_umap.png")


# ──────────────────────────────────────────────────────────────────────────────
def plot_compartment_summary(adata: ad.AnnData, out_dir: Path) -> None:
    """Fig 9 — Summary pie charts: cell type and compartment breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Gut Mucosal Cell Population Summary", fontsize=14, fontweight="bold")

    from ..data.markers import CELL_TYPE_COLORS

    # Cell type pie
    if "cell_type" in adata.obs.columns:
        ct_counts = adata.obs["cell_type"].value_counts()
        colors    = [CELL_TYPE_COLORS.get(ct, "#aaaaaa") for ct in ct_counts.index]
        wedges, texts, autotexts = axes[0].pie(
            ct_counts.values, labels=ct_counts.index,
            colors=colors, autopct="%1.1f%%",
            pctdistance=0.82, startangle=140,
            wedgeprops=dict(edgecolor=BG, linewidth=1.5),
            textprops={"color": TEXT, "fontsize": 7},
        )
        for at in autotexts:
            at.set_fontsize(6)
        axes[0].set_title("By Cell Type", fontsize=11)

    # Compartment pie
    if "compartment" in adata.obs.columns:
        comp_counts = adata.obs["compartment"].value_counts()
        comp_colors = {"Epithelial": "#3498db", "Immune": "#e74c3c", "Stromal": "#2ecc71", "Unknown": "#7f8c8d"}
        colors = [comp_colors.get(c, "#aaaaaa") for c in comp_counts.index]
        axes[1].pie(comp_counts.values, labels=comp_counts.index, colors=colors,
                    autopct="%1.1f%%", pctdistance=0.82, startangle=140,
                    wedgeprops=dict(edgecolor=BG, linewidth=2),
                    textprops={"color": TEXT, "fontsize": 9})
        axes[1].set_title("By Compartment", fontsize=11)

    plt.tight_layout()
    _save(fig, out_dir / "fig9_compartment_summary.png")
