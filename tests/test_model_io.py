"""Tests for model save / load / predict."""
import json
import numpy as np
import pytest
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


@pytest.fixture(scope="module")
def trained_artefacts(tmp_path_factory):
    """Run a tiny pipeline and save models."""
    import scanpy as sc
    from src.data import load_simulated
    from src.models.pipeline import run_qc, preprocess, build_neighbors_and_umap, cluster
    from src.models.model_io import save_models

    cfg = {
        "qc": {"min_genes_per_cell": 3, "max_genes_per_cell": 10000,
               "max_pct_mito": 50, "min_cells_per_gene": 1},
        "preprocessing": {"target_sum": 10000, "log1p": True,
                           "n_top_genes": 50, "n_pcs": 10},
        "clustering": {"n_neighbors": 5, "n_umap_components": 2,
                       "kmeans_k": 4, "leiden_resolution": 0.3},
    }
    adata = load_simulated(n_cells=150, n_genes=120, random_seed=42)
    adata = run_qc(adata, cfg)
    adata = preprocess(adata, cfg)
    sc.pp.neighbors(adata, n_neighbors=5, n_pcs=8)
    sc.tl.umap(adata)
    adata = cluster(adata, cfg)

    out_dir = tmp_path_factory.mktemp("outputs")
    km  = adata.uns.pop("_kmeans_model")
    pca = adata.uns.pop("_pca_model")
    paths = save_models(adata, km, pca, cfg, out_dir)
    return out_dir, km, pca, adata, paths


class TestSaveModels:
    def test_files_created(self, trained_artefacts):
        out_dir, *_ = trained_artefacts
        model_dir = out_dir / "models"
        assert (model_dir / "kmeans_model.joblib").exists()
        assert (model_dir / "pca_model.joblib").exists()
        assert (model_dir / "pipeline_config.json").exists()
        assert (model_dir / "model_card.md").exists()

    def test_config_json_valid(self, trained_artefacts):
        out_dir, *_ = trained_artefacts
        cfg_path = out_dir / "models" / "pipeline_config.json"
        with open(cfg_path) as f:
            meta = json.load(f)
        assert "n_cells" in meta
        assert "kmeans_k" in meta
        assert "hvg_list" in meta
        assert len(meta["hvg_list"]) > 0

    def test_model_card_markdown(self, trained_artefacts):
        out_dir, *_ = trained_artefacts
        card = (out_dir / "models" / "model_card.md").read_text()
        assert "KMeans" in card
        assert "PCA" in card
        assert "Silhouette" in card


class TestLoadModels:
    def test_loads_correctly(self, trained_artefacts):
        from src.models.model_io import load_models
        out_dir, *_ = trained_artefacts
        artefacts = load_models(out_dir / "models")
        assert "kmeans" in artefacts
        assert "pca" in artefacts
        assert "hvg_list" in artefacts

    def test_kmeans_has_centroids(self, trained_artefacts):
        from src.models.model_io import load_models
        out_dir, *_ = trained_artefacts
        artefacts = load_models(out_dir / "models")
        km = artefacts["kmeans"]
        assert hasattr(km, "cluster_centers_")
        assert km.cluster_centers_.ndim == 2

    def test_pca_n_components(self, trained_artefacts):
        from src.models.model_io import load_models
        out_dir, km_orig, pca_orig, *_ = trained_artefacts
        artefacts = load_models(out_dir / "models")
        assert artefacts["pca"].n_components_ == pca_orig.n_components_

    def test_missing_dir_raises(self):
        from src.models.model_io import load_models
        with pytest.raises(FileNotFoundError):
            load_models("/nonexistent/path/models")


class TestPredictNewCells:
    def test_predict_returns_array(self, trained_artefacts):
        from src.models.model_io import load_models, predict_new_cells
        from src.data import build_gene_panel
        out_dir, *_ = trained_artefacts
        artefacts  = load_models(out_dir / "models")
        gene_names = build_gene_panel(120)
        new_counts = np.random.negative_binomial(1, 0.5, size=(10, 120)).astype(np.float32)
        labels = predict_new_cells(new_counts, gene_names, artefacts)
        assert labels.shape == (10,)

    def test_labels_within_range(self, trained_artefacts):
        from src.models.model_io import load_models, predict_new_cells
        from src.data import build_gene_panel
        out_dir, _, _, _, _ = trained_artefacts
        artefacts  = load_models(out_dir / "models")
        gene_names = build_gene_panel(120)
        new_counts = np.random.randint(0, 10, size=(20, 120)).astype(np.float32)
        labels = predict_new_cells(new_counts, gene_names, artefacts)
        k = artefacts["kmeans"].n_clusters
        assert all(0 <= l < k for l in labels)
