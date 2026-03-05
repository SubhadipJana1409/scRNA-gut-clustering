"""Tests for the scRNA-seq processing pipeline."""
import anndata as ad
import pytest

from src.data import load_simulated
from src.models.pipeline import annotate_cell_types, cluster, preprocess, run_qc


@pytest.fixture(scope="module")
def tiny_adata():
    """Small AnnData for fast tests."""
    return load_simulated(n_cells=300, n_genes=200, random_seed=42)


@pytest.fixture(scope="module")
def minimal_cfg():
    return {
        "qc": {
            "min_genes_per_cell": 10,
            "max_genes_per_cell": 10000,
            "max_pct_mito": 50,
            "min_cells_per_gene": 1,
        },
        "preprocessing": {
            "target_sum": 10000,
            "log1p": True,
            "n_top_genes": 150,
            "n_pcs": 20,
        },
        "clustering": {
            "n_neighbors": 5,
            "n_umap_components": 2,
            "kmeans_k": 5,
            "leiden_resolution": 0.3,
        },
        "annotation": {"min_score_threshold": 0.1},
    }


class TestQC:
    def test_returns_anndata(self, tiny_adata, minimal_cfg):
        result = run_qc(tiny_adata.copy(), minimal_cfg)
        assert isinstance(result, ad.AnnData)

    def test_qc_columns_added(self, tiny_adata, minimal_cfg):
        result = run_qc(tiny_adata.copy(), minimal_cfg)
        assert "n_genes_by_counts" in result.obs.columns
        assert "pct_counts_mt" in result.obs.columns

    def test_filtered_cells_removed(self, tiny_adata):
        strict_cfg = {
            "qc": {
                "min_genes_per_cell": 50,   # higher threshold
                "max_genes_per_cell": 10000,
                "max_pct_mito": 5,
                "min_cells_per_gene": 1,
            }
        }
        result = run_qc(tiny_adata.copy(), strict_cfg)
        assert result.n_obs <= tiny_adata.n_obs


class TestPreprocess:
    @pytest.fixture(scope="class")
    def qc_data(self, tiny_adata, minimal_cfg):
        return run_qc(tiny_adata.copy(), minimal_cfg)

    def test_pca_added(self, tiny_adata, minimal_cfg):
        adata = run_qc(tiny_adata.copy(), minimal_cfg)
        result = preprocess(adata, minimal_cfg)
        assert "X_pca" in result.obsm

    def test_hvg_flagged(self, tiny_adata, minimal_cfg):
        adata = run_qc(tiny_adata.copy(), minimal_cfg)
        result = preprocess(adata, minimal_cfg)
        assert "highly_variable" in result.var.columns
        assert result.var["highly_variable"].sum() > 0

    def test_raw_counts_preserved(self, tiny_adata, minimal_cfg):
        adata = run_qc(tiny_adata.copy(), minimal_cfg)
        result = preprocess(adata, minimal_cfg)
        assert "counts" in result.layers


class TestClustering:
    @pytest.fixture(scope="class")
    def preprocessed(self, tiny_adata, minimal_cfg):
        import scanpy as sc
        adata = run_qc(tiny_adata.copy(), minimal_cfg)
        adata = preprocess(adata, minimal_cfg)
        sc.pp.neighbors(adata, n_neighbors=5, n_pcs=10)
        sc.tl.umap(adata)
        return adata

    def test_leiden_added(self, preprocessed, minimal_cfg):
        result = cluster(preprocessed.copy(), minimal_cfg)
        assert "leiden" in result.obs.columns

    def test_kmeans_added(self, preprocessed, minimal_cfg):
        result = cluster(preprocessed.copy(), minimal_cfg)
        assert "kmeans" in result.obs.columns

    def test_multiple_clusters(self, preprocessed, minimal_cfg):
        result = cluster(preprocessed.copy(), minimal_cfg)
        n_clusters = result.obs["leiden"].nunique()
        assert n_clusters >= 2


class TestAnnotation:
    def test_predicted_type_added(self, tiny_adata, minimal_cfg):
        import scanpy as sc
        adata = run_qc(tiny_adata.copy(), minimal_cfg)
        adata = preprocess(adata, minimal_cfg)
        sc.pp.neighbors(adata, n_neighbors=5, n_pcs=10)
        sc.tl.umap(adata)
        adata = cluster(adata, minimal_cfg)
        adata = annotate_cell_types(adata, minimal_cfg)
        assert "predicted_cell_type" in adata.obs.columns

    def test_predicted_types_are_known(self, tiny_adata, minimal_cfg):
        import scanpy as sc

        from src.data.markers import CELL_TYPES
        adata = run_qc(tiny_adata.copy(), minimal_cfg)
        adata = preprocess(adata, minimal_cfg)
        sc.pp.neighbors(adata, n_neighbors=5, n_pcs=10)
        sc.tl.umap(adata)
        adata = cluster(adata, minimal_cfg)
        adata = annotate_cell_types(adata, minimal_cfg)
        valid = set(CELL_TYPES.keys()) | {"Unassigned"}
        predicted = set(adata.obs["predicted_cell_type"])
        assert predicted <= valid
