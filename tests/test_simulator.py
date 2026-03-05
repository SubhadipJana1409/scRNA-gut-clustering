"""Tests for scRNA-seq count simulator."""
import numpy as np
import pytest
from src.data.simulator import simulate_counts, build_gene_panel


class TestBuildGenePanel:
    def test_length(self):
        panel = build_gene_panel(500)
        assert len(panel) == 500

    def test_unique(self):
        panel = build_gene_panel(300)
        assert len(panel) == len(set(panel))

    def test_marker_genes_included(self):
        panel = build_gene_panel(2000)
        assert "LGR5" in panel
        assert "MUC2" in panel
        assert "FABP1" in panel


class TestSimulateCounts:
    @pytest.fixture(scope="class")
    def small_data(self):
        return simulate_counts(n_cells=200, n_genes=300, random_seed=0)

    def test_shapes(self, small_data):
        counts, cells, genes, labels = small_data
        assert counts.shape == (200, 300)
        assert len(cells) == 200
        assert len(genes) == 300
        assert len(labels) == 200

    def test_non_negative(self, small_data):
        counts, *_ = small_data
        assert (counts >= 0).all()

    def test_sparse(self, small_data):
        counts, *_ = small_data
        sparsity = (counts == 0).sum() / counts.size
        # Real scRNA-seq is ~90%+ sparse
        assert sparsity > 0.5, f"Sparsity only {sparsity:.1%}"

    def test_all_cell_types_present(self, small_data):
        _, _, _, labels = small_data
        from src.data.markers import CELL_TYPES
        present = set(labels)
        # Most cell types should appear (some rare ones may be absent in 200 cells)
        assert len(present) >= 5

    def test_reproducible(self):
        c1, *_ = simulate_counts(n_cells=50, n_genes=100, random_seed=42)
        c2, *_ = simulate_counts(n_cells=50, n_genes=100, random_seed=42)
        np.testing.assert_array_equal(c1, c2)

    def test_different_seeds(self):
        c1, *_ = simulate_counts(n_cells=50, n_genes=100, random_seed=1)
        c2, *_ = simulate_counts(n_cells=50, n_genes=100, random_seed=2)
        assert not np.array_equal(c1, c2)
