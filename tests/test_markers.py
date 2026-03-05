"""Tests for marker gene definitions."""
import pytest
from src.data.markers import (
    CELL_TYPES, CELL_TYPE_PROPORTIONS, ALL_MARKER_GENES,
    CELL_TYPE_COLORS, COMPARTMENTS
)


def test_cell_types_not_empty():
    assert len(CELL_TYPES) >= 10


def test_all_have_markers():
    for name, ct in CELL_TYPES.items():
        assert len(ct["markers_up"]) >= 3, f"{name} has < 3 markers"


def test_proportions_sum_to_one():
    total = sum(CELL_TYPE_PROPORTIONS.values())
    assert abs(total - 1.0) < 0.01, f"Proportions sum to {total}"


def test_proportions_cover_all_types():
    for ct in CELL_TYPES:
        assert ct in CELL_TYPE_PROPORTIONS, f"{ct} missing from proportions"


def test_all_marker_genes_nonempty():
    assert len(ALL_MARKER_GENES) >= 50


def test_known_markers_present():
    """Key IBD/gut cell markers must be in the database."""
    must_have = ["LGR5", "MUC2", "CHGA", "FABP1", "CD3D", "CD68", "COL1A1"]
    for g in must_have:
        assert g in ALL_MARKER_GENES, f"{g} missing from marker gene list"


def test_faecalibacterium_not_in_markers():
    """Microbiome genera should not be in scRNA-seq gene list."""
    assert "Faecalibacterium" not in ALL_MARKER_GENES


def test_compartments_cover_all_types():
    all_comp_types = [ct for types in COMPARTMENTS.values() for ct in types]
    for ct in CELL_TYPES:
        assert ct in all_comp_types, f"{ct} not in any compartment"


def test_colors_hex():
    for name, color in CELL_TYPE_COLORS.items():
        assert color.startswith("#"), f"{name} color not hex: {color}"
        assert len(color) == 7, f"{name} color wrong length: {color}"
