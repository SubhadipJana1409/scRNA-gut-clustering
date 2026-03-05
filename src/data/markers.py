"""
Real published gut mucosal cell type marker genes.

Sources
-------
1. Smillie et al. 2019, Nature — "Intra- and Inter-cellular Rewiring of the
   Human Colon during Ulcerative Colitis"
   → 51 cell subsets across Epithelial, Stromal, and Immune compartments
   → github.com/cssmillie/ulcerative_colitis/blob/master/cell_subsets.txt

2. Elmentaite et al. 2021, Nature — "Cells of the human intestinal tract
   mapped across space and time"
   → Gut Cell Atlas: 428,000 cells, small + large intestine

3. Kong et al. 2023, Nature — IBD spatial transcriptomics colitis atlas

4. Human Protein Atlas (proteinatlas.org) — single-cell type RNA tissue

These markers are the gold standard for gut mucosal cell identification.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  EPITHELIAL COMPARTMENT  (from Smillie 2019 + Elmentaite 2021)
# ──────────────────────────────────────────────────────────────────────────────

STEM_CELL = {
    "name": "Stem Cells",
    "compartment": "Epithelial",
    "color": "#2ecc71",
    # LGR5+ intestinal stem cells (Barker et al. 2007, Nature)
    "markers_up": [
        "LGR5", "ASCL2", "OLFM4", "SMOC2", "RGMB", "EPHB2",
        "SOX9", "AXIN2", "PROM1", "CD44", "LRIG1",
    ],
    "markers_down": ["KRT20", "FABP1", "MUC2"],
}

TRANSIT_AMPLIFYING = {
    "name": "Transit-Amplifying (TA)",
    "compartment": "Epithelial",
    "color": "#27ae60",
    # Cycling progenitor cells (MKI67+)
    "markers_up": [
        "MKI67", "TOP2A", "PCNA", "MCM2", "MCM5", "MCM6",
        "TYMS", "CCNB1", "CDK1", "HIST1H4C", "BIRC5",
    ],
    "markers_down": ["LGR5", "FABP1"],
}

ENTEROCYTE = {
    "name": "Enterocytes",
    "compartment": "Epithelial",
    "color": "#3498db",
    # Absorptive enterocytes (most abundant gut epithelial cell)
    "markers_up": [
        "FABP1", "FABP2", "ALPI", "APOA4", "VIL1", "SLC5A1",
        "SLC2A2", "SLC15A1", "ACE2", "ANPEP", "SI",
        "APOB", "APOC3", "CUBN", "AMN",
    ],
    "markers_down": ["MUC2", "CHGA", "LGR5"],
}

BEST4_ENTEROCYTE = {
    "name": "Best4+ Enterocytes",
    "compartment": "Epithelial",
    "color": "#1abc9c",
    # Rare BEST4+ electrogenic enterocytes (Parikh 2019, Nature)
    "markers_up": [
        "BEST4", "OTOP2", "CFTR", "CA7", "SPIB",
        "CEACAM7", "UGT2A3",
    ],
    "markers_down": ["FABP1", "MUC2"],
}

GOBLET_CELL = {
    "name": "Goblet Cells",
    "compartment": "Epithelial",
    "color": "#e74c3c",
    # Mucus-secreting goblet cells
    "markers_up": [
        "MUC2", "TFF1", "TFF3", "CLCA1", "SPDEF",
        "FCGBP", "ITLN1", "RETNLB", "AGR2", "GALNT12",
        "SLC26A3", "ZG16",
    ],
    "markers_down": ["FABP1", "CHGA"],
}

ENTEROENDOCRINE = {
    "name": "Enteroendocrine Cells",
    "compartment": "Epithelial",
    "color": "#9b59b6",
    # Hormone-secreting EEC (GLP-1, serotonin, etc.)
    "markers_up": [
        "CHGA", "CHGB", "SCG2", "NEUROD1", "PAX4",
        "GCG", "SST", "CCK", "GIP", "NTS", "PYY",
        "NEUROG3", "PCSK1", "TPH1",
    ],
    "markers_down": ["MUC2", "FABP1"],
}

TUFT_CELL = {
    "name": "Tuft Cells",
    "compartment": "Epithelial",
    "color": "#e67e22",
    # Chemosensory cells, important for helminth immunity
    "markers_up": [
        "DCLK1", "TRPM5", "POU2F3", "RGS13", "GFI1B",
        "LRMP", "SH2D6", "BMX", "PTPRC",
    ],
    "markers_down": ["MUC2", "CHGA"],
}

# ──────────────────────────────────────────────────────────────────────────────
#  IMMUNE COMPARTMENT  (from Smillie 2019 T-cell / Myeloid clusters)
# ──────────────────────────────────────────────────────────────────────────────

CD4_T_CELL = {
    "name": "CD4+ T Cells",
    "compartment": "Immune",
    "color": "#f39c12",
    "markers_up": [
        "CD3D", "CD3E", "CD3G", "CD4", "IL7R",
        "TCF7", "LEF1", "CCR7", "SELL", "LTB",
        "FOXP3", "IL2RA",  # Treg subset
    ],
    "markers_down": ["CD8A", "CD68", "JCHAIN"],
}

CD8_T_CELL = {
    "name": "CD8+ T Cells / IELs",
    "compartment": "Immune",
    "color": "#d35400",
    # Intraepithelial lymphocytes (IELs)
    "markers_up": [
        "CD8A", "CD8B", "CD3D", "CD3E",
        "GZMB", "GZMK", "PRF1", "NKG7",
        "ITGAE", "ITGB7", "CD69",
    ],
    "markers_down": ["CD4", "CD68"],
}

PLASMA_B_CELL = {
    "name": "Plasma / B Cells",
    "compartment": "Immune",
    "color": "#8e44ad",
    # IgA-secreting plasma cells (dominant in gut)
    "markers_up": [
        "JCHAIN", "IGHA1", "IGHA2", "IGHG1", "IGHG4",
        "CD79A", "CD38", "PRDM1", "XBP1", "MZB1",
        "CD19", "MS4A1",  # B-cell markers
    ],
    "markers_down": ["CD3D", "CD68"],
}

MACROPHAGE = {
    "name": "Macrophages / Myeloid",
    "compartment": "Immune",
    "color": "#c0392b",
    # Tissue-resident macrophages + DCs
    "markers_up": [
        "CD68", "CSF1R", "MRC1", "CD14", "FCGR3A",
        "ITGAM", "TYROBP", "LYZ", "AIF1", "ADGRE1",
        "IL1B", "TNF", "CXCL8",  # inflammatory markers
        "HLA-DRA", "HLA-DRB1",
    ],
    "markers_down": ["CD3D", "JCHAIN"],
}

# ──────────────────────────────────────────────────────────────────────────────
#  STROMAL COMPARTMENT  (from Smillie 2019 Fibroblast/Endothelial clusters)
# ──────────────────────────────────────────────────────────────────────────────

FIBROBLAST = {
    "name": "Fibroblasts / Myofibroblasts",
    "compartment": "Stromal",
    "color": "#7f8c8d",
    # Lamina propria fibroblasts (WNT2B+, RSPO3+ subtypes in Smillie 2019)
    "markers_up": [
        "COL1A1", "COL1A2", "COL3A1", "DCN", "LUM",
        "PDGFRA", "FAP", "THY1", "ACTA2", "TAGLN",
        "WNT2B", "RSPO3", "WNT5B",
    ],
    "markers_down": ["CD3D", "EPCAM", "PECAM1"],
}

ENDOTHELIAL = {
    "name": "Endothelial Cells",
    "compartment": "Stromal",
    "color": "#95a5a6",
    # Vascular + lymphatic endothelium
    "markers_up": [
        "PECAM1", "VWF", "CDH5", "CLDN5", "ENG",
        "TIE1", "KDR", "FLT1", "PLVAP", "ICAM1",
        "ESAM", "ROBO4",
    ],
    "markers_down": ["EPCAM", "COL1A1", "CD3D"],
}

# ──────────────────────────────────────────────────────────────────────────────
#  MASTER CELL TYPE DICTIONARY
# ──────────────────────────────────────────────────────────────────────────────

CELL_TYPES = {
    "Stem Cells":              STEM_CELL,
    "Transit-Amplifying":      TRANSIT_AMPLIFYING,
    "Enterocytes":             ENTEROCYTE,
    "Best4+ Enterocytes":      BEST4_ENTEROCYTE,
    "Goblet Cells":            GOBLET_CELL,
    "Enteroendocrine":         ENTEROENDOCRINE,
    "Tuft Cells":              TUFT_CELL,
    "CD4+ T Cells":            CD4_T_CELL,
    "CD8+ T Cells/IELs":       CD8_T_CELL,
    "Plasma/B Cells":          PLASMA_B_CELL,
    "Macrophages":             MACROPHAGE,
    "Fibroblasts":             FIBROBLAST,
    "Endothelial":             ENDOTHELIAL,
}

# Expected proportions in healthy colon (based on Smillie 2019 cell counts)
CELL_TYPE_PROPORTIONS = {
    "Stem Cells":              0.04,
    "Transit-Amplifying":      0.08,
    "Enterocytes":             0.22,
    "Best4+ Enterocytes":      0.03,
    "Goblet Cells":            0.12,
    "Enteroendocrine":         0.02,
    "Tuft Cells":              0.01,
    "CD4+ T Cells":            0.12,
    "CD8+ T Cells/IELs":       0.10,
    "Plasma/B Cells":          0.10,
    "Macrophages":             0.06,
    "Fibroblasts":             0.06,
    "Endothelial":             0.04,
}

# All unique marker genes (for building the gene panel)
ALL_MARKER_GENES = sorted(set(
    gene
    for ct in CELL_TYPES.values()
    for gene in ct["markers_up"] + ct.get("markers_down", [])
))

# Cell type colors for plotting
CELL_TYPE_COLORS = {name: ct["color"] for name, ct in CELL_TYPES.items()}

# Compartment-level groupings (from Smillie 2019 cell_subsets.txt)
COMPARTMENTS = {
    "Epithelial": ["Stem Cells", "Transit-Amplifying", "Enterocytes",
                   "Best4+ Enterocytes", "Goblet Cells", "Enteroendocrine", "Tuft Cells"],
    "Immune":     ["CD4+ T Cells", "CD8+ T Cells/IELs", "Plasma/B Cells", "Macrophages"],
    "Stromal":    ["Fibroblasts", "Endothelial"],
}
