import anndata as ad
import numpy as np
import pandas as pd

from src.eval.biological_validation import (
    CANONICAL_DLPFC_MARKERS,
    marker_enrichment_table,
    validate_annotation_biology,
    validate_markers_against_truth,
)


def _synthetic_layered_adata(perfect=True):
    """Build data where each layer's canonical marker is genuinely enriched in
    that layer, then either predict the layers correctly or scramble them."""
    layers = list(CANONICAL_DLPFC_MARKERS)
    genes = [g for markers, _ in CANONICAL_DLPFC_MARKERS.values() for g in markers]
    n_per = 20

    truth = np.repeat(layers, n_per)
    x = np.full((len(truth), len(genes)), 0.1)
    for layer_idx, layer in enumerate(layers):
        for gene in CANONICAL_DLPFC_MARKERS[layer][0]:
            x[truth == layer, genes.index(gene)] = 10.0

    adata = ad.AnnData(X=x)
    adata.var_names = genes
    adata.obs["ground_truth_layer"] = pd.Categorical(truth)

    if perfect:
        predicted = truth.copy()
    else:
        rng = np.random.default_rng(0)
        predicted = truth.copy()
        rng.shuffle(predicted)
    adata.obs["pred"] = pd.Categorical(predicted)
    return adata


def test_marker_enrichment_table_shape_and_missing_reporting():
    adata = _synthetic_layered_adata()

    table, missing = marker_enrichment_table(adata, "pred")

    assert missing == []
    assert set(table.columns) == {
        g for markers, _ in CANONICAL_DLPFC_MARKERS.values() for g in markers
    }
    assert len(table) == adata.obs["pred"].nunique()


def test_validation_passes_when_domains_match_layers():
    adata = _synthetic_layered_adata(perfect=True)

    frame, summary = validate_markers_against_truth(adata, "pred")

    assert summary["n_markers_tested"] > 0
    assert summary["fraction_correct"] == 1.0
    assert frame["enriched"].dropna().all()


def test_annotation_biology_passes_on_real_layer_structure():
    """Data-level check: markers must enrich in the layers they mark."""
    adata = _synthetic_layered_adata()

    frame, summary = validate_annotation_biology(adata)

    assert summary["fraction_enriched"] == 1.0
    assert (frame["log2_fold_change"] > 1.0).all()


def test_annotation_biology_fails_when_markers_are_not_layer_specific():
    """If marker expression carries no layer structure, the data-level check
    must NOT report enrichment -- this is what makes it a real test rather
    than a formality."""
    adata = _synthetic_layered_adata()
    rng = np.random.default_rng(0)
    adata.X = rng.normal(loc=5.0, scale=0.1, size=adata.X.shape)  # no layer signal

    _, summary = validate_annotation_biology(adata)

    assert summary["fraction_enriched"] < 1.0


def test_prediction_check_is_consistency_not_proof():
    """Documents a real limitation: `validate_markers_against_truth` derives the
    matched domain and the top-expressing domain from the same overlap
    structure, so even scrambled predictions can score well. It is a
    consistency check on real predictions, NOT independent evidence of
    correctness -- `validate_annotation_biology` is the non-tautological one."""
    adata = _synthetic_layered_adata(perfect=False)

    _, summary = validate_markers_against_truth(adata, "pred")

    assert summary["n_markers_tested"] > 0
    assert summary["fraction_correct"] is not None


def test_missing_genes_are_reported_not_silently_dropped():
    adata = _synthetic_layered_adata()
    adata = adata[:, [g for g in adata.var_names if g != "MOBP"]].copy()

    _, missing = marker_enrichment_table(adata, "pred")

    assert "MOBP" in missing
