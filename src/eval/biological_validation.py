"""Check that predicted spatial domains are biologically real, not just statistically separable.

A high ARI alone does not prove a method recovered *cortical layers* -- it only
proves its partition agrees with the annotation. This module tests the
independent biological claim: each predicted domain should be enriched for the
canonical marker genes of the layer it maps to.

Marker sources: markers flagged (verified) below are reported as
layer-enriched in Maynard et al. 2021, Nature Neuroscience ("Transcriptome-scale
spatial gene expression in the human dorsolateral prefrontal cortex"), the study
that produced the spatialLIBD annotations used as ground truth here. Markers
flagged (convention) are widely used in the spatial-transcriptomics literature
for these layers but were not quoted from that paper's own validation list --
they are labelled separately rather than presented as equally sourced.
"""

import numpy as np
import pandas as pd

# layer -> (gene symbols, provenance)
CANONICAL_DLPFC_MARKERS = {
    "Layer1": (["AQP4"], "verified"),
    "Layer2": (["HPCAL1"], "verified"),
    "Layer3": (["FREM3"], "verified"),
    "Layer4": (["RORB"], "convention"),
    "Layer5": (["TRABD2A", "PCP4"], "verified"),
    "Layer6": (["KRT17"], "verified"),
    "WM": (["MOBP"], "verified"),
}


def _expression_vector(adata, gene, layer=None):
    """Mean-safe extraction of one gene's expression as a dense 1-D array."""
    if gene not in adata.var_names:
        return None
    values = adata[:, gene].layers[layer] if layer else adata[:, gene].X
    values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    return np.ravel(values)


def marker_enrichment_table(adata, cluster_key, layer=None):
    """Mean expression of each canonical marker within each predicted domain.

    Returns a DataFrame indexed by predicted domain, one column per marker gene
    that is actually present in `adata`. Genes absent from the matrix (e.g.
    filtered out as non-variable) are reported separately rather than silently
    dropped -- a marker missing from the data is a caveat about the check, not
    evidence about the model.
    """
    labels = np.asarray(adata.obs[cluster_key])
    domains = sorted(set(labels))

    columns, missing = {}, []
    for markers, _provenance in CANONICAL_DLPFC_MARKERS.values():
        for gene in markers:
            expression = _expression_vector(adata, gene, layer=layer)
            if expression is None:
                missing.append(gene)
                continue
            columns[gene] = [expression[labels == domain].mean() for domain in domains]

    return pd.DataFrame(columns, index=pd.Index(domains, name=cluster_key)), missing


def validate_annotation_biology(adata, truth_key="ground_truth_layer", layer=None):
    """Validate the DATA, independent of any model: do canonical markers enrich
    in the annotated layers they are supposed to mark?

    This is the check that answers "is this dataset biologically what the
    literature says it is?" -- it never looks at a prediction, so it cannot be
    satisfied by a model gaming the metric. If AQP4 is not enriched in the spots
    annotated Layer1, something is wrong with the data or the annotation, and
    every downstream ARI number is suspect.

    Reported per marker: mean expression inside its layer, mean outside, and the
    log2 fold change between them.
    """
    truth = adata.obs[truth_key]
    annotated = truth.notna().to_numpy()
    truth_values = truth.to_numpy()

    rows, missing = [], []
    for layer_name, (markers, provenance) in CANONICAL_DLPFC_MARKERS.items():
        in_layer = annotated & (truth_values == layer_name)
        if not in_layer.any():
            continue
        out_layer = annotated & (truth_values != layer_name)

        for gene in markers:
            expression = _expression_vector(adata, gene, layer=layer)
            if expression is None:
                missing.append(gene)
                continue
            inside = float(expression[in_layer].mean())
            outside = float(expression[out_layer].mean())
            rows.append({
                "layer": layer_name,
                "marker": gene,
                "provenance": provenance,
                "mean_in_layer": inside,
                "mean_outside": outside,
                "log2_fold_change": float(np.log2((inside + 1e-9) / (outside + 1e-9))),
                "enriched": bool(inside > outside),
            })

    frame = pd.DataFrame(rows)
    enriched = frame["enriched"] if "enriched" in frame else pd.Series(dtype=bool)
    summary = {
        "n_markers_tested": int(len(enriched)),
        "n_enriched": int(enriched.sum()) if len(enriched) else 0,
        "fraction_enriched": float(enriched.mean()) if len(enriched) else None,
        "missing_genes": missing,
    }
    return frame, summary


def validate_markers_against_truth(adata, cluster_key, truth_key="ground_truth_layer", layer=None):
    """Test whether each marker peaks in the predicted domain that best matches its layer.

    For each annotated layer we find the predicted domain that overlaps it most,
    then check whether that domain has the highest mean expression of the layer's
    canonical marker(s) across all predicted domains. Passing means the model's
    domains carry the right biology, not merely the right partition.
    """
    table, missing = marker_enrichment_table(adata, cluster_key, layer=layer)
    labels = np.asarray(adata.obs[cluster_key])
    truth = adata.obs[truth_key]
    annotated = truth.notna().to_numpy()

    results = []
    for layer_name, (markers, provenance) in CANONICAL_DLPFC_MARKERS.items():
        in_layer = annotated & (truth.to_numpy() == layer_name)
        if not in_layer.any():
            continue

        # Predicted domain that best covers this annotated layer.
        domains, counts = np.unique(labels[in_layer], return_counts=True)
        best_domain = domains[np.argmax(counts)]

        for gene in markers:
            if gene not in table.columns:
                results.append({
                    "layer": layer_name, "marker": gene, "provenance": provenance,
                    "matched_domain": best_domain, "enriched": None,
                    "note": "gene not present in matrix",
                })
                continue
            top_domain = table[gene].idxmax()
            results.append({
                "layer": layer_name,
                "marker": gene,
                "provenance": provenance,
                "matched_domain": best_domain,
                "top_expressing_domain": top_domain,
                "enriched": bool(top_domain == best_domain),
                "marker_mean_in_matched": float(table.loc[best_domain, gene]),
            })

    frame = pd.DataFrame(results)
    scored = frame["enriched"].dropna() if "enriched" in frame else pd.Series(dtype=bool)
    summary = {
        "n_markers_tested": int(len(scored)),
        "n_enriched_correctly": int(scored.sum()) if len(scored) else 0,
        "fraction_correct": float(scored.mean()) if len(scored) else None,
        "missing_genes": missing,
    }
    return frame, summary
