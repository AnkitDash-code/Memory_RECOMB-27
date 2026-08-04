# Hop-Fusion implementation status

The Hop-Fusion plan is implemented as a separate ablation path. The existing
fixed-hop `SpatialAddressMemoryLayer` and the rejected adaptive gate remain
available as comparators.

## Physical scale

`src/data/physical_scale.py` defines the platform spacing namespace and the
single conversion function used by the new model:

```python
um_radius_to_hop_count(radius_um, platform, avg_edge_length_um)
```

`preprocess()` and `preprocess_hvg()` measure the positive graph-edge lengths
and store the result in `adata.uns["spatial_scale"]`. The heterogeneity proxy
and Hop-Fusion window both consume that measured value; they do not use a
platform-specific hardcoded hop count.

## DLPFC lock workflow

`configs/hop_fusion_selection.json` is intentionally marked
`pending_dlpfc_selection`. Run:

```powershell
.venv\Scripts\python.exe -m src.eval.cross_validate_hop_fusion
```

The selector uses the existing three-slice DLPFC validation partition,
excludes 151673, reports the three requested ablations, and writes
`configs/hop_fusion_dlpfc.json` with `status: "locked"` only after selection.
That locked file is the only config accepted by the generalization runner.

## Generalization workflow

Breast cancer is evaluated with contiguous spatial-block holdouts and is
reported as a same-platform domain-size test. Stereo-seq OB must be supplied
as an explicit local `.h5ad` containing `obsm['spatial']`:

```powershell
.venv\Scripts\python.exe -m src.eval.run_hop_fusion_generalization `
  --stereo-path data\stereoseq_olfactory_bulb.h5ad `
  --n-clusters <chosen-common-cluster-count>
```

Stereo-seq is scored with silhouette, spatial coherence, and optional supplied
marker-set separation. No supervised ARI is inferred for it.
