"""Garfield (Zhou, github.com/zhou-1314/Garfield, PyPI `garfield`, v1.0.0,
pre-publication as of 2026) wrapper, used as a fourth comparator.

STATUS: **WSL/Linux only, genuinely blocked on Windows.** Garfield pulls in
`pybedtools` -> `pysam` -> `htslib`, none of which ship Windows wheels (unlike
the STAGATE/PyG `torch_sparse` situation, which turned out to be a stale
claim -- see `run_stagate.py`). Verified working inside WSL2 Ubuntu with the
same CUDA 12.8 / torch 2.11.0 stack as the rest of this project:
    uv pip install garfield gseapy
    uv pip install torch_sparse torch_scatter \
      --find-links "https://data.pyg.org/whl/torch-2.11.0+cu128.html"
(module import name is `Garfield`, capitalized -- differs from the PyPI/pip
package name `garfield`.)

Garfield v1.0.0 ships no tutorial, no quickstart, and no worked example
anywhere in its repo. Its documented default entry point (`DataProcess`) and
its own `GarfieldTrainer` class are both broken for single-sample spatial
data. Getting a real embedding out required three independent workarounds,
each confirmed via direct traceback/source inspection, not guessed:

1. `Garfield.preprocessing.preprocessing_rna(adata, ...)` requires
   `adata.obs['batch']` to exist even for one unbatched sample (its internal
   `sc.pp.highly_variable_genes(..., batch_key='batch')` call KeyErrors
   otherwise) -- fixed by adding a single dummy batch column before calling.

2. `Garfield.model.Garfield.__init__` unconditionally re-runs
   `Garfield.preprocessing.preprocess.DataProcess(...)` on `gf_params['adata_list']`,
   and DataProcess is broken for both of its two documented input shapes: a
   list of file paths crashes at `adata.obsm` (line 94 assumes every list
   element is already an AnnData), and a list of real AnnData objects crashes
   inside `concat_data`'s single-element branch (`read_multi_scData` expects a
   path string, calls `.split("/")` on the AnnData object). There is no way to
   pass adata_list that reaches the intended processing path.
   Workaround: DataProcess has an *undocumented* early-return -- if any
   element of `adata_list` already has a non-empty `obsm['garfield_latent']`,
   it returns the input unchanged, skipping all of the broken code. So this
   wrapper preprocesses with `preprocessing_rna` itself (which works), stamps
   a dummy placeholder into `obsm['garfield_latent']` purely to trip that
   early-return, and passes the *single* AnnData (not a list) as
   `gf_params['adata_list']` -- `DataProcess` then returns it verbatim as
   `model.adata`.

3. `GarfieldTrainer` (the class Garfield's own README implies is the training
   entry point) cannot actually train: its loop calls
   `self.model(data_batch=..., decoder_type=..., augment_type=...)`, but
   `Garfield` never implements `forward()` for that call signature (it falls
   through to `nn.Module._forward_unimplemented`), so `GarfieldTrainer.train()`
   always raises `TypeError` on its first real batch. Separately, `Garfield`
   itself overrides `nn.Module.train()` with its own zero-argument method that
   runs a *complete*, self-contained training pipeline (data loaders, epoch
   loop, early stopping, held-out edge-reconstruction eval) -- but this breaks
   the standard `nn.Module.train(mode: bool)` contract, so `model.eval()`
   (which calls `self.train(False)`) also raises `TypeError`.
   Workaround: skip `GarfieldTrainer` entirely and call `model.train()`
   (zero args) directly right after construction -- this is, in practice, the
   package's real (if confusingly named and undocumented) training entry
   point. Do not call `model.eval()` afterward for the same reason.

The returned embedding (`model.get_latent_representation()`, shape
n_obs x bottle_neck_neurons, default 20) is scored downstream by
`src/eval/clustering.py`, the identical protocol used for every method here.

Smoke test on DLPFC 151673, single seed, default hyperparameters:
ARI = 0.246 -- below both our method (0.303) and GraphST/STAGATE (~0.5-0.63).
Each training run takes ~5 minutes on an RTX 4050 Laptop GPU regardless of
this being a small (3639-spot) slice, so a full 12-slice x 5-seed evaluation
matching the other comparators' protocol is a ~5 hour WSL GPU commitment; see
`outputs/logs/stage2_progress.md` (Phase B3) for whether that was run and the
resulting numbers.
"""

import numpy as np
import torch


def run_garfield(adata, n_clusters, device, seed=0, n_epochs=100,
                 rna_n_top_features=3000, n_components=50, bottle_neck_neurons=20):
    """Train Garfield on a raw-counts AnnData; returns (model, processed_adata).

    Expects RAW counts in adata.X and adata.obsm['spatial'] (a fresh slice,
    not one already run through this repo's preprocess()), mirroring how
    run_graphst.py and run_stagate.py handle their own comparators' native
    preprocessing.

    The embedding is `model.get_latent_representation()`; `processed_adata`
    (== model.adata) carries obsm['spatial'] for downstream spatial-aware
    clustering refinement.
    """
    from Garfield._settings import settings
    from Garfield.model import Garfield as GarfieldModel
    from Garfield.preprocessing import preprocessing_rna

    adata = adata.copy()
    adata.layers["counts"] = adata.X.copy()
    adata.obs["batch"] = "0"

    _, hvg_adata = preprocessing_rna(
        adata,
        min_features=100,
        min_cells=3,
        target_sum=1e4,
        used_hvgs=True,
        used_pca_graph=True,
        rna_n_top_features=rna_n_top_features,
        n_components=n_components,
        n=6,
        batch_key="batch",
        metric="euclidean",
        svd_solver="arpack",
    )
    # Placeholder to trip DataProcess's early-return (see module docstring, workaround 2).
    hvg_adata.obsm["garfield_latent"] = np.zeros((hvg_adata.n_obs, 1), dtype=np.float32)

    gf_params = settings.gf_params.copy()
    gf_params.update(dict(
        adata_list=hvg_adata,
        profile="spatial",
        data_type="single-modal",
        graph_const_method="Squidpy",
        sample_col="batch",
        adj_key="spatial_connectivities",
        cluster_num=n_clusters,
        bottle_neck_neurons=bottle_neck_neurons,
        n_epochs=n_epochs,
        seed=seed,
        device_id=0 if device.type == "cuda" else None,
        monitor=True,
        verbose=False,
    ))

    model = GarfieldModel(gf_params)
    model.train()  # Garfield's own overridden method; see workaround 3. Do NOT use GarfieldTrainer.

    with torch.no_grad():
        embedding = model.get_latent_representation()
    return model, embedding
