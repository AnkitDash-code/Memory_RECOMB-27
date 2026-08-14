import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.load_breast_cancer import load_breast_cancer, N_REGIONS
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding, consensus_cluster
from src.models.train_zism_model import train_zism_model
from src.models.train_ldcm_model import train_ldcm_model

ROOT = Path('outputs/logs')
ROOT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_dlpfc(train_fn, emb_key):
    raw = load_dlpfc_slice('151673')
    adata = preprocess_hvg(raw.copy())
    truth = adata.obs['ground_truth_layer']
    mask = truth.notna().to_numpy()
    coords = adata.obsm['spatial']
    _, trained, history = train_fn(
        adata.copy(),
        seed=0,
        device=DEVICE,
        epochs=600,
        log_every=200,
        verbose=False,
    )
    emb = trained.obsm[emb_key]
    labels = cluster_embedding(emb, int(truth.nunique()), coords=coords, refine=True)
    ari = float(adjusted_rand_score(truth[mask], np.asarray(labels)[mask]))
    return {
        'slice': '151673',
        'seed': 0,
        'ari': ari,
        'n_spots': int(adata.n_obs),
        'n_layers': int(truth.nunique()),
        'history_summary': {
            'epoch_0': history[0],
            'epoch_final': history[-1],
        },
        'full_history': history,
    }


def run_breast(train_fn, emb_key, model_name):
    raw = load_breast_cancer()
    adata = preprocess_hvg(raw.copy(), platform='visium')
    truth = adata.obs['ground_truth_region']
    mask = truth.notna().to_numpy()
    coords = adata.obsm['spatial']
    per_seed = []
    label_sets = []
    seed_diagnostics = []
    for seed in range(5):
        _, trained, history = train_fn(
            adata.copy(),
            seed=seed,
            device=DEVICE,
            epochs=600,
            log_every=200,
            verbose=False,
        )
        emb = trained.obsm[emb_key]
        labels = cluster_embedding(emb, N_REGIONS, coords=coords, refine=True)
        ari = float(adjusted_rand_score(truth[mask], np.asarray(labels)[mask]))
        per_seed.append(ari)
        label_sets.append(labels)
        final = history[-1]
        seed_diagnostics.append({
            'seed': seed,
            'ari': ari,
            'n_slots_used': final.get('n_slots_used'),
            'usage_entropy': final.get('usage_entropy'),
            'max_entropy': final.get('max_entropy'),
            'collapsed': final.get('n_slots_used', 0) <= 1,
        })
    consensus = consensus_cluster(label_sets, N_REGIONS)
    consensus_ari = float(adjusted_rand_score(truth[mask], np.asarray(consensus)[mask]))
    return {
        'n_spots': int(adata.n_obs),
        'n_regions': N_REGIONS,
        'model': model_name,
        'seeds': list(range(5)),
        'baselines': {
            'ours_SpatialAddressMemoryLayer': {'mean': 0.412, 'std': 0.072, 'consensus': 0.546},
            'graphst': {'mean': 0.621, 'std': 0.021, 'consensus': 0.643},
        },
        model_name.lower(): {
            'per_seed': per_seed,
            'mean': float(np.mean(per_seed)),
            'std': float(np.std(per_seed)),
            'consensus': consensus_ari,
            'seed_diagnostics': seed_diagnostics,
        },
    }


for model_name, fn, emb_key in [
    ('ZISM', train_zism_model, 'X_zism'),
    ('LDCM', train_ldcm_model, 'X_ldcm'),
]:
    print(f'Running {model_name} DLPFC...')
    dlpfc = run_dlpfc(fn, emb_key)
    dlpfc_path = ROOT / f'{model_name.lower()}_dlpfc_smoke.json'
    dlpfc_path.write_text(json.dumps(dlpfc, indent=2))
    print(f'Saved {dlpfc_path}')

    print(f'Running {model_name} Breast Cancer...')
    bc = run_breast(fn, emb_key, model_name)
    bc_path = ROOT / f'{model_name.lower()}_breast_cancer_results.json'
    bc_path.write_text(json.dumps(bc, indent=2))
    print(f'Saved {bc_path}')
