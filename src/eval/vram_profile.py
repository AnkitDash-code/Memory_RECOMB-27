import time

import torch


def profile_forward(model, x, device):
    model = model.to(device)
    x = x.to(device)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    start = time.time()
    with torch.no_grad():
        out = model(x)
    elapsed = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    return elapsed, peak_mb
