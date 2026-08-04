"""k-NN evaluation utilities for encoder features.

Provides a standard kNN top-1 evaluator and a standalone wrapper that loads
a checkpoint and runs kNN on the corresponding dataset.
"""

import torch
import torch.nn.functional as F
from src.utils.post_training import extract_val_dl_from_ckpt, extract_model_weights, extract_ckpt_args
from src.models.simclr_model import SimCLR

@torch.no_grad()
def knn_top1(encoder, tr_dl, va_dl, device, k=1, temperature=0.07):
    """
    kNN on encoder features.
    For k=1: nearest neighbor.
    For k>1: similarity-weighted vote (standard in SSL eval).
    """
    encoder.eval()

    feats, labels = [], []
    for x, y in tr_dl:
        x = x.to(device, non_blocking=True)
        h = F.normalize(encoder(x), dim=1)
        feats.append(h)
        labels.append(y.to(device, non_blocking=True))
    feats = torch.cat(feats, 0)     # [N, D]
    labels = torch.cat(labels, 0)   # [N]
    ft = feats.t()                  # [D, N]

    num_classes = int(labels.max().item()) + 1

    correct = total = 0
    for x, y in va_dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        h = F.normalize(encoder(x), dim=1)  # [B, D]
        sims = (h @ ft)                     # [B, N]

        if k == 1:
            nn = sims.argmax(dim=1)
            pred = labels[nn]
        else:
            vals, idx = sims.topk(k=k, dim=1, largest=True, sorted=True)  # [B,k]
            nbr_labels = labels[idx]                                      # [B,k]
            weights = (vals / temperature).exp()                          # [B,k]

            scores = torch.zeros(h.size(0), num_classes, device=device)
            scores.scatter_add_(1, nbr_labels, weights)
            pred = scores.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

    return 100.0 * correct / max(1, total)

def knn_standalone(ckpt_path, dl_kwargs = {"workers": 3}, k=1, temperature=0.07):
    args = extract_ckpt_args(ckpt_path, as_args=True)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)

    eval_tr_dl, eval_va_dl = extract_val_dl_from_ckpt(ckpt_path, kwargs=dl_kwargs)

    model_weights = extract_model_weights(ckpt_path)
    model = SimCLR()
    model.load_state_dict(model_weights)
    model.to(device)

    return knn_top1(model.encoder, eval_tr_dl, eval_va_dl, device, k=k, temperature=temperature)