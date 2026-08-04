"""Linear-probe evaluation utilities.

Functions to extract frozen encoder features and train a single linear
classifier for top-1 accuracy (and some adversarial robustness helpers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.post_training import ModuleWrapper, extract_val_dl_from_ckpt, extract_model_weights, extract_ckpt_args
from src.models.simclr_model import SimCLR
from types import SimpleNamespace



@torch.no_grad()
def _extract_features(encoder, dl, device):
    encoder.eval()
    feats, ys = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        h = encoder(x)              # IMPORTANT: use encoder features, not projection z
        feats.append(h.detach())
        ys.append(y)
    return torch.cat(feats, 0), torch.cat(ys, 0)


def linear_probe_top1(encoder, tr_dl, va_dl, num_classes, device,
                      epochs=10, lr=0.1, wd=0.0, feat_batch=2048):
    """
    Freeze encoder, train a single Linear layer on frozen encoder features.
    Returns val top-1 (%).
    """
    with torch.no_grad():
        Xtr, ytr = _extract_features(encoder, tr_dl, device)
        Xva, yva = _extract_features(encoder, va_dl, device)

    feat_dim = Xtr.shape[1]
    clf = nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    n = Xtr.shape[0]
    for _ in range(epochs):
        print(f"Epoch {_+1}/{epochs}")
        perm = torch.randperm(n, device=device)
        Xtr_shuf = Xtr[perm]
        ytr_shuf = ytr[perm]

        clf.train()
        for i in range(0, n, feat_batch):
            xb = Xtr_shuf[i:i+feat_batch]
            yb = ytr_shuf[i:i+feat_batch]
            loss = F.cross_entropy(clf(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    clf.eval()
    with torch.no_grad():
        pred = clf(Xva).argmax(dim=1)
        return (pred == yva).float().mean().item() * 100.0


def linear_probe_standalone(ckpt_path, dl_kwargs = {"workers": 3}, lp_kwargs={}):
    args = extract_ckpt_args(ckpt_path, as_args=True)
    eval_tr_dl, eval_va_dl = extract_val_dl_from_ckpt(ckpt_path, kwargs=dl_kwargs)
    base_ds = eval_tr_dl.dataset.dataset if hasattr(eval_tr_dl.dataset, "dataset") else eval_tr_dl.dataset
    num_classes = len(base_ds.classes)
    model = SimCLR()
    state_dict = extract_model_weights(ckpt_path)
    try:
        model.load_state_dict(state_dict)
        encoder = model.encoder
    except Exception as e:
        model = SimCLR()
        model = ModuleWrapper(model)
        model.load_state_dict(state_dict)
        encoder = model.module.encoder
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:        device = "cpu"
    model = model.to(device)
    for key, value in lp_kwargs.items():
        setattr(args, key, value)
    top1 = linear_probe_top1(encoder, eval_tr_dl, eval_va_dl, num_classes=num_classes,
                    device=device, epochs=args.lp_epochs, lr=args.lp_lr, wd=args.lp_wd)
    return top1

def linear_probe_adversarial_robustness(encoder, va_dl, num_classes, device,
                                        clf=None, epsilon=8/255, num_steps=10,
                                        step_size=2/255, epochs=10, lr=0.1):
    """
    Evaluate adversarial robustness of linear probe using PGD attacks.
    Returns clean accuracy and adversarial accuracy (%).
    """
    with torch.no_grad():
        Xva, yva = _extract_features(encoder, va_dl, device)
    
    feat_dim = Xva.shape[1]
    if clf is None:
        clf = nn.Linear(feat_dim, num_classes).to(device)
        opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9)
        for _ in range(epochs):
            perm = torch.randperm(Xva.shape[0], device=device)
            for i in range(0, Xva.shape[0], 256):
                idx = perm[i:i+256]
                loss = F.cross_entropy(clf(Xva[idx]), yva[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
    
    clf.eval()
    
    # Clean accuracy
    with torch.no_grad():
        clean_pred = clf(Xva).argmax(dim=1)
        clean_acc = (clean_pred == yva).float().mean().item() * 100.0
    
    # PGD adversarial accuracy
    Xva_adv = _pgd_attack(clf, Xva, yva, epsilon, num_steps, step_size)
    with torch.no_grad():
        adv_pred = clf(Xva_adv).argmax(dim=1)
        adv_acc = (adv_pred == yva).float().mean().item() * 100.0
    
    return clean_acc, adv_acc

def linear_probe_adversarial_robustness_standalone(ckpt_path, dl_kwargs = {"workers": 3}, epsilon=8/255, num_steps=10, step_size=2/255):
    args = extract_ckpt_args(ckpt_path, as_args=True)
    eval_tr_dl, eval_va_dl = extract_val_dl_from_ckpt(ckpt_path, kwargs=dl_kwargs)
    base_ds = eval_tr_dl.dataset.dataset if hasattr(eval_tr_dl.dataset, "dataset") else eval_tr_dl.dataset
    num_classes = len(base_ds.classes)
    model = SimCLR()
    state_dict = extract_model_weights(ckpt_path)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device)
    clean_acc, adv_acc = linear_probe_adversarial_robustness(model.encoder, eval_va_dl, num_classes=num_classes,
                                                            device=device, epochs=args.lp_epochs, lr=args.lp_lr,
                                                            epsilon=epsilon, num_steps=num_steps, step_size=step_size)
    return clean_acc, adv_acc

def _pgd_attack(model, X, y, epsilon, num_steps, step_size):
    """PGD adversarial attack on features."""
    X_adv = X.clone().detach().requires_grad_(True)
    
    for _ in range(num_steps):
        with torch.enable_grad():
            loss = F.cross_entropy(model(X_adv), y)
        grad = torch.autograd.grad(loss, X_adv)[0]
        X_adv = X_adv + step_size * grad.sign()
        X_adv = torch.clamp(X_adv, X - epsilon, X + epsilon)
        X_adv = X_adv.detach().requires_grad_(True)
    
    return X_adv.detach()