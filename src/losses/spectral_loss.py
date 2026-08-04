"""Spectral regularization utilities.

Functions to estimate the power-law exponent of activation eigenspectra and
compute a loss that penalizes deviations from a target exponent.
"""

import os

from src.latest_neural_data.model_acts import extract_model_activations_from_cache
import torch
import numpy as np
from src.models.simclr_model import SimCLR
from src.utils.post_training import extract_val_dl_from_ckpt, extract_model_weights, extract_ckpt_args
from src.utils.model_activations import ModelActivations
from torch import nn


def spectral_loss(activation, device=None, n_components=40, target_alpha=1.0, bounds=(5, 15), log_base="10", eps=1e-12):
    """Compute a spectral regularization loss that penalizes deviation from `target_alpha`.

    Args:
        activation: tensor of shape [N, ...] containing activations for a batch/dataset.
        device: optional device string to move loss to.
        n_components: number of principal components to use.
        target_alpha: desired power-law exponent.
        bounds: tuple (min_pc, max_pc) indicating PC indices used for regression.
        log_base: '10' or 'e' for log base used when fitting the slope.

    Returns:
        (loss_value, alpha): loss tensor (absolute deviation) and estimated alpha.
    """
    X = activation.view(activation.size(0), -1)
    X = X - X.mean(axis=0)
    B, N = X.shape
    q=min(n_components, B, N)
    with torch.amp.autocast(enabled=False, device_type=device):
        X32 = X.float()
        U, S, V = torch.pca_lowrank(X32, q=q)
    eigvals = S**2
    eigvals = eigvals / (torch.sum(eigvals) + eps)
    min_idx = max(1, int(bounds[0]))
    max_idx = min(int(bounds[1]), eigvals.shape[-1])
    idx = torch.arange(min_idx, max_idx+1, device=eigvals.device, dtype=eigvals.dtype)
    if log_base == "e":
        log_fn = torch.log
    elif log_base == "10":
        log_fn = torch.log10
    y = log_fn(eigvals[min_idx-1:max_idx] + eps)
    x = log_fn(idx + 0.0)

    # x = (x - x.mean()) / (x.std() + eps)
    x= x - x.mean()
    y = y - y.mean(dim=-1, keepdim=True)
    slope = (y * x).mean(dim=-1) / (x.pow(2).mean() + eps)
    alpha = -slope
    loss_value = (alpha - target_alpha).abs().mean()
    return loss_value if device is None else loss_value.to(device), alpha



def just_alpha_fixed(activation, device=None, n_components=40, bounds=(5, 15), eps=1e-12, log_base="10",return_eigvals=False):
    """Estimate the power-law exponent `alpha` of activation eigenspectrum.

    Similar to `spectral_loss` but returns only the estimated `alpha`. When
    `return_eigvals` is True, also returns the normalized eigenvalues.
    """
    X = activation.view(activation.size(0), -1)
    # print(f"Activation shape after view: {X.shape}")
    X = X - X.mean(axis=0)
    B, N = X.shape
    q=min(n_components, B, N)
    U, S, V = torch.pca_lowrank(X, q=q)
    eigvals = S**2
    eigvals = eigvals / (torch.sum(eigvals) + eps)

    min_idx = max(1, int(bounds[0]))
    max_idx = min(int(bounds[1]), eigvals.shape[-1])

    # 1-based PC indices, inclusive range
    idx = torch.arange(min_idx, max_idx + 1, device=eigvals.device, dtype=eigvals.dtype)
    if log_base == "e":
        log_fn = torch.log
    elif log_base == "10":
        log_fn = torch.log10
    y = log_fn(eigvals[min_idx - 1:max_idx] + eps)
    x = log_fn(idx + 0.0)

    # center only, do not standardize x
    x = x - x.mean()
    y = y - y.mean(dim=-1, keepdim=True)

    slope = (y * x).mean(dim=-1) / (x.pow(2).mean() + eps)
    alpha = -slope
    if return_eigvals:
        return alpha if device is None else alpha.to(device), eigvals
    return alpha if device is None else alpha.to(device)

class ModuleWrapper(nn.Module):
    """Thin wrapper that exposes `module` attribute for compatibility with saved checkpoints.

    Some checkpoints save models wrapped in `nn.DataParallel` or similar; this
    helper provides a simple namespace that mirrors that layout when loading.
    """
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def just_alpha_imgnet_standalone(ckpt_path, dl_kwargs = {"workers": 3, "imagenet_root": "/path/to/imagenet"}, alpha_kwargs={}):
    """Helper: compute average `alpha` on ImageNet validation set for a checkpoint.

    Loads model weights from `ckpt_path`, builds eval loaders with `dl_kwargs`,
    and returns the mean estimated alpha across processed batches.
    """
    args = extract_ckpt_args(ckpt_path, as_args=True)
    eval_tr_dl, eval_va_dl = extract_val_dl_from_ckpt(ckpt_path, kwargs=dl_kwargs)
    base_ds = eval_tr_dl.dataset.dataset if hasattr(eval_tr_dl.dataset, "dataset") else eval_tr_dl.dataset
    num_classes = len(base_ds.classes)
    model = SimCLR()
    state_dict = extract_model_weights(ckpt_path)
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        model = SimCLR()
        model = ModuleWrapper(model)
        model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:        device = "cpu"
    model = model.to(device)
    for key, value in alpha_kwargs.items():
        setattr(args, key, value)

    activationclass = ModelActivations(model, layers=[args.neural_ev_layer])
    activationclass.register_hooks()
    alphas = []
    with torch.no_grad():
        for batch in eval_tr_dl:
            inputs = batch[0].to(device)
            _ = model(inputs)
            a = just_alpha_fixed(activationclass.activations[args.neural_ev_layer], device=device)
            alphas.append(float(a.detach().float().cpu().item()))
    avg_alpha = float(np.mean(alphas))
    return avg_alpha

def obtain_imgnet_acts(ckpt_path, dl_kwargs = {"workers": 3, "imagenet_root": "/path/to/imagenet"}, verbose=False, detach=False, compress=False):
    """Extract and cache ImageNet activations for the model checkpoint.

    Saves activations to `<ckpt_dir>/imgnet_acts.pt` and returns the stacked tensor.
    Supports optional float16 compression and detaching of tensors.
    """
    ckpt_dir = "/".join(ckpt_path.split("/")[:-1])
    acts_path = f"{ckpt_dir}/imgnet_acts.pt"
    if os.path.exists(acts_path):
        if verbose:
            print(f"ImageNet activations already exist at {acts_path}, loading...")
        return torch.load(acts_path)
    else:
        acts_path = f"{ckpt_dir}/imgnet_acts.pt"        
        args = extract_ckpt_args(ckpt_path, as_args=True)
        eval_tr_dl, eval_va_dl = extract_val_dl_from_ckpt(ckpt_path, kwargs=dl_kwargs)
        base_ds = eval_tr_dl.dataset.dataset if hasattr(eval_tr_dl.dataset, "dataset") else eval_tr_dl.dataset
        num_classes = len(base_ds.classes)
        model = SimCLR()
        print(f"Extracting model weights from checkpoint at {ckpt_path}...")
        state_dict = extract_model_weights(ckpt_path)
        try:
            print("Attempting to load state dict without ModuleWrapper...")
            model.load_state_dict(state_dict)
            print("Loaded state dict without ModuleWrapper.")
        except Exception as e:
            model = SimCLR()
            model = ModuleWrapper(model)
            print("Attempting to load state dict with ModuleWrapper...")
            model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:        device = "cpu"
        model = model.to(device)

        activationclass = ModelActivations(model, layers=[args.neural_ev_layer], detach=detach)
        activationclass.register_hooks()
        acts= []
        with torch.no_grad():
            for i, batch in enumerate(eval_tr_dl):
                if verbose and i % 10 == 0:
                    print(f"Processing batch {i}/{len(eval_tr_dl)} for ImageNet activations...")
                inputs = batch[0].to(device)
                _ = model(inputs)
                act = activationclass.activations[args.neural_ev_layer].detach().cpu()
                if compress:
                    act = act.to(torch.float16)
                acts.append(act)
        acts = torch.cat(acts, dim=0)
        ckpt_dir = "/".join(ckpt_path.split("/")[:-1])
        acts_path = f"{ckpt_dir}/imgnet_acts.pt"
        torch.save(acts, acts_path)
        if verbose:
            print(f"Saved ImageNet activations to {acts_path}")
        return acts
    
def effective_dim_standalone(ckpt_path, dl_kwargs = {"workers": 3, "imagenet_root": "/path/to/imagenet"}):
    """Compute effective dimensionality (participation ratio) from cached ImageNet activations."""
    acts = obtain_imgnet_acts(ckpt_path, dl_kwargs=dl_kwargs)
    pca = torch.pca_lowrank(acts.float(), q=min(40, acts.shape[0], acts.shape[1]))
    eigvals = pca[1]**2
    eigvals = eigvals / (torch.sum(eigvals) + 1e-12)
    ED = (sum(eigvals)**2) / (sum(eigvals**2) + 1e-12)
    return ED


def just_alpha_brainscore_standalone(ckpt_path, dl_kwargs = {"workers": 3}, alpha_kwargs={}):
    """Estimate alpha on the brainscore (MajajHong) stimulus set using cached neural images.

    Loads model weights from `ckpt_path`, extracts model activations using the
    MajajHong cache, then returns the estimated alpha scalar.
    """
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
    else:        device = "cpu"
    model = model.to(device)
    for key, value in alpha_kwargs.items():
        setattr(args, key, value)

    model_activations, stimulus_ids = extract_model_activations_from_cache(
            model=model,
            cache_dir=args.neural_data_dir,
            layer_name=args.neural_ev_layer,
            batch_size=args.batch_size
        )
    model_activations = torch.tensor(model_activations) if not isinstance(model_activations, torch.Tensor) else model_activations
    alphas = []
    a = just_alpha_fixed(model_activations.to(device), device=device)
    return a.item()
    

        

if __name__ == "__main__":
    from src.models.simclr_model import SimCLR
    from src.utils.model_activations import ModelActivations
    model = SimCLR(out_dim=128).to('cpu')
    activationclass = ModelActivations(model, layers=['encoder.layer4.0.bn1'])
    activationclass.register_hooks()
    x = torch.randn(4, 3, 128, 128).to('cpu')
    z = model(x)
    print("Output shape:", z.shape)
    print(z.max(), z.min())
    print(z.mean())
    acts = activationclass.fetch_activations('encoder.layer4.0.bn1')
    print("Fetched activation shape:", acts.shape)
    loss_value, alpha = spectral_loss(acts, device='cpu')
    print("Spectral loss value:", loss_value.item())
    print("Alpha:", alpha.item())
    # acts = torch.randn(16, 128, 8, 8)
    # loss_value, alpha = spectral_loss(acts, device='cpu')
    # print("Spectral loss value:", loss_value.item())
    # print("Alpha:", alpha.item())