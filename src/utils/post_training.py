"""Post-training utilities: checkpoint inspection, activation extraction, plotting.

Contains helpers to find run artifacts, extract model arguments from checkpoints,
load/save evaluation results (forward / reverse explained variance), and simple
plotting helpers used by the analysis notebooks.
"""

import os
from types import SimpleNamespace
import pandas as pd
import torch
from src.latest_neural_data.ev_helper import forward_ev, reverse_ev
from src.latest_neural_data.model_acts import extract_model_activations_from_cache
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from src.models.simclr_model import SimCLR
import torch.nn as nn
import torch.nn.functional as F
from src.datamod.imagenet_ssl import (
    build_ssl_train_loader,
    build_ssl_val_loader,
    build_eval_loaders,
)
def fetch_csv_path_from_ckpt_path(ckpt_path):
    """Return the expected CSV log path corresponding to a checkpoint path.

    The function walks three directory levels up from `ckpt_path` and looks
    for `logs/simclr_baseline.csv`.
    """
    parent_folder = os.path.dirname(ckpt_path)
    parent_folder = os.path.dirname(parent_folder)
    parent_folder = os.path.dirname(parent_folder)
    csv_path = os.path.join(parent_folder, "logs/simclr_baseline.csv")
    if os.path.exists(csv_path):
        return csv_path
    else:
        raise FileNotFoundError(f"CSV file not found at {csv_path}")  

def fetch_full_args_from_ckpt_path(ckpt_path):
    """Return a merged dict of checkpoint args and CSV stats for a run."""
    csv_path = fetch_csv_path_from_ckpt_path(ckpt_path)
    args = extract_ckpt_args(ckpt_path)
    stats = extract_stats(csv_path)
    for k, value in stats.items():
        if k not in args:
            args[k] = value
    return args

def fetch_fr_ev_path_from_ckpt_path(ckpt_path, no_err=False, subset=None):
    """Return paths for reverse/forward EV npy files associated with a checkpoint.

    Looks three levels up from `ckpt_path` into `logs/neural_predictivity/`.
    If `subset` is provided, returns subset-specific filenames. If files are
    missing and `no_err` is True, returns the expected paths without raising.
    """
    parent_folder = os.path.dirname(ckpt_path)
    parent_folder = os.path.dirname(parent_folder)
    parent_folder = os.path.dirname(parent_folder)
    if subset is not None:
        rev_path = os.path.join(parent_folder, f"logs/neural_predictivity/reverse_ev_{subset}.npy")
        fev_path = os.path.join(parent_folder, f"logs/neural_predictivity/forward_ev_{subset}.npy")
    else:
        rev_path = os.path.join(parent_folder, "logs/neural_predictivity/reverse_ev.npy")
        fev_path = os.path.join(parent_folder, "logs/neural_predictivity/forward_ev.npy")
    if os.path.exists(rev_path) and os.path.exists(fev_path):
        return rev_path, fev_path
    else:
        if not no_err:
            raise FileNotFoundError(f"Reverse or forward EV file not found at {rev_path} or {fev_path}")
        return rev_path, fev_path

def fetch_ev_arrs_from_ckpt_path(ckpt_path, subset=None):
    """Load and return (reverse_ev, forward_ev) arrays for a checkpoint.

    Raises FileNotFoundError if the expected npy files are missing.
    """
    rev_path, fev_path = fetch_fr_ev_path_from_ckpt_path(ckpt_path, subset=subset)
    r_ev = np.load(rev_path)
    f_ev = np.load(fev_path)
    return r_ev, f_ev


def fetch_all_paths(root_dir):
    """Recursively search `root_dir` for run folders and return list of (csv, ckpt) tuples.

    A run is recognized when a `ckpts` folder contains `simclr/last.pt` and the
    sibling `logs/simclr_baseline.csv` exists.
    """
    model_paths = []
    for root, dirs, files in os.walk(root_dir):
        if os.path.basename(root) == "ckpts":
            folder = os.path.dirname(root)
            model_path = os.path.join(root, "simclr/last.pt")
            csv_path = os.path.join(folder, "logs/simclr_baseline.csv")
            if os.path.exists(csv_path) and os.path.exists(model_path):
                model_paths.append((csv_path, model_path))
    return model_paths

def extract_stats(csv_path):
    """Read a CSV log and return the last line as a dict of stats."""
    data = pd.read_csv(csv_path)
    last_informative = {}
    for i in range(len(data)):
        values = dict(data.iloc[i])
        last_informative = values
    return last_informative

def write_to_csv_from_csv_path(csv_path, new_data: dict):
    """Update the last row of an existing CSV with values from `new_data`.

    Creates new columns when keys are absent.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for key, value in new_data.items():
            if key in df.columns:
                df.loc[df.index[-1], key] = value
            else:
                df[key] = np.nan
                df.loc[df.index[-1], key] = value
        df.to_csv(csv_path, index=False)
    else:
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

def write_to_csv_from_ckpt(ckpt_path: str, new_data: dict):
    """Convenience wrapper: update CSV corresponding to `ckpt_path` with `new_data`."""
    csv_path = fetch_csv_path_from_ckpt_path(ckpt_path)
    data = pd.read_csv(csv_path)
    for key, value in new_data.items():
        if key in data.columns:
            data.loc[data.index[-1], key] = value
        else:
            data[key] = np.nan
            data.loc[data.index[-1], key] = value
    data.to_csv(csv_path, index=False)

def extract_ckpt_args(ckpt_path, as_args=False):
    """Load checkpoint and return the embedded `args` dict (or SimpleNamespace).

    This helper tolerates CPU-only environments by mapping to CPU on load
    if needed.
    """
    try:
        ckpt = torch.load(ckpt_path, weights_only=False)
    except:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    args = ckpt["args"]
    if as_args:
        args = SimpleNamespace(**args)
    return args

def extract_val_dl_from_ckpt(ckpt_path, kwargs={}):
    """Return (train_loader, val_loader) using args saved inside a checkpoint.

    Accepts `kwargs` to override loader arguments (e.g., `batch_size`).
    """
    args = extract_ckpt_args(ckpt_path, as_args=True)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)

    pin_memory = (device == "cuda")
    for key, value in kwargs.items():
        setattr(args, key, value)
    eval_tr_dl, eval_va_dl = build_eval_loaders(
        root=args.imagenet_root,
        batch_size=args.batch_size,
        workers=args.workers,
        img_size=args.img_size,
        pin_memory=pin_memory,
        limit_train=None,
        limit_val=None,
        )
    return eval_tr_dl, eval_va_dl

def to_float(str):
    """Safe float conversion: returns 0 on failure."""
    try:
        flt = float(str)
    except:
        flt = 0
    return flt

to_flt = np.vectorize(to_float)

def extract_model_weights(ckpt_path):
    """Load checkpoint and return the saved model `state_dict`."""
    try:
        ckpt = torch.load(ckpt_path, weights_only=False)
    except:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    model_weights = ckpt['model']
    return model_weights

def df_from_model_paths(model_paths):
    runs = []
    for i, (csv_path, ckpt_path) in enumerate(model_paths):
        args = extract_ckpt_args(ckpt_path)
        stats = extract_stats(csv_path)
        # if not "F_EV" in stats:
        #     continue
        for key, value in stats.items():
            if key not in args:
                args[key] = value
        args["ckpt_path"] = ckpt_path
        args["csv_path"] = csv_path
        runs.append(args)
    df = pd.DataFrame(runs)
    df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df

def df_from_root_dir(root_dir):
    model_paths = fetch_all_paths(root_dir)
    df = df_from_model_paths(model_paths)
    return df

class ModuleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def extract_model_brainscore_acts(ckpt_path, stimulus_dir=None):
    args = extract_ckpt_args(ckpt_path, as_args=True)
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

    if stimulus_dir is None:
        stimulus_dir = args.neural_data_dir


    model_activations, _, = extract_model_activations_from_cache(
            model=model,
            cache_dir=stimulus_dir,
            layer_name=args.neural_ev_layer,
            batch_size=args.batch_size,
        )
    return model_activations

def extract_model_brainscore_acts_with_neural(ckpt_path, neural_data_dir=None, layer=None):
    args = extract_ckpt_args(ckpt_path, as_args=True)
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

    if neural_data_dir is None:
        neural_data_dir = args.neural_data_dir

    if layer is not None:
        neural_layer = layer
    else:
        neural_layer = args.neural_ev_layer
    print(f"using layer {neural_layer}")

    model_activations, _, neural_activations = extract_model_activations_from_cache(
            model=model,
            cache_dir=neural_data_dir,
            layer_name=neural_layer,
            batch_size=32,
            return_neural_activations=True
        )
    return model_activations, neural_activations

def effective_dim_brainscore_standalone(ckpt_path, stimulus_dir=None):
    model_activations = torch.from_numpy(extract_model_brainscore_acts(ckpt_path, stimulus_dir=stimulus_dir))
    pca = torch.pca_lowrank(model_activations, q=min(40, model_activations.shape[0], model_activations.shape[1]))
    eigvals = pca[1]**2
    eigvals = eigvals / (torch.sum(eigvals) + 1e-12)
    ED = (sum(eigvals)**2) / (sum(eigvals**2) + 1e-12)
    return ED.item()

def common_subset(ckpt_path):
    r_ev, f_ev = fetch_ev_arrs_from_ckpt_path(ckpt_path)
    print(f"Fetched forward EV with shape {f_ev.shape} and reverse EV with shape {r_ev.shape}")
    n_top = int(0.2*len(r_ev))
    top_idx = np.argsort(r_ev)[-n_top:]
    print(f"Selected top {n_top} indices for common subset")
    return top_idx

def unique_subset(ckpt_path):
    r_ev, f_ev = fetch_ev_arrs_from_ckpt_path(ckpt_path)
    print(f"Fetched forward EV with shape {f_ev.shape} and reverse EV with shape {r_ev.shape}")
    n_bottom = int(0.2*len(r_ev))
    bottom_idx = np.argsort(r_ev)[:n_bottom]
    print(f"Selected bottom {n_bottom} indices for unique subset")
    return bottom_idx
    

def fr_ev_new(ckpt_path, old_style=True, neural_data_dir=None, subset=None, subset_array=None, layer = None, only_f=False):
    args =  fetch_full_args_from_ckpt_path(ckpt_path)
    if neural_data_dir is None:
        neural_data_dir = args["neural_data_dir"]
    model_acts, neural_acts = extract_model_brainscore_acts_with_neural(ckpt_path, neural_data_dir=neural_data_dir, layer=layer)
    print(f"Extracted model activations with shape {model_acts.shape} and neural activations with shape {neural_acts.shape}")
    r_ev_path, f_ev_path= fetch_fr_ev_path_from_ckpt_path(ckpt_path, no_err=True)
    parent_rev = os.path.dirname(r_ev_path)
    parent_fev = os.path.dirname(f_ev_path)
    os.makedirs(parent_rev, exist_ok=True)
    os.makedirs(parent_fev, exist_ok=True)
    if subset is not None and subset_array is not None:
        model_acts = model_acts[:, subset_array]
        r_ev_path = r_ev_path.replace(".npy", f"_{subset}.npy")
        f_ev_path = f_ev_path.replace(".npy", f"_{subset}.npy")
        print(f"Subsetted model activations to shape {model_acts.shape} and neural activations to shape {neural_acts.shape} using subset {subset}")

    if not only_f:
        r_ev = reverse_ev(model_acts, neural_acts, full_ev_vector=True, unrevamped=True)
        np.save(r_ev_path, r_ev)
        print(f"Saved reverse EV to {r_ev_path}")
    f_ev = forward_ev(model_acts, neural_acts, full_ev_vector=True, unrevamped=True)
    np.save(f_ev_path, f_ev)
    print(f"Saved forward EV to {f_ev_path}")
    if not only_f:
        r_ev = np.load(r_ev_path)
        return r_ev, f_ev
    else:
        return f_ev

def plot_ev_graph(r_ev, f_ev, bins_num=50, title="Histogram of Explained Variance"):
    bins = np.linspace(0, 100, bins_num)
    r_mean = np.nanmean(r_ev)
    f_mean = np.nanmean(f_ev)
    plt.figure(figsize=(8, 5))
    plt.hist(r_ev, bins=bins, color="steelblue", edgecolor="black", alpha=0.8, density=True, label="R-EV")
    plt.hist(f_ev, bins=bins, color="green", edgecolor="black", alpha=0.6, density=True, label="F-EV")
    plt.axvline(r_mean, color="navy", linestyle="--", linewidth=2, label=f"Mean Reverse EV: {int(r_mean)}")
    plt.axvline(f_mean, color="darkgreen", linestyle="--", linewidth=2, label=f"Mean Forward EV: {int(f_mean)}",)
    plt.title(title)
    plt.xlabel("Forward & Reverse Explained Variance")
    plt.ylabel("Units")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def ev_arr_from_ckpt(ckpt_path, unrevamped=True, neural_data_folder="/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache/"):
    args = fetch_full_args_from_ckpt_path(ckpt_path)
    try:
        r_ev, f_ev = fetch_ev_arrs_from_ckpt_path(ckpt_path)
    except FileNotFoundError:
        neural_activations = np.load(os.path.join(neural_data_folder, "neural_activations.npy"))
        model_weights = extract_model_weights(ckpt_path)
        model = SimCLR()
        model.load_state_dict(model_weights)
        # layer = 'encoder.layer4.1'
        layer = args.neural_ev_layer
        print(f"extracting model activations from {layer}")
        model_activations, stimulus_ids = extract_model_activations_from_cache(
            model=model,
            cache_dir=neural_data_folder,
            layer_name=layer,
            batch_size=args.batch_size
        )
        r_ev = reverse_ev(model_activations, neural_activations, full_ev_vector=True, unrevamped=unrevamped)
        f_ev = forward_ev(model_activations, neural_activations, full_ev_vector=True, unrevamped=unrevamped)

    return r_ev, f_ev

def plot_ev_from_df_2(df, unrevamped=True, bins_num=50, neural_data_folder="/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache/"):
    num = len(df.index)
    if isinstance(df, pd.Series):
        num = 1
        df = df.to_frame()
    for i in range(num):
        data = df.iloc[i]
        print(data["ckpt_path"])
        ckpt_path = data["ckpt_path"]
        r_ev, f_ev = ev_arr_from_ckpt(ckpt_path, unrevamped=unrevamped, neural_data_folder=neural_data_folder)
        plot_ev_graph(r_ev, f_ev, bins_num=bins_num, title=f"Histogram of Explained Variance | alpha: {data['alpha']:.2f} | spectral loss coefficient: {data['spectral_loss_coeff']:.2f} | forward EV: {data['F_EV']:.2f} | reverse EV: {data['R_EV']:.2f} | tag: {data['tag']}")
    
def plot_ev_ckpt_2(ckpt_path, unrevamped=True, bins_num=50, neural_data_folder="/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache/"):
    r_ev, f_ev = ev_arr_from_ckpt(ckpt_path, unrevamped=unrevamped, neural_data_folder=neural_data_folder)
    plot_ev_graph(r_ev, f_ev, bins_num=bins_num, title=f"Histogram of Explained Variance | ckpt: {os.path.basename(ckpt_path)}")

def plot_ev_from_df(df, unrevamped=True, bins_num=50, neural_data_folder="/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache/"):
    # bins = np.linspace(0, 100, bins_num)
    neural_activations = np.load(os.path.join(neural_data_folder, "neural_activations.npy"))
    num = len(df.index)
    if isinstance(df, pd.Series):
        num = 1
        df = df.to_frame()
    for i in range(num):
        data = df.iloc[i]
        print(data["ckpt_path"])
        ckpt_path = data["ckpt_path"]
        model_weights = extract_model_weights(ckpt_path)
        model = SimCLR()
        model.load_state_dict(model_weights)
        # layer = 'encoder.layer4.1'
        layer = data.get("neural_ev_layer", "encoder.layer4.1")
        print(f"extracting model activations from {layer}")
        model_activations, stimulus_ids = extract_model_activations_from_cache(
            model=model,
            cache_dir=neural_data_folder,#"REVERSE_PRED_FINAL/majajhong_cache"
            layer_name=layer,
            batch_size=32
        )
        print(model_activations.shape)
        r_ev = reverse_ev(model_activations, neural_activations, full_ev_vector=True, unrevamped=unrevamped)
        f_ev = forward_ev(model_activations, neural_activations, full_ev_vector=True, unrevamped=unrevamped)

        X = model_activations - np.mean(model_activations, axis=0)
        pca = PCA(n_components=50)
        pca.fit(X)
        eigenvalues = pca.explained_variance_
        d_eff = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        
        plot_ev_graph(r_ev, f_ev, bins_num=bins_num, title=f"Histogram of Explained Variance | alpha: {data['alpha']:.2f} | spectral loss coefficient: {data['spectral_loss_coeff']:.2f} | forward EV: {data['F_EV']:.2f} | reverse EV: {data['R_EV']:.2f} | tag: {data['tag']} | ED: {d_eff:.2f}")

def plot_ev_ckpt(ckpt_path, bins_num=50, unrevamped=True, neural_data_folder="/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache/"):
    bins_num = 50
    bins = np.linspace(0, 100, bins_num)
    neural_activations = np.load(os.path.join(neural_data_folder, "neural_activations.npy"))
    print(ckpt_path)
    model_weights = extract_model_weights(ckpt_path)
    model = SimCLR()
    model.load_state_dict(model_weights)
    # layer = 'encoder.layer4.1'

    args = extract_ckpt_args(ckpt_path, as_args=True)
    layer = getattr(args, "neural_ev_layer", "encoder.layer4.1")
    print(f"extracting model activations from {layer}")

    model_activations, stimulus_ids = extract_model_activations_from_cache(
        model=model,
        cache_dir=neural_data_folder,#"REVERSE_PRED_FINAL/majajhong_cache"
        layer_name=layer,
        batch_size=32
    )
    r_ev = reverse_ev(model_activations, neural_activations, full_ev_vector=True, unrevamped=unrevamped)
    f_ev = forward_ev(model_activations, neural_activations, full_ev_vector=True, unrevamped=unrevamped)

    X = model_activations - np.mean(model_activations, axis=0)
    pca = PCA(n_components=50)
    pca.fit(X)
    eigenvalues = pca.explained_variance_
    d_eff = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

    plot_ev_graph(r_ev, f_ev, bins_num=bins_num, title=f"Histogram of Explained Variance | alpha: {getattr(args.alpha, 'nan'):.2f} | spectral loss coefficient: {getattr(args.spectral_loss_coeff, 'nan'):.2f} | forward EV: {getattr(args.F_EV, 'nan'):.2f} | reverse EV: {getattr(args.R_EV, 'nan'):.2f} | tag: {getattr(args.tag, 'nan')} | ED: {d_eff:.2f}")
