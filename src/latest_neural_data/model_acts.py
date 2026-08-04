"""Utilities to extract model activations from a cached image folder.

These helpers load stimulus IDs and images from a `majajhong_cache`-style
directory and run a forward pass to collect activations for a specified
layer. Results are cached to a pickle to speed repeated runs.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import pickle


def _state_dicts_equal(state_a, state_b):
    """Return True if two state dicts are equal (tensor-aware comparison)."""
    if state_a is None or state_b is None:
        return False
    if state_a.keys() != state_b.keys():
        return False

    for key in state_a.keys():
        val_a = state_a[key]
        val_b = state_b[key]
        if torch.is_tensor(val_a) and torch.is_tensor(val_b):
            if not torch.equal(val_a.detach().cpu(), val_b.detach().cpu()):
                return False
        else:
            if val_a != val_b:
                return False
    return True

def extract_model_activations_from_cache(
    model, 
    cache_dir="./majajhong_cache",
    layer_name=None,
    batch_size=32,
    device=None,
    return_neural_activations=False
):
    """Extract activations for `layer_name` from cached images.

    Returns `(activations, stimulus_ids)` or `(activations, stimulus_ids, neural_activations)` if
    `return_neural_activations` is True.
    """
    cache_dir = Path(cache_dir)
    images_dir = cache_dir / "images"

    stimulus_ids = np.load(cache_dir / "stimulus_ids.npy", allow_pickle=True)
    n_images = len(stimulus_ids)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    try:
        cache_path = cache_dir / f"model_activations_{layer_name}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            cached_model = cached.get("model")
            if _state_dicts_equal(cached_model, model.state_dict()):
                if return_neural_activations:
                    neural_activations = np.load(cache_dir / "neural_activations.npy")
                    return cached["activations"], stimulus_ids, neural_activations
                return cached["activations"], stimulus_ids
    except Exception as e:
        print(f"Warning: Failed to load cache due to {e}. Recomputing activations.")

    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    layer_module = dict(model.named_modules())[layer_name]
    hook = layer_module.register_forward_hook(hook_fn(layer_name))
    
    # Setup preprocessing
    preprocess = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Extract features in batches
    all_features = []
    
    for batch_start in range(0, n_images, batch_size):
        batch_end = min(batch_start + batch_size, n_images)
        batch_stim_ids = stimulus_ids[batch_start:batch_end]
        
        # Load images
        batch_images = []
        for stim_id in batch_stim_ids:
            img_path = images_dir / f"{stim_id}.png"
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img)
            batch_images.append(img_tensor)
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Forward pass
        with torch.no_grad():
            _ = model(batch_tensor)
        
        # Extract activations
        batch_activations = activations[layer_name]
        

        batch_features = batch_activations.view(batch_activations.shape[0], -1).numpy()
        all_features.append(batch_features)
        
        if batch_end % (batch_size * 5) == 0 or batch_end == n_images:
            print(f"  Processed {batch_end}/{n_images} images...")
    
    hook.remove()
    
    # Combine all features
    model_activations = np.vstack(all_features).astype(np.float32)


    #####################
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "model": model.state_dict(),
                "activations": model_activations
            }, f)
    except Exception as e:
        print(f"Warning: Failed to save cache due to {e}.")
    ####################
    print(return_neural_activations)
    if return_neural_activations:
        neural_activations = np.load(cache_dir / "neural_activations.npy")
        return model_activations, stimulus_ids, neural_activations
    else:
        return model_activations, stimulus_ids