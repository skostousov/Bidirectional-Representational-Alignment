"""Download and cache neural data and stimulus images used for evaluation.

This script extracts the Majaj & Hong dataset via `brainscore_vision`,
converts it into a (images, neurons, repeats) array, and saves stimulus
images into `latest_neural_data/majajhong_cache/images/`.
"""

import numpy as np
import brainscore_vision
from pathlib import Path
from matplotlib import image
from torchvision import transforms

benchmark = brainscore_vision.load_benchmark('MajajHong2015public.IT-pls')
neural_data_full = brainscore_vision.load_dataset("MajajHong2015.public")
neural_data_it = neural_data_full.sel(region='IT')
neural_data_it = neural_data_it.squeeze('time_bin')

neural_data_full_it = neural_data_it.values.T  # (n_presentations, n_neuroids)
stimulus_ids_it = neural_data_it.stimulus_id.values
n_neuroids = neural_data_it.shape[0]
n_presentations = neural_data_full_it.shape[0]


unique_stimuli, inverse_indices = np.unique(stimulus_ids_it, return_inverse=True)
n_images = len(unique_stimuli)

reps_per_image = np.bincount(inverse_indices)
max_reps = int(np.max(reps_per_image))


neural_activations = np.full((n_images, n_neuroids, max_reps), np.nan, dtype=np.float32)

for img_idx, stim_id in enumerate(unique_stimuli):
    # Get all presentations for this stimulus
    mask = stimulus_ids_it == stim_id
    stim_presentations = neural_data_full_it[mask, :]  # (n_reps_this_image, n_neuroids)
    n_reps_this = stim_presentations.shape[0]
    
    # Store all repetitions for this image
    neural_activations[img_idx, :, :n_reps_this] = stim_presentations.T


cache_dir = Path("latest_neural_data/majajhong_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

images_cache_dir = cache_dir / "images"
images_cache_dir.mkdir(parents=True, exist_ok=True)

neural_cache_path = cache_dir / "neural_activations.npy"
np.save(neural_cache_path, neural_activations)

stim_cache_path = cache_dir / "stimulus_ids.npy"
np.save(stim_cache_path, unique_stimuli, allow_pickle=True)


benchmark._assembly.stimulus_set.get_stimulus(unique_stimuli[0])

for stim_id in unique_stimuli:
    if (images_cache_dir / f"{stim_id}.png").exists():
        continue
    image_source_path = benchmark._assembly.stimulus_set.get_stimulus(stim_id)
    img = image.imread(image_source_path)
    img = transforms.ToPILImage()(img)
    image_dest_path = images_cache_dir / f"{stim_id}.png"
    img.save(image_dest_path)
    print(f"Saved image {stim_id} to {image_dest_path}")
    print(f"Source path: {image_source_path}")