# Bidirectional Representational Alignment — Code and Experiments

Accompanying code and analyses for the paper 
```
S. Kostousov, A. Kaushik, & B. Laschowski, "Bidirectional Representational Alignment Between Biological and Artificial Neural Networks", bioRxiv, 2026
```
Note that the code an paper are affiliated with the Machine Intelligence Lab at the University of Toronto. 

 This repository provides the training, evaluation, and analysis code used to manipulate and study representational geometry in self-supervised vision models and to evaluate bidirectional representational alignment with biological neural data.

**Summary:**
- **Goal:** Explore how spectral regularization of internal representations (eigenspectrum decay exponent) affects forward and reverse predictivity between artificial neural networks and biological neural recordings, and whether this can improve bidirectional representational alignment.
- **Approach:** Train SimCLR-based ResNet-50 encoders with an added spectral regularization term that penalizes deviations of the eigenspectrum decay exponent from a target value. Evaluate forward predictivity (model -> neurons), reverse predictivity (neurons -> model), and a composite bidirectional predictivity index (BPI).

**Repository Structure**
- `scripts/` — training and experiment entrypoints (`train_simclr.py`, `hpo_train_multi.py`).
- `src/` — core code and analysis utilities:
	- `models/` — model definitions (e.g., `basic_cnn.py`, `simclr_model.py`)
	- `losses/` — contrastive and spectral-regularizer code (`simclr.py`, `spectral_loss.py`)
	- `eval/` — evaluation utilities (`linear_probe.py`, `knn.py`)
	- `latest_neural_data/` — neural-data download and preprocessing helpers (`save_neural_data.py`, `ev_helper.py`,  `moodel_acts.py`)
	- `notebooks/` — analysis & figure-generation notebook (`notebooks/Primary Analysis.ipynb`)
	- `utils/` — dataframe helpers for analysis, downstream analysis, and activation extractors (`construct_df.py`, `model_activations.py`, `post_training.py`)


**Dependencies**
<p>Install the usual PyTorch ecosystem (torch, torchvision, torchaudio) and  scientific Python packages (numpy, scipy, scikit-learn, pandas, matplotlib).
The forward and reverse predictivty metrics are pieced together from the reverse_pred library (https://github.com/vital-kolab/reverse_pred) and can be found in `src/latest_neural_data/ev_helper.py`. 
The neural data is downloaded via the brainscore_vision library (the specific dataset is `MajajHong2015public.IT-pls`). It must be downloaded by running the script src/latest_neural_data/save_neural_data.py before anything else is run.
The code uses optuna for hyperparameter optimization.

**Data**
- ImageNet (used for self-supervised training): experiments in the paper used a subset (100 images per class, balanced split). The repository does not include ImageNet.
- Biological neural data (macaque IT responses) used for evaluation are not included here (can be downloaded via src/latest_neural_data/save_neural_data.py).

**Quickstart**

- Figures: Indicate location of trained model checkpoint and run `notebooks/Primary Analysis.ipynb` to obtain figures.
- Singletraining  run: `scripts/train_simclr.py` with the relevant arguments. For example, to train a model with spectral regularization (lambda=3.0), target alpha=1.0, for 200 epochs, and with a temperature of 0.2, run:

	```bash
	python scripts/train_simclr.py --data-dir /path/to/imagenet --epochs 200 --target_alpha 1.0 --spectral_regularization_strength 3.0 --tau 0.2 --batch-size 512 --lr 0.1	```

- Sweep many configs via the following hyperparameter tuning script: `scripts/hpo_train_multi.py`


**Citation**
If you use this code or ideas in your work, please cite:
```
S. Kostousov, A. Kaushik, & B. Laschowski, "Bidirectional Representational Alignment Between Biological and Artificial Neural Networks", bioRxiv, 2026
```

**Acknowledgements**
code for forward and reverse predictivity evaluation (in src/latest_neural_data/ev_helper.py) is based on the reverse_pred library (https://github.com/vital-kolab/reverse_pred)
```
S. Muzellec, K. Kar, "Reverse Predictivity: Going Beyond One-Way Mapping to Compare Artificial Neural Network Models and Brains", bioRxiv, 2025

```

**Affiliation**

This research was conducted at the [Machine Intelligence Lab](https://github.com/Comp-Neuro-Lab) at the University of Toronto, under the supervision of [Dr. Brokoslaw Laschowski](https://github.com/DrLaschowski).
For related research projects and code, see the [Machine Intelligence Lab GitHub organization](https://github.com/Comp-Neuro-Lab).
