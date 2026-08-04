# this code is based on the reverse_pred library (https://github.com/vital-kolab/reverse_pred) and code is pieced from there.
# article{muzellec_kar_2025_reversepredictivity,
# title   = {Reverse Predictivity: Going Beyond One-Way Mapping to Compare Artificial Neural Network Models and Brains},
# author  = {Muzellec, Sabine and Kar, Kohitij},
# journal = {bioRxiv},
# year    = {2025}

import os
import numpy as np
from sklearn import linear_model
from scipy import stats
import numpy as np

def ridge_regress(X_train, Y_train, X_test, model=None, monkey=None, fold=None):
    clf = linear_model.Ridge(alpha=0.1)
    clf.fit(X_train, Y_train)
    Y_test_pred = clf.predict(X_test)

    if model is not None:
        # Save the weights for later use
        np.save(f'./results_for_figures/model2monkey/{model}_to_{monkey}_ridge_weights_{fold}.npy', clf.coef_)    

    return Y_test_pred

def get_train_test_indices(totalIndices, nrfolds=10, foldnumber=0, seed=1):

    np.random.seed(seed)
    inds = np.arange(totalIndices)
    np.random.shuffle(inds)
    splits = np.array_split(inds, nrfolds)
    test_indices = inds[np.isin(inds, splits[foldnumber])]
    train_indices = inds[np.logical_not(np.isin(inds, test_indices))]
    return train_indices, test_indices


def get_splithalves(var, ax=1, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    # Transpose var so that the split axis becomes axis 0 (easier for shuffling along slices)
    var = np.swapaxes(var, 0, ax)
    
    shuffled = var.copy()
    rng.shuffle(shuffled, axis=0)  # shuffle along the new 0th axis (original ax)
    
    split1, split2 = np.array_split(shuffled, 2, axis=0)
    split_mean1 = np.nanmean(split1, axis=0)
    split_mean2 = np.nanmean(split2, axis=0)

    # Swap axes back to original configuration
    return (
        np.swapaxes(split1, 0, ax),
        np.swapaxes(split2, 0, ax),
        np.swapaxes(split_mean1, 0, ax - 1 if ax > 0 else 0),
        np.swapaxes(split_mean2, 0, ax - 1 if ax > 0 else 0),
    )

def spearmanbrown_correction(var):  # Spearman Brown Correct the correlation value
    spc_var = (2 * var) / (1 + var)
    return spc_var


def get_splithalf_corr(var, ax=1, type='spearman'):
    """
    specify the variable (var) for which splits are required,
    along which axis (ax)?
    which correlation method do you want (type)?
    """
    _, _, split_mean1, split_mean2 = get_splithalves(var, ax=ax)
    if (type == 'spearman'):
        split_half_correlation = stats.spearmanr(split_mean1, split_mean2)  # get the Spearman Correlation
    else:
        split_half_correlation = stats.pearsonr(split_mean1, split_mean2)  # get the Pearson Correlation
    return {'split_half_corr': split_half_correlation[0],
            'p-value': split_half_correlation[1],
            'type': type
            }


def get_splithalf_corr(var, ax=1, type='spearman'):
    _, _, split_mean1, split_mean2 = get_splithalves(var, ax=ax)  # e.g., output shape (samples, neurons)
    
    # Make sure the inputs are 2D
    assert split_mean1.ndim == 2 and split_mean2.ndim == 2, "Split halves must be 2D"

    correlations = []
    for i in range(split_mean1.shape[1]):  # iterate over neurons
        x, y = split_mean1[:, i], split_mean2[:, i]
        if type == 'spearman':
            r, _ = stats.spearmanr(x, y)
        else:
            r, _ = stats.pearsonr(x, y)
        correlations.append(r)

    return {
        'split_half_corr': np.array(correlations),
        'type': type
    }


def get_predictions_multioutput(responses, predictor, ncomp=10, nrfolds=10, seed=0, model=None, monkey=None):
    nrImages, n_targets = responses.shape
    ypred = np.full((nrImages, n_targets), np.nan)

    for i in range(nrfolds):
        train, test = get_train_test_indices(nrImages, nrfolds=nrfolds, foldnumber=i, seed=seed)
        pred = ridge_regress(predictor[train, :], responses[train, :], predictor[test, :], model=model, monkey=monkey, fold=i)
        ypred[test, :] = pred

    return ypred

def get_neural_neural_splithalfcorr(rate_predicted, rate_predictor, ncomp=10, nrfolds=10, seed=0):
    # Split-half correlation of each predicted neuron
    shc_predicted = get_splithalf_corr(rate_predicted, ax=2)  # shape: (n_neurons,) or (n_neurons, n_neurons)
    # Predict using split 1 and split 2 of the predictor
    sp1_predictor, sp2_predictor, _, _ = get_splithalves(rate_predictor, ax=2)

    p1 = get_predictions_multioutput(np.nanmean(rate_predicted, axis=2), np.nanmean(sp1_predictor, axis=2),
                                     nrfolds=nrfolds, ncomp=ncomp, seed=seed)
    p2 = get_predictions_multioutput(np.nanmean(rate_predicted, axis=2), np.nanmean(sp2_predictor, axis=2),
                                     nrfolds=nrfolds, ncomp=ncomp, seed=seed)

    prediction_shc = np.array([stats.pearsonr(p1[:, i], p2[:, i])[0] for i in range(p1.shape[1])])
    prediction_shc = spearmanbrown_correction(prediction_shc)

    mat = shc_predicted['split_half_corr']

    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        diag_vals = np.diag(mat)
    else:
        diag_vals = mat

    neuron_shc = spearmanbrown_correction(diag_vals)

    return prediction_shc, neuron_shc

def get_neural_model_splithalfcorr(model_features, rate, ncomp=10, nrfolds=10, seed=0):
    """
    model_features: shape (n_images, n_model_units) - deterministic
    rate: shape (n_images, n_neurons, n_repeats) - noisy
    """
    sp1, sp2, _, _ = get_splithalves(rate, ax=2)  # split neural responses along repetitions

    # Predict each split of neural data from fixed model features
    p1 = get_predictions_multioutput(model_features, np.nanmean(sp1, axis=2), nrfolds=nrfolds, ncomp=ncomp, seed=seed)
    p2 = get_predictions_multioutput(model_features, np.nanmean(sp2, axis=2), nrfolds=nrfolds, ncomp=ncomp, seed=seed)

    # Compute split-half correlation per neuron
    corr = np.array([stats.pearsonr(p1[:, i], p2[:, i])[0] for i in range(p1.shape[1])])
    model_shc = spearmanbrown_correction(corr)

    return model_shc, 1.0

def get_model_neural_splithalfcorr(rate, model_features, ncomp=10, nrfolds=10, seed=0):
    """
    Predict noisy neural responses from model features.
    - rate: shape (images, neurons, repeats)
    - model_features: shape (images, model_units)
    """
    # Split the rate data along the repetition axis
    sp1, sp2, _, _ = get_splithalves(rate, ax=2)

    # Compute SHC for the neural rate
    shc = get_splithalf_corr(rate, ax=2)

    # Model predictions from averaged neural splits
    target_sp1 = np.nanmean(sp1, axis=2)  # (images, neurons)
    target_sp2 = np.nanmean(sp2, axis=2)  # (images, neurons)

    # Predict both splits from the model features
    p1 = get_predictions_multioutput(target_sp1, model_features, nrfolds=nrfolds, ncomp=ncomp, seed=seed)
    p2 = get_predictions_multioutput(target_sp2, model_features, nrfolds=nrfolds, ncomp=ncomp, seed=seed)

    # Compute split-half correlation of model predictions per neuron
    model_shc = np.array([stats.pearsonr(p1[:, i], p2[:, i])[0] for i in range(p1.shape[1])])
    model_shc = spearmanbrown_correction(model_shc)

    neural_shc = spearmanbrown_correction(shc['split_half_corr'])

    return model_shc, neural_shc

def get_model_neural_splithalfcorr_revamped(rate, model_features, ncomp=10, nrfolds=10, seed=0):

    shc = get_splithalf_corr(rate, ax=2)

    neural_shc = spearmanbrown_correction(shc['split_half_corr'])

    return 1.0, neural_shc

def get_neural_model_splithalfcorr_revamped(model_features, rate, ncomp=10, nrfolds=10, seed=0):

    shc = get_splithalf_corr(rate, ax=2)

    neural_shc = spearmanbrown_correction(shc['split_half_corr'])

    return 1.0, np.nanmean(neural_shc)
    # return 1.0, 1.0

def get_neural_neural_splithalfcorr_revamped(rate1, rate2, ncomp=10, nrfolds=10, seed=0):

    shc1 = get_splithalf_corr(rate1, ax=2)
    shc2 = get_splithalf_corr(rate2, ax=2)

    neural_shc1 = spearmanbrown_correction(shc1['split_half_corr'])
    neural_shc2 = spearmanbrown_correction(shc2['split_half_corr'])

    return np.nanmean(neural_shc1), np.nanmean(neural_shc2)


def get_all_preds(neurons_predicted, neurons_predictor, ncomp, model=None, monkey=None):
    if len(neurons_predicted.shape) == 3:
        mean_target = np.nanmean(neurons_predicted, axis=2)   # shape: (n_images, n_target_neurons)
    else:
        mean_target = neurons_predicted

    if len(neurons_predictor.shape) == 3:
        mean_source = np.nanmean(neurons_predictor, axis=2)   # shape: (n_images, n_source_neurons)
    else:
        mean_source = neurons_predictor
    p = get_predictions_multioutput(mean_target, mean_source, ncomp=ncomp, model=model, monkey=monkey)
    return p

def predictivity(x, y, rho_xx, rho_yy):
    assert x.shape == y.shape, "Input and prediction shapes must match"
    n_neurons = x.shape[1]

    raw_corr = np.array([stats.pearsonr(x[:, i], y[:, i])[0] for i in range(n_neurons)])
    denominator = np.sqrt(rho_xx * rho_yy)
    corrected_raw_corr = raw_corr / denominator
    ev = (corrected_raw_corr ** 2) * 100
    return ev

def get_all_stats(p, neurons_predicted, neurons_predictor, ncomp, unrevamped=False):
    if len(neurons_predicted.shape) == 3:
        mean_target = np.nanmean(neurons_predicted, axis=2)   # shape: (n_images, n_target_neurons)
    else:
        mean_target = neurons_predicted

    # if len(neurons_predicted.shape) == 3 and len(neurons_predictor.shape) == 3:
    #     mshc, nshc = get_neural_neural_splithalfcorr(neurons_predicted, neurons_predictor, ncomp=ncomp)

    if len(neurons_predicted.shape) == 2 and len(neurons_predictor.shape) == 3:
        if unrevamped:
            mshc, nshc = get_neural_model_splithalfcorr(neurons_predicted, neurons_predictor, ncomp=ncomp)
        else:
            mshc, nshc = get_neural_model_splithalfcorr_revamped(neurons_predicted, neurons_predictor, ncomp=ncomp)

    if len(neurons_predicted.shape) == 3 and len(neurons_predictor.shape) == 2:
        if unrevamped:
            mshc, nshc = get_model_neural_splithalfcorr(neurons_predicted, neurons_predictor, ncomp=ncomp)
        else:
            mshc, nshc = get_model_neural_splithalfcorr_revamped(neurons_predicted, neurons_predictor, ncomp=ncomp)

    if len(neurons_predicted.shape) == 2 and len(neurons_predictor.shape) == 2:
        mshc, nshc = 1.0, 1.0

    if len(neurons_predictor.shape) == 3 and len(neurons_predicted.shape) == 3:
        if unrevamped:
            mshc, nshc = get_neural_neural_splithalfcorr(neurons_predicted, neurons_predictor, ncomp=ncomp)
        else:
            mshc, nshc = get_neural_neural_splithalfcorr_revamped(neurons_predicted, neurons_predictor, ncomp=ncomp)

    ev = predictivity(mean_target, p, nshc, mshc)  # Now p and mean_target are 2D
    return ev


def reverse_ev(model_activations, neural_activations, full_ev_vector = False, unrevamped=False, lib=False):
    """Compute reverse explained variance: predict model units from neural data.

    If `full_ev_vector` is True, return per-unit EV; otherwise return the mean.
    """
    responses = np.nanmean(neural_activations, axis=2)
    prediction = get_all_preds(model_activations, responses, ncomp=20)
    ev = get_all_stats(prediction, model_activations, neural_activations, ncomp=20, unrevamped=unrevamped)
    if full_ev_vector:
        return ev
    else:
        return np.nanmean(ev)

def forward_ev(model_activations, neural_activations, full_ev_vector = False, unrevamped=False, lib=False):
    """Compute forward explained variance: predict neural responses from model features.

    Uses split-half reliability to select reliable neurons (default threshold 0.7).
    Returns per-unit EV when `full_ev_vector` is True, otherwise the mean EV.
    """
    shc = get_splithalf_corr(neural_activations, ax=2)["split_half_corr"]
    mask = shc > 0.7
    selected_rates = neural_activations[:, mask]
    selected_responses = np.nanmean(selected_rates, axis=2)
    prediction = get_all_preds(selected_responses, model_activations, ncomp=20)
    ev = get_all_stats(prediction, selected_rates, model_activations, ncomp=20, unrevamped=unrevamped)
    if full_ev_vector:
        return ev
    else:
        return np.nanmean(ev)


