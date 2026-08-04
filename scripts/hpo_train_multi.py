"""Small Optuna-based tuner that wraps `train_simclr.main`.

This script defines an `objective` that suggests temperature, spectral loss
coefficients, and target alpha (optionally) and runs `main` to collect
forward/reverse EV metrics for optimization.
"""

import optuna
from scripts.train_simclr import main, parse_args
import argparse
from types import SimpleNamespace
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("--optuna_study_name", type=str, default="study_name")
ap.add_argument("--optuna_db", type=str, default="optuna.db")
ap.add_argument("--tune_spectral_loss_coeff", action="store_true")
ap.add_argument("--tune_target_alpha", action="store_true")
ap.add_argument("--tune_temperature", action="store_true")
ap.add_argument("--random_sampler", action="store_true")
ap.add_argument("--tau_min", type=float, default=0.05)
ap.add_argument("--tau_max", type=float, default=1.0)
ap.add_argument("--tau_step", type=float, default=0.05)
ap.add_argument("--spectral_loss_coeff_min", type=float, default=0.0)
ap.add_argument("--spectral_loss_coeff_max", type=float, default=2.0)
ap.add_argument("--spectral_loss_coeff_step", type=float, default=0.05)
ap.add_argument("--target_alpha_min", type=float, default=0.0)
ap.add_argument("--target_alpha_max", type=float, default=2.0)
ap.add_argument("--target_alpha_step", type=float, default=0.05)

args = parse_args(ap=ap)

if args.random_sampler:
    tau_values = np.arange(args.tau_min, args.tau_max + args.tau_step, args.tau_step).tolist()
    spectral_loss_coeffs = np.arange(args.spectral_loss_coeff_min, args.spectral_loss_coeff_max + args.spectral_loss_coeff_step, args.spectral_loss_coeff_step).tolist()
    target_alpha = np.arange(args.target_alpha_min, args.target_alpha_max + args.target_alpha_step, args.target_alpha_step).tolist()

    study = optuna.create_study(
        storage='sqlite:///{}'.format(args.optuna_db),
        study_name=args.optuna_study_name,
        load_if_exists=True,
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.RandomSampler()
    )
else:
        study = optuna.create_study(
        storage='sqlite:///{}'.format(args.optuna_db),
        study_name=args.optuna_study_name,
        load_if_exists=True,
        directions=['maximize', 'maximize']
    )

def objective(trial):
    """Optuna objective that samples hyperparameters and runs a training job.

    Suggests temperature, spectral loss coeff, and target alpha (depending on
    flags), then calls `main(args)` and returns `(f_ev, r_ev)` as the
    multi-objective target for Optuna.
    """
    args.tag = ""
    if args.tune_temperature:
        if args.random_sampler:
            args.tau = trial.suggest_categorical(f"tau_cat_{args.tau_min}_{args.tau_max}_{args.tau_step}", tau_values)
        else:
            args.tau = trial.suggest_float(f"tau_{args.tau_min}_{args.tau_max}_{args.tau_step}", args.tau_min, args.tau_max, step=args.tau_step)
        args.tag += "_tuning_temperature"
        print(f"tau: {args.tau}")
    if args.tune_spectral_loss_coeff:
        if args.random_sampler:
            args.spectral_loss_coeff = trial.suggest_categorical(f"spectral_loss_coeff_cat_{args.spectral_loss_coeff_min}_{args.spectral_loss_coeff_max}_{args.spectral_loss_coeff_step}", spectral_loss_coeffs)
        else:
            args.spectral_loss_coeff = trial.suggest_float(f"spectral_loss_coeff_{args.spectral_loss_coeff_min}_{args.spectral_loss_coeff_max}_{args.spectral_loss_coeff_step}", args.spectral_loss_coeff_min, args.spectral_loss_coeff_max, step=args.spectral_loss_coeff_step)
        args.tag += "_tuning_spectral_loss_coeff"
        print(f"spectral_loss_coeff: {args.spectral_loss_coeff}")
    if args.tune_target_alpha:
        assert args.skip_alpha == False, "Cannot tune target_alpha if skip_alpha is True"
        assert args.spectral_loss_coeff != 0.0, "Cannot tune target_alpha if spectral_loss_coeff is 0.0"
        if args.random_sampler:
            args.target_alpha = trial.suggest_categorical(f"target_alpha_cat_{args.target_alpha_min}_{args.target_alpha_max}_{args.target_alpha_step}", target_alpha)
        else:
            args.target_alpha = trial.suggest_float(f"target_alpha_{args.target_alpha_min}_{args.target_alpha_max}_{args.target_alpha_step}", args.target_alpha_min, args.target_alpha_max, step=args.target_alpha_step)
        args.tag += "_tuning_target_alpha"
        print(f"target_alpha: {args.target_alpha}")
    
    f_ev, r_ev = main(args)
    return f_ev, r_ev



study.optimize(objective, n_trials=1)