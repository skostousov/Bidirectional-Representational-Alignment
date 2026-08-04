"""Main training script for SimCLR-style self-supervised runs.

This module provides `parse_args()` and `main()` used to run single training
jobs. It implements the SimCLR InfoNCE objective, optional debiased variant,
and an optional spectral regularizer applied to a chosen encoder layer.
"""

import argparse
import os
import time
import csv
import random
import numpy as np
import copy

import torch
import torch.nn.functional as F
from torch import optim
from torch.amp import GradScaler, autocast

from src.datamod.imagenet_ssl import (
    build_ssl_train_loader,
    build_ssl_val_loader,
    build_eval_loaders,
)

from src.datamod.cifar10_ssl import (
    build_ssl_train_loader_cifar10,
    build_ssl_val_loader_cifar10,
    build_eval_loaders_cifar10,
)
from src.models.simclr_model import SimCLR, SimCLR_Aux, downsizedSimCLR
from src.losses.simclr import info_nce, debiased_info_nce
from src.losses.spectral_loss import just_alpha_fixed, spectral_loss, just_alpha
from src.losses.group_sparsity_loss import aux_loss
from src.utils.model_activations import ModelActivations

from src.eval.knn import knn_top1
from src.eval.linear_probe import linear_probe_top1

from src.metrics.FR_EV_offline import F_R_EV, three_channel_transform

from src.latest_neural_data.ev_helper import forward_ev, reverse_ev
from src.latest_neural_data.model_acts import extract_model_activations_from_cache



# Utilities
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr_with_warmup(
    optimizer,
    base_lr,
    epoch,
    step_in_epoch,
    steps_per_epoch,
    warmup_epochs,
    total_epochs,
):
    """
    Per-step cosine schedule with linear warmup.
    """
    global_step = epoch * steps_per_epoch + step_in_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    if warmup_steps > 0 and global_step < warmup_steps:
        lr = base_lr * float(global_step + 1) / float(warmup_steps)
    else:
        t = (global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        lr = 0.5 * base_lr * (1.0 + np.cos(np.pi * t))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# Affinity statistics helper
@torch.no_grad()
def compute_affinity_stats(t: torch.Tensor, thr: float = 0.7):
    """
    Compute summary statistics of teacher affinity matrix.

    Args:
        t: [2B, D] normalized teacher embeddings
        thr: near-positive threshold on r_ij

    Returns:
        dict with mean_r, median_r, tail95_r, frac_near_pos
    """
    # cosine similarity
    r = t @ t.t()  # [-1, 1]
    r = 0.5 * (r + 1.0)  # -> [0, 1]

    B2 = r.size(0)
    diag = torch.eye(B2, device=r.device, dtype=torch.bool)
    r = r.masked_fill(diag, float("nan"))

    flat = r[~torch.isnan(r)]

    mean_r = flat.mean().item()
    median_r = flat.median().item()
    tail95_r = torch.quantile(flat.float(), 0.95).item()
    frac_near = (flat > thr).float().mean().item()

    return {
        "mean_r": mean_r,
        "median_r": median_r,
        "tail95_r": tail95_r,
        "frac_near_pos": frac_near,
    }


@torch.no_grad()
def ssl_val_infonce_and_alpha(
    model,
    ssl_val_dl,
    tau,
    device,
    activationclass=None,
    alpha_layer=None,
    compute_alpha=True,
    *,
    use_debiased: bool = False,
    teacher=None,
    teacher_feat: str = "encoder",
    gamma: float = 1.0,
):
    """Validate using the SAME SSL objective as training.

    - If `use_debiased` is False: standard InfoNCE.
    - If `use_debiased` is True: debiased InfoNCE with teacher-derived weights.

    Also optionally aggregates alpha over the full SSL val set.
    """
    model.eval()
    if teacher is not None:
        teacher.eval()

    total_loss = 0.0
    n = 0
    alphas = []

    for q, k in ssl_val_dl:
        q = q.to(device, non_blocking=True).contiguous()
        k = k.to(device, non_blocking=True).contiguous()

        z1 = model(q)
        z2 = model(k)

        if use_debiased:
            if teacher is None:
                raise ValueError("ssl_val_infonce_and_alpha: use_debiased=True requires a teacher")
            # Teacher features for affinity weights (no grad)
            ht1 = teacher.encoder(q)
            ht2 = teacher.encoder(k)
            if teacher_feat == "encoder":
                t1, t2 = ht1, ht2
            else:
                t1, t2 = teacher.proj(ht1), teacher.proj(ht2)

            loss = debiased_info_nce(z1, z2, t1, t2, tau=tau, gamma=gamma)
        else:
            loss = info_nce(z1, z2, tau=tau)

        total_loss += loss.item()
        n += 1

        if compute_alpha and activationclass is not None and alpha_layer is not None:
            a = just_alpha_fixed(activationclass.activations[alpha_layer], device=device)
            alphas.append(float(a.detach().float().cpu().item()))

    model.train()
    avg_loss = total_loss / max(1, n)
    avg_alpha = float(np.mean(alphas)) if (compute_alpha and len(alphas) > 0) else 0.0
    return avg_loss, avg_alpha



@torch.no_grad()
def spectrum_pr(model, dl, device, batches=2, max_per_batch=64, use="z"):
    """
    Participation Ratio (PR) + eigen stats from a few batches.
    Mirrors old spectrum_probe, but uses torch end-to-end.
    use:
      - "z": projection output (matches old model(q))
      - "h": encoder features
    """
    global args
    model.eval()
    ys = []
    it = 0

    for batch in dl:
        # SSL: (q,k) ; supervised eval: (x,y)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x = batch[0]
        else:
            x = batch

        x = x[:max_per_batch].to(device, non_blocking=True).contiguous()
        model_encoder = model.module.encoder if isinstance(model, torch.nn.DataParallel) else model.encoder
        model_proj = model.module.proj if isinstance(model, torch.nn.DataParallel) else model.proj
        h = model_encoder(x)
        if use == "z":
            y = model_proj(h)
            y = F.normalize(y, dim=1)
        else:
            y = F.normalize(h, dim=1)

        ys.append(y.detach().float().cpu())
        it += 1
        if it >= batches:
            break

    if len(ys) == 0:
        model.train()
        return 0.0, 0.0, 0.0

    Y = torch.cat(ys, dim=0)  # [N, D] on CPU
    Y = Y - Y.mean(dim=0, keepdim=True)
    C = (Y.T @ Y) / max(1, Y.shape[0])  # [D, D]
    eig = torch.linalg.eigvalsh(C)      # ascending
    eig = torch.clamp(eig, min=0)

    pr = (eig.sum() ** 2) / (eig.pow(2).sum() + 1e-12)
    lam1 = eig[-1]
    lam_min = eig[0]

    model.train()
    return float(pr.item()), float(lam1.item()), float(lam_min.item())

# def parse_args_ckpt():
#     ap = argparse.ArgumentParser()    
    
#     ap.add_argument("--ckpt_path", type=str, required=True)

#     return ap.parse_args() lol ignore this 



def parse_args(ap=None):
    if ap is None:
        ap = argparse.ArgumentParser()    

    ap.add_argument("--imagenet_root", type=str, required=False)
    ap.add_argument("--cifar10_root", type=str, required=False)

    ap.add_argument("--parallel", action="store_true", help="Use torch.nn.DataParallel (not recommended)")

    ap.add_argument("--tag", type=str, default="", help="Optional tag to identify the run (appended to save_dir)")

    # core
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--warmup_epochs", type=int, default=10)

    ap.add_argument("--downsized_resnet", action="store_true", help="Use resnet18 with fewer channels for faster iteration and testing")

    # stability toggles
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=0.0, help="0 disables")

    # subset bring-up
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)

    # logging + saving
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_dir", type=str, default="", help="empty disables")

    # metrics switches
    ap.add_argument("--skip_knn", action="store_true")
    ap.add_argument("--skip_alpha", action="store_true")
    ap.add_argument("--skip_neural_ev", action="store_true")
    ap.add_argument("--skip_linear_probe", action="store_true")
    ap.add_argument("--skip_pr", action="store_true")
    ap.add_argument("--skip_ssl_val", action="store_true")

    # evaluation cadnce
    ap.add_argument("--eval_every", type=int, default=10, help="run knn/linear probe/pr every N epochs")

    # linear pro=be specifics
    ap.add_argument("--lp_epochs", type=int, default=5)
    ap.add_argument("--lp_lr", type=float, default=0.1)
    ap.add_argument("--lp_wd", type=float, default=0.0)

    # spectral loss
    ap.add_argument("--spectral_loss_coeff", type=float, default=0.0)
    ap.add_argument("--spectral_loss_warmup_epochs", type=int, default=0)
    ap.add_argument("--target_alpha", type=float, default=1.0)

    # auxiliary loss
    ap.add_argument("--use_aux_loss", action="store_true", help="Use auxiliary loss (group sparsity + KL)")
    ap.add_argument("--aux_loss_coeff", type=float, default=0.0)
    ap.add_argument("--groupsize", type=int, default=8, help="Group size for auxiliary loss (only relevant if --use_aux_loss)")
    
    # neural EV
    ap.add_argument("--neural_ev_layer", type=str, default="encoder.layer4.0.bn1")
    # ap.add_argument("--neural_data_dir", type=str, default="src/metrics/neural_data")
    ap.add_argument("--neural_data_dir", type=str, default="src/REVERSE_PRED_FINAL/majajhong_cache")

    # seed
    ap.add_argument("--seed", type=int, default=0)

    # EMA teacher (for debiasedcontrastive learning)
    ap.add_argument("--use_ema_teacher", action="store_true", help="Enable EMA teacher over encoder")
    ap.add_argument("--ema_m", type=float, default=0.996, help="EMA momentum for teacher update")

    # Teacher feature space selection for affinity
    ap.add_argument(
        "--teacher_feat",
        type=str,
        default="encoder",
        choices=["encoder", "proj"],
        help="Which teacher features to use for affinity: encoder (Option B) or proj (Option A)",
    )

    # Debiased contrastive learning (teacher affinity weights)
    ap.add_argument("--use_debiased", action="store_true",
                    help="Use teacher affinity weights in InfoNCE denominator")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Exponent for weights: w_ij ∝ (1 - r_ij)^gamma")

    subparser = ap.add_subparsers(dest="command")
    ckpt_parser = subparser.add_parser("ckpt")
    ckpt_parser.add_argument("--ckpt_path", type=str, required=True)
    ckpt_parser.add_argument("--more_epochs", type=int, default=0)

    return ap.parse_args()


def main(args=None):
    """Run a training run based on parsed `args`.

    Returns a tuple `(forward_ev, reverse_ev)` collected at the end of
    training (or immediately for `ckpt` commands). The function coordinates
    data loaders, model construction, training loop, evaluation hooks, and
    checkpoint saving according to CLI flags.
    """
    f_ev, r_ev = 0.0, 0.0

    version = "fixed alpha_loss_04042026"

    if args is None:
        args = parse_args()
    set_seed(args.seed)

    if args.command == "ckpt":
        # load chckpoint and override args
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        parent_dir = os.path.dirname(args.ckpt_path)
        torch.save(ckpt, os.path.join(parent_dir, "previous_scinet_run.pt"))
        image_net_root = args.imagenet_root
        ckpt_args = ckpt.get("args", {})
        for k, v in ckpt_args.items():
            if hasattr(args, k):
                if k not in ["epochs", "imagenet_root", "more_epochs", "ckpt_path"]:
                    if v == "True":
                        v = True
                    elif v == "False":
                        v = False
                    setattr(args, k, v)
        epochs_completed = ckpt.get("epoch", 0)
        print(f"number of more epochs to train: {args.more_epochs}")
        args.epochs = min(epochs_completed + args.more_epochs, args.max_epochs)
        print(f"epochs completed: {epochs_completed}, total epochs now: {args.epochs}") 
        print(f"Resuming from checkpoint {args.ckpt_path}, continuing to epoch {epochs_completed + 1}")
        save_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.ckpt_path)))
        best_ssl_val = ckpt.get("best_ssl_val", float("inf"))
        best_linear_probe = ckpt.get("best_linear_probe", 0.0)

    else:
        # save dir
        start_ts = time.strftime("%Y%m%d-%H%M%S")
        save_dir = args.save_dir.strip()
        if save_dir:
            if args.tag != "":
                save_dir = os.path.join(save_dir, f"start_{start_ts}_{args.tag}")
            else:
                save_dir = os.path.join(save_dir, f"start_{start_ts}")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(os.path.join(save_dir, "ckpts", "simclr"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
        else:
            os.makedirs("ckpts/simclr", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
        epochs_completed = 0
        best_ssl_val = float("inf")
        best_linear_probe = 0.0

    # imagenet_root validation
    if ((args.imagenet_root is None) or (str(args.imagenet_root).strip() == "")) and ((args.cifar10_root is None) or (str(args.cifar10_root).strip() == "")):
        if args.command == "ckpt":
            # try to recover from checkpoint args if present
            ckpt_args = ckpt.get("args", {}) if "ckpt" in locals() else {}
            imagenet_root_from_ckpt = ckpt_args.get("imagenet_root", "")
            cifar_root_from_ckpt = ckpt_args.get("cifar10_root", "")
            if str(imagenet_root_from_ckpt).strip() != "":
                args.imagenet_root = imagenet_root_from_ckpt
                print(f"imagenet_root not provided; using checkpoint imagenet_root={args.imagenet_root}")
            elif str(cifar_root_from_ckpt).strip() != "":
                args.cifar10_root = cifar_root_from_ckpt
                print(f"cifar10_root not provided; using checkpoint cifar10_root={args.cifar10_root}")
            else:
                raise ValueError("--imagenet_root must be provided when resuming unless it exists in the checkpoint args")
        else:    
            raise ValueError("--imagenet_root is required")

    # device preference
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)

    pin_memory = (device == "cuda")
    amp_enabled = (args.amp and device == "cuda")
    scaler = GradScaler("cuda") if amp_enabled else None



    print(f"spectral loss coeff: {args.spectral_loss_coeff}")
    print(f"tau={args.tau} bs={args.batch_size} lr={args.lr} wd={args.wd} warmup={args.warmup_epochs} epochs={args.epochs}")
    print(f"workers={args.workers} amp={amp_enabled} grad_clip={args.grad_clip}")
    print(f"eval_every={args.eval_every} (knn/linear probe/pr), seed={args.seed}")
    print(
        f"debiased={bool(args.use_debiased)} gamma={args.gamma} "
        f"ema_teacher={bool(args.use_ema_teacher)} teacher_feat={args.teacher_feat}"
    )

    # Loaders
    if args.imagenet_root:
        ssl_train_dl = build_ssl_train_loader(
        root=args.imagenet_root,
        batch_size=args.batch_size,
        workers=args.workers,
        img_size=args.img_size,
        pin_memory=pin_memory,
        limit_train=args.limit_train,
        )
        ssl_val_dl = build_ssl_val_loader(
        root=args.imagenet_root,
        batch_size=args.batch_size,
        workers=args.workers,
        img_size=args.img_size,
        pin_memory=pin_memory,
        limit_val=args.limit_val,
        )
        # clean eval loaders for kNN + linear probe
        eval_tr_dl, eval_va_dl = build_eval_loaders(
        root=args.imagenet_root,
        batch_size=args.batch_size,
        workers=args.workers,
        img_size=args.img_size,
        pin_memory=pin_memory,
        limit_train=None,
        limit_val=None,
        )
    elif args.cifar10_root:
        ssl_train_dl = build_ssl_train_loader_cifar10(
        root=args.cifar10_root,
        batch_size=args.batch_size,
        workers=args.workers,
        img_size=args.img_size,
        pin_memory=pin_memory,
        limit_train=args.limit_train,
        )
        ssl_val_dl = build_ssl_val_loader_cifar10(
        root=args.cifar10_root,
        batch_size=args.batch_size,
        workers=args.workers,
        img_size=args.img_size,
        pin_memory=pin_memory,
        limit_val=args.limit_val,
        )
        # clean eval loaders for kNN + linear probe
        eval_tr_dl, eval_va_dl = build_eval_loaders_cifar10(
        root=args.cifar10_root,
        batch_size=args.batch_size,
        workers=args.workers,
        img_size=args.img_size,
        pin_memory=pin_memory,
        limit_train=None,
        limit_val=None,
        )

    base_ds = eval_tr_dl.dataset.dataset if hasattr(eval_tr_dl.dataset, "dataset") else eval_tr_dl.dataset
    num_classes = len(base_ds.classes)

    # Model + hooks
    if args.use_aux_loss:
        groupsize = args.groupsize
        model = SimCLR_Aux(out_dim=128, group_size=groupsize).to(device)
    elif args.downsized_resnet:
        model = downsizedSimCLR(out_dim=128).to(device)
    else:
        model = SimCLR(out_dim=128).to(device)
    if args.command == "ckpt":
        model.load_state_dict(ckpt["model"])

    if args.parallel:
        model = torch.nn.DataParallel(model)
    # EMA teacher setup (full model: encoder + proj)
    teacher = None
    if args.use_ema_teacher:
        teacher = copy.deepcopy(model).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # If resuming, load teacher state if present
        if args.command == "ckpt" and ("teacher" in ckpt) and (ckpt["teacher"] is not None):
            teacher.load_state_dict(ckpt["teacher"])

        print(f"EMA teacher enabled: ema_m={args.ema_m}")

    if args.use_debiased and (teacher is None):
        raise ValueError("--use_debiased requires --use_ema_teacher (EMA teacher not enabled)")

    if args.parallel and not args.neural_ev_layer.startswith("module."):
        args.neural_ev_layer = f"module.{args.neural_ev_layer}"
    activationclass = ModelActivations(model, layers=[args.neural_ev_layer])
    activationclass.register_hooks()
    print(f"Registered activation layers: {activationclass.layers}")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.command == "ckpt":
        optimizer.load_state_dict(ckpt["opt"])

    # logging
    log_path = os.path.join(save_dir, "logs", "simclr_baseline.csv") if save_dir else "logs/simclr_baseline.csv"
    header = [
        "ts", "epoch", "lr", "tau", "batch_size", "img_size",
        "train_loss", "ssl_val_loss",
        "knn_top1", "linear_probe_top1",
        "PR_z", "lam1_z", "lam_min_z",
        "alpha", "spectral_loss_coeff",
        "BPI", "F_EV", "R_EV",
        "device", "seed", "spec_loss_warmup_epochs", "use_debiased", "gamma",
        "teacher_feat",
        "mean_r", "median_r", "tail95_r", "frac_near_pos", "downsized_resnet", "save_note", "version"
    ]
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # Train
    model.train()
    steps_per_epoch = len(ssl_train_dl)

    for epoch in range(epochs_completed, args.epochs):
        epoch_loss = 0.0
        n_steps = 0

        # Affinity stats accumulator
        epoch_aff_stats = {
            "mean_r": [],
            "median_r": [],
            "tail95_r": [],
            "frac_near_pos": [],
        }

        optimizer.zero_grad(set_to_none=True)

        for it, (q, k) in enumerate(ssl_train_dl):
            q = q.to(device, non_blocking=True).contiguous()
            k = k.to(device, non_blocking=True).contiguous()

            lr_now = cosine_lr_with_warmup(
                optimizer,
                base_lr=args.lr,
                epoch=epoch,
                step_in_epoch=it,
                steps_per_epoch=steps_per_epoch,
                warmup_epochs=args.warmup_epochs,
                total_epochs=args.epochs,
            )

            if amp_enabled:
                with autocast("cuda"):
                    if args.use_aux_loss:
                        z1, aux1 = model(q)
                        z2, aux2 = model(k)
                    else:
                        z1 = model(q)
                        z2 = model(k)

                    if args.use_debiased:
                        # Teacher affinity weights (no grad)
                        with torch.no_grad():
                            ht1 = teacher.encoder(q)
                            ht2 = teacher.encoder(k)

                            if args.teacher_feat == "encoder":
                                # Option B: encoder space (recommended for neural fidelity)
                                t1 = ht1
                                t2 = ht2
                            else:
                                # Option A: projection space (SimCLR-aligned)
                                t1 = teacher.proj(ht1)
                                t2 = teacher.proj(ht2)
                        # Affinity stats
                        with torch.no_grad():
                            t = torch.cat([F.normalize(t1.float(), dim=1),
                                           F.normalize(t2.float(), dim=1)], dim=0)
                            stats = compute_affinity_stats(t, thr=0.7)
                            for k_stat in epoch_aff_stats:
                                epoch_aff_stats[k_stat].append(stats[k_stat])
                        l1 = debiased_info_nce(z1, z2, t1, t2, tau=args.tau, gamma=args.gamma)
                    else:
                        l1 = info_nce(z1, z2, tau=args.tau)

                    if args.spectral_loss_coeff != 0.0 and epoch >= int(args.spectral_loss_warmup_epochs):
                        acts = activationclass.activations[args.neural_ev_layer]
                        l2, alpha = spectral_loss(acts, device, target_alpha=args.target_alpha)

                        assert acts.requires_grad
                        assert l2.requires_grad

                    else:
                        l2 = torch.tensor(0.0, device=device)
                        alpha = torch.tensor(0.0, device=device)

                    if args.use_aux_loss:
                        aux_loss = aux_loss(aux1, args.lam_aux)
                        l2 = l2 + args.aux_loss_coeff * aux_loss

                    loss = l1 + args.spectral_loss_coeff * l2
                    loss = loss / args.accum_steps

                scaler.scale(loss).backward()

                if (it + 1) % args.accum_steps == 0:
                    if args.grad_clip and args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                    scaler.step(optimizer)
                    scaler.update()

                    # EMA update after the student step
                    if teacher is not None:
                        with torch.no_grad():
                            m = float(args.ema_m)
                            for ps, pt in zip(model.parameters(), teacher.parameters()):
                                pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)

                    optimizer.zero_grad(set_to_none=True)

            else:
                z1 = model(q)
                z2 = model(k)

                if args.use_debiased:
                    # Teacher affinity weights (no grad)
                    with torch.no_grad():
                        ht1 = teacher.encoder(q)
                        ht2 = teacher.encoder(k)

                        if args.teacher_feat == "encoder":
                            # Option B: encoder space (recommended for neural fidelity)
                            t1 = ht1
                            t2 = ht2
                        else:
                            # Option A: projection space (SimCLR-aligned)
                            t1 = teacher.proj(ht1)
                            t2 = teacher.proj(ht2)
                    # Affinity stats
                    with torch.no_grad():
                        t = torch.cat([F.normalize(t1.float(), dim=1),
                                       F.normalize(t2.float(), dim=1)], dim=0)
                        stats = compute_affinity_stats(t, thr=0.7)
                        for k_stat in epoch_aff_stats:
                            epoch_aff_stats[k_stat].append(stats[k_stat])
                    l1 = debiased_info_nce(z1, z2, t1, t2, tau=args.tau, gamma=args.gamma)
                else:
                    l1 = info_nce(z1, z2, tau=args.tau)

                if args.spectral_loss_coeff != 0.0 and epoch >= int(args.spectral_loss_warmup_epochs):
                    acts = activationclass.activations[args.neural_ev_layer]
                    l2, alpha = spectral_loss(acts, device, target_alpha=args.target_alpha)

                    assert acts.requires_grad
                    assert l2.requires_grad

                else:
                    l2 = torch.tensor(0.0, device=device)
                    alpha = torch.tensor(0.0, device=device)

                loss = l1 + args.spectral_loss_coeff * l2
                loss = loss / args.accum_steps
                loss.backward()

                if (it + 1) % args.accum_steps == 0:
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                    # EMA update after the student step
                    if teacher is not None:
                        with torch.no_grad():
                            m = float(args.ema_m)
                            for ps, pt in zip(model.parameters(), teacher.parameters()):
                                pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)

                    optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * args.accum_steps
            n_steps += 1

            if (it + 1) % args.log_every == 0:
                print(f"epoch {epoch+1} iter {it+1}/{steps_per_epoch} lr {lr_now:.5f} loss {epoch_loss/max(1,n_steps):.4f} | alpha {alpha:.3f}")

        train_avg = epoch_loss / max(1, n_steps)
        print(f"--- Epoch {epoch+1} done | avg train loss {train_avg:.4f} ---")

        do_eval = ((not (args.eval_every == 0)) and ((epoch + 1) % args.eval_every == 0)) or (epoch + 1 == args.epochs)

        # SSL val loss (proper)
        if args.skip_ssl_val and not do_eval:
            ssl_val_avg, val_alpha = 0.0, 0.0
        else:
            ssl_val_avg, val_alpha = ssl_val_infonce_and_alpha(
                model,
                ssl_val_dl,
                tau=args.tau,
                device=device,
                activationclass=activationclass,
                alpha_layer=args.neural_ev_layer,
                compute_alpha=(not args.skip_alpha),
                use_debiased=bool(args.use_debiased),
                teacher=teacher,
                teacher_feat=args.teacher_feat,
                gamma=float(args.gamma),
            )
        val_tag = "Debiased" if bool(args.use_debiased) else "InfoNCE"
        print(f"epoch {epoch+1} | ssl val {val_tag} {ssl_val_avg:.4f} | alpha {val_alpha:.3f}")
        
        # Alpha metric (optional)
        # if args.skip_alpha:
        #     val_alpha = 0.0
        # # else:
        # #     with torch.no_grad():
        # #         q, _ = next(iter(ssl_val_dl))
        # #         q = q.to(device, non_blocking=True).contiguous()
        # #         _ = model(q)
        # #         val_alpha = float(just_alpha(activationclass.activations[args.neural_ev_layer], device=device).cpu().item())
        # print(f"epoch {epoch+1} | alpha {val_alpha:.3f}")

        # kNN + linear probe + PR (cadenced)

        knn_acc = 0.0
        lp_acc = 0.0
        pr_z = 0.0
        lam1_z = 0.0
        lammin_z = 0.0
        bpi = 0.0
        f_ev = 0.0
        r_ev = 0.0

        if do_eval:
            model_enc = model.module.encoder if args.parallel else model.encoder
            if not args.skip_knn:
                knn_acc = knn_top1(model_enc, eval_tr_dl, eval_va_dl, device=device, k=1)
                print(f"epoch {epoch+1} | kNN(1) top1 {knn_acc:.2f}%")

            if not args.skip_linear_probe:
                lp_acc = linear_probe_top1(
                    model_enc, eval_tr_dl, eval_va_dl, num_classes=num_classes,
                    device=device, epochs=args.lp_epochs, lr=args.lp_lr, wd=args.lp_wd
                )
                print(f"epoch {epoch+1} | linear probe top1 {lp_acc:.2f}%")

            if not args.skip_pr:
                pr_z, lam1_z, lammin_z = spectrum_pr(
                    model, ssl_train_dl, device=device, batches=2, max_per_batch=64, use="z"
                )
                print(f"epoch {epoch+1} | PR(z) {pr_z:.1f} | lam1 {lam1_z:.3g} | lam_min {lammin_z:.3g}")
            
            if not args.skip_neural_ev:
                try:
                    neural_activations = np.load(os.path.join(args.neural_data_dir, "neural_activations.npy"))

                    model_activations, stimulus_ids = extract_model_activations_from_cache(
                            model=model,
                            cache_dir=args.neural_data_dir,#"REVERSE_PRED_FINAL/majajhong_cache"
                            layer_name=args.neural_ev_layer,  # Auto-detect
                            batch_size=32
                        )
                    f_ev, r_ev = forward_ev(model_activations, neural_activations, unrevamped=True), reverse_ev(model_activations, neural_activations, unrevamped=True)
                    bpi = 2*f_ev*r_ev/(f_ev + r_ev + 1e-12)
                    
                    # ev_dict = F_R_EV(
                    # model,
                    # activation_layer=args.neural_ev_layer,
                    # neural_data_dir=args.neural_data_dir,
                    # alpha=0.5,
                    # transforms=three_channel_transform,
                    # reliability_threshold=0.7,
                    # batch_size=4,)
                    # bpi, f_ev, r_ev = ev_dict["BPI"], ev_dict["F_EV"], ev_dict["R_EV"]

                    print(f"epoch {epoch+1} | BPI {bpi:.3f} | F_EV {f_ev:.3f} | R_EV {r_ev:.3f}")
                except Exception as e:
                    print(f"epoch {epoch+1} | Error computing neural EV: {e}")
                    bpi, f_ev, r_ev = 0.0, 0.0, 0.0
            # else:
            #     ev_dict = {"BPI": 0.0, "F_EV": 0.0, "R_EV": 0.0}
            #     bpi, f_ev, r_ev = ev_dict["BPI"], ev_dict["F_EV"], ev_dict["R_EV"]


    
        # # Neural EV (optional, expensive)
    
        # if args.skip_neural_ev:
        #     ev_dict = {"BPI": 0.0, "F_EV": 0.0, "R_EV": 0.0}
        # else:
        #     ev_dict = F_R_EV(
        #         model,
        #         activation_layer=args.neural_ev_layer,
        #         neural_data_dir=args.neural_data_dir,
        #         alpha=0.5,
        #         transforms=three_channel_transform,
        #         reliability_threshold=0.7,
        #         batch_size=4,
        #     )

        # bpi, f_ev, r_ev = ev_dict["BPI"], ev_dict["F_EV"], ev_dict["R_EV"]
        # if not args.skip_neural_ev:
        #     print(f"epoch {epoch+1} | BPI {bpi:.3f} | F_EV {f_ev:.3f} | R_EV {r_ev:.3f}")

        # Checkpointing
        ts = int(time.time())
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "teacher": (teacher.state_dict() if teacher is not None else None),
            "args": vars(args),
            "best_ssl_val": best_ssl_val,
            "best_linear_probe": best_linear_probe,
        }

        ckpt_dir = os.path.join(save_dir, "ckpts", "simclr") if save_dir else "ckpts/simclr"
        last_path = os.path.join(ckpt_dir, "last.pt")
        torch.save(ckpt, last_path)

        save = "last.pt"

        if ssl_val_avg < best_ssl_val:
            best_ssl_val = ssl_val_avg
            best_path = os.path.join(ckpt_dir, "best_ssl_val.pt")
            torch.save(ckpt, best_path)
            save = "best_ssl_val.pt"
            print(f"epoch {epoch+1} | new best ssl val {best_ssl_val:.4f} -> saved best_ssl_val.pt")

        if lp_acc > best_linear_probe:
            best_linear_probe = lp_acc
            best_lp_path = os.path.join(ckpt_dir, "best_linear_probe.pt")
            torch.save(ckpt, best_lp_path)
            save = "best_linear_probe.pt"
            print(f"epoch {epoch+1} | new best linear probe {best_linear_probe:.2f}% -> saved best_linear_probe.pt")

        # Affinity stats: compute epoch means
        if args.use_debiased and len(epoch_aff_stats["mean_r"]) > 0:
            mean_r = float(np.mean(epoch_aff_stats["mean_r"]))
            median_r = float(np.mean(epoch_aff_stats["median_r"]))
            tail95_r = float(np.mean(epoch_aff_stats["tail95_r"]))
            frac_near_pos = float(np.mean(epoch_aff_stats["frac_near_pos"]))
        else:
            mean_r = median_r = tail95_r = frac_near_pos = 0.0

        # Log row
        row = [
            ts, epoch + 1, optimizer.param_groups[0]["lr"], args.tau, args.batch_size, args.img_size,
            train_avg, ssl_val_avg,
            knn_acc, lp_acc,
            pr_z, lam1_z, lammin_z,
            val_alpha, args.spectral_loss_coeff,
            bpi, f_ev, r_ev,
            device, args.seed, args.spectral_loss_warmup_epochs, int(args.use_debiased), float(args.gamma),
            args.teacher_feat,
            mean_r, median_r, tail95_r, frac_near_pos,
            args.downsized_resnet, save, version
        ]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
    
    return f_ev, r_ev


if __name__ == "__main__":
    main()
