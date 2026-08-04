
"""Helpers to construct a DataFrame summarizing saved training runs.

This module walks a root directory for run folders (containing `ckpts` and `logs`) and
returns a pandas DataFrame with extracted args and stats. It uses a small cache
to avoid re-parsing unchanged run metadata.
"""

import torch
import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
from src.utils.post_training import extract_ckpt_args, extract_stats

def to_float(str):
    """Safely convert a string to float, returning 0 on failure."""
    try:
        flt = float(str)
    except:
        flt = 0
    return flt

to_flt = np.vectorize(to_float)

scratch_path = Path("/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/")


def _load_runs_cache(path: Path):
    """Load a runs cache file (returns empty dict on any error)."""
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            cache = pickle.load(f)
    except Exception:
        return {}
    if not isinstance(cache, dict):
        return {}
    return cache


def _save_runs_cache(path: Path, cache: dict):
    """Persist the runs cache to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f)


def _cache_key(csv_path: str, ckpt_path: str):
    """Compute a stable cache lookup key for a run CSV and checkpoint pair."""
    return f"{csv_path}||{ckpt_path}"


def _cache_path_for_csv(csv_path: str) -> Path:
    return Path(csv_path).with_name("runs_cache.pkl")


def _cache_entry_valid(entry: dict, csv_path: str, ckpt_path: str):
    if not isinstance(entry, dict):
        return False
    try:
        csv_mtime = os.path.getmtime(csv_path)
        ckpt_mtime = os.path.getmtime(ckpt_path)
    except OSError:
        return False
    return (
        entry.get("csv_path") == csv_path
        and entry.get("ckpt_path") == ckpt_path
        and entry.get("csv_mtime") == csv_mtime
        and entry.get("ckpt_mtime") == ckpt_mtime
    )


def _add_datetime_to_args(args: dict):
    ts = args.get("ts")
    if ts is None:
        return
    dt = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
    if pd.notna(dt):
        args["datetime"] = dt


def construct_df(
    root_path: Path = scratch_path,
    show_progress: bool = True,
    force_refetch: bool = False,
) -> pd.DataFrame:
    """Scan `root_path` for runs and return a DataFrame summarizing them.

    The function locates `ckpts/simclr/last.pt` and `logs/simclr_baseline.csv`
    pairs and merges checkpoint args with CSV stats.
    """
    model_paths = []
    for root, dirs, files in os.walk(root_path):
        if os.path.basename(root) == "ckpts":
            folder = os.path.dirname(root)
            model_path = os.path.join(root, "simclr/last.pt")
            csv_path = os.path.join(folder, "logs/simclr_baseline.csv")
            if os.path.exists(csv_path) and os.path.exists(model_path):
                model_paths.append((csv_path, model_path))

    cache_by_path = {}
    updated_cache_paths = set()
    runs = []

    for i, (csv_path, ckpt_path) in enumerate(model_paths):
        if show_progress:
            print(f"retrieving trial {i+1}/{len(model_paths)}", end="\r")

        cache_path = _cache_path_for_csv(csv_path)
        if cache_path not in cache_by_path:
            cache_by_path[cache_path] = _load_runs_cache(cache_path)

        runs_cache = cache_by_path[cache_path]
        key = _cache_key(csv_path, ckpt_path)
        cached_entry = runs_cache.get(key)
        cache_is_valid = (not force_refetch) and _cache_entry_valid(
            cached_entry, csv_path, ckpt_path
        )

        if cache_is_valid:
            status = cached_entry.get("status", "ok")
            if status == "missing_version":
                continue
            if status == "ok" and "run" in cached_entry:
                args = dict(cached_entry["run"])
                _add_datetime_to_args(args)
            else:
                cache_is_valid = False

        if not cache_is_valid:
            args = extract_ckpt_args(ckpt_path)
            stats = extract_stats(csv_path)
            if "version" not in stats:
                runs_cache[key] = {
                    "csv_path": csv_path,
                    "ckpt_path": ckpt_path,
                    "csv_mtime": os.path.getmtime(csv_path),
                    "ckpt_mtime": os.path.getmtime(ckpt_path),
                    "status": "missing_version",
                }
                updated_cache_paths.add(cache_path)
                continue

            for k, value in stats.items():
                if k not in args:
                    args[k] = value
            args["ckpt_path"] = ckpt_path
            args["csv_path"] = csv_path
            _add_datetime_to_args(args)

            runs_cache[key] = {
                "csv_path": csv_path,
                "ckpt_path": ckpt_path,
                "csv_mtime": os.path.getmtime(csv_path),
                "ckpt_mtime": os.path.getmtime(ckpt_path),
                "status": "ok",
                "run": dict(args),
            }
            updated_cache_paths.add(cache_path)

        runs.append(args)

    for path in updated_cache_paths:
        _save_runs_cache(path, cache_by_path[path])

    df = pd.DataFrame(runs)
    return df