# main.py
from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_configs(config_dir: str) -> Dict[str, Any]:
    order = ["base.yaml", "data.yaml", "model.yaml", "train.yaml", "eval.yaml"]
    cfg: Dict[str, Any] = {}
    for name in order:
        path = os.path.join(config_dir, name)
        if os.path.isfile(path):
            cfg = deep_update(cfg, load_yaml(path))
    return cfg


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_config_snapshot(cfg: Dict[str, Any], output_dir: str) -> None:
    ensure_dir(output_dir)
    snapshot_path = os.path.join(output_dir, "resolved_config.json")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    result = deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override hatalı: {item}. Doğru format: a.b.c=value")
        key, value = item.split("=", 1)
        keys = key.split(".")
        cursor = result
        for k in keys[:-1]:
            if k not in cursor or not isinstance(cursor[k], dict):
                cursor[k] = {}
            cursor = cursor[k]

        low = value.lower()
        if low == "true":
            parsed: Any = True
        elif low == "false":
            parsed = False
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        cursor[keys[-1]] = parsed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RSNA Seminar Project Main Entry")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "build_train_metadata",
            "build_test_metadata",
            "train",
            "eval",
            "miniexp",
        ],
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="base.yaml, data.yaml, model.yaml, train.yaml, eval.yaml klasörü",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Örnek: training.epochs=2 dataset.image_size=256",
    )
    args = parser.parse_args()

    cfg = load_configs(args.config_dir)
    cfg = apply_overrides(cfg, args.override)

    seed = int(cfg["runtime"]["seed"])
    deterministic = bool(cfg["runtime"]["deterministic"])
    set_seed(seed, deterministic)

    output_root = cfg["paths"]["outputs_root"]
    ensure_dir(output_root)
    save_config_snapshot(cfg, output_root)

    if args.mode == "build_train_metadata":
        from src.data.build_rsna_metadata import build_train_metadata

        build_train_metadata(cfg)

    elif args.mode == "build_test_metadata":
        from src.data.build_rsna_test_metadata import build_test_metadata

        build_test_metadata(cfg)

    elif args.mode == "train":
        from src.engine.train import run_training

        run_training(cfg)

    elif args.mode == "eval":
        from src.engine.train import run_evaluation

        run_evaluation(cfg)

    elif args.mode == "miniexp":
        from src.engine.train import run_minimum_experiment

        run_minimum_experiment(cfg)

    else:
        raise ValueError(f"Bilinmeyen mod: {args.mode}")


if __name__ == "__main__":
    main()