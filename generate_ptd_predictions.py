#!/usr/bin/env python3

"""
Generate dev + train predictions for Paraphrase Type Detection (PTD) using
the existing top-3 ensemble checkpoints (no retraining needed).

Usage:
    python generate_ptd_predictions.py --use_gpu
"""

import argparse
import os

import pandas as pd
import torch

from bart_detection import (
    find_optimal_thresholds,
    predict_ensemble,
    safe_train_test_split,
    test_ensemble,
    transform_data,
)

BEST_3_PATHS = [
    "models/bart_ptd_epoch6.pt",
    "models/bart_ptd_epoch10.pt",
    "models/bart_ptd_epoch7.pt",
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    for p in BEST_3_PATHS:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing checkpoint: {p}")

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    train_df, dev_df = safe_train_test_split(train_dataset, test_size=0.2, random_state=args.seed)
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    print(f"train={len(train_df)} dev={len(dev_df)} (same 80/20 split used during training)")

    train_data = transform_data(train_df, is_training=False, batch_size=args.batch_size)
    dev_data = transform_data(dev_df, is_training=False, batch_size=args.batch_size)

    print("Running top-3 ensemble on dev set to calibrate thresholds...")
    dev_ens_probs, dev_labels = predict_ensemble(BEST_3_PATHS, dev_data, device)
    optimal_thresholds = find_optimal_thresholds(dev_labels, dev_ens_probs)

    os.makedirs("predictions/bart", exist_ok=True)

    dev_results = test_ensemble(BEST_3_PATHS, dev_data, dev_df["id"], device, thresholds=optimal_thresholds)
    dev_out = "predictions/bart/etpc-paraphrase-detection-dev-output.csv"
    dev_results.to_csv(dev_out, index=False)
    print(f"Wrote {len(dev_results)} rows to {dev_out}")

    print("Running top-3 ensemble on train set...")
    train_results = test_ensemble(BEST_3_PATHS, train_data, train_df["id"], device, thresholds=optimal_thresholds)
    train_out = "predictions/bart/etpc-paraphrase-detection-train-output.csv"
    train_results.to_csv(train_out, index=False)
    print(f"Wrote {len(train_results)} rows to {train_out}")


if __name__ == "__main__":
    main()
