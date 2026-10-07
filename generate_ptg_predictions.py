#!/usr/bin/env python3

"""
Generate dev + train predictions for Paraphrase Type Generation (PTG) using
the existing best checkpoint (no retraining needed).

Usage:
    python generate_ptg_predictions.py --use_gpu
"""

import argparse
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from bart_generation import safe_train_test_split, test_model, transform_data

BEST_CHECKPOINT = "models/bart_generation_best.pt"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    if not os.path.exists(BEST_CHECKPOINT):
        raise FileNotFoundError(f"Missing checkpoint: {BEST_CHECKPOINT}")

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True).to(device)
    model.load_state_dict(torch.load(BEST_CHECKPOINT, map_location=device))
    model.eval()

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    train_df, dev_df = safe_train_test_split(train_dataset, test_size=0.2, random_state=args.seed)
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    print(f"train={len(train_df)} dev={len(dev_df)} (same 80/20 split used during training)")

    train_data = transform_data(train_df, tokenizer, max_length=128, shuffle=False, batch_size=args.batch_size)
    dev_data = transform_data(dev_df, tokenizer, max_length=128, shuffle=False, batch_size=args.batch_size)

    os.makedirs("predictions/bart", exist_ok=True)

    print("Generating dev predictions...")
    dev_results = test_model(dev_data, dev_df["id"], device, model, tokenizer)
    dev_out = "predictions/bart/etpc-paraphrase-generation-dev-output.csv"
    dev_results.to_csv(dev_out, index=False)
    print(f"Wrote {len(dev_results)} rows to {dev_out}")

    print("Generating train predictions...")
    train_results = test_model(train_data, train_df["id"], device, model, tokenizer)
    train_out = "predictions/bart/etpc-paraphrase-generation-train-output.csv"
    train_results.to_csv(train_out, index=False)
    print(f"Wrote {len(train_results)} rows to {train_out}")


if __name__ == "__main__":
    main()
