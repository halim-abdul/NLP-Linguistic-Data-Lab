#!/usr/bin/env python3

"""
Generate predictions on the TRAIN split for a saved checkpoint.

The provided pipeline (multitask_classifier.py --option test) only writes
dev + test predictions. The Part-2 repository instructions also ask for
predictions on the train split, so this script fills that gap by reusing
the exact same model/dataset classes.

Usage:
    python generate_train_predictions.py --task sst --filepath models/<checkpoint>.pt --use_gpu --local_files_only
    python generate_train_predictions.py --task sts --filepath models/<checkpoint>.pt --use_gpu --local_files_only
    python generate_train_predictions.py --task qqp --filepath models/<checkpoint>.pt --use_gpu --local_files_only
"""

import argparse
import csv

import torch
from torch.utils.data import DataLoader

from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from multitask_classifier import (
    MultitaskBERT,
    evaluate_sst,
    get_device,
    safe_torch_load,
)

OUT_PATHS = {
    "sst": ("predictions/bert/sst-sentiment-train-output.csv", "Predicted_Sentiment"),
    "sts": ("predictions/bert/sts-similarity-train-output.csv", "Predicted_Similarity"),
    "qqp": ("predictions/bert/quora-paraphrase-train-output.csv", "Predicted_Is_Paraphrase"),
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["sst", "sts", "qqp"])
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sst_train", type=str, default="data/sst-sentiment-train.csv")
    parser.add_argument("--quora_train", type=str, default="data/quora-paraphrase-train.csv")
    parser.add_argument("--sts_train", type=str, default="data/sts-similarity-train.csv")
    parser.add_argument("--etpc_train", type=str, default="data/etpc-paraphrase-train-split.csv")
    return parser.parse_args()


def main():
    args = get_args()
    device = get_device(args.use_gpu)
    print(f"Loading checkpoint {args.filepath} on {device} ...")

    saved = safe_torch_load(args.filepath, device)
    config = saved["model_config"]

    model = MultitaskBERT(config)
    missing, unexpected = model.load_state_dict(saved["model"], strict=False)
    if missing:
        print(f"  (non-strict load) missing keys ignored: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint not present in model: {unexpected}")
    model = model.to(device)
    model.eval()

    sst_train, _, quora_train, sts_train, _ = load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, args.etpc_train, split="train"
    )

    out_path, out_col = OUT_PATHS[args.task]

    if args.task == "sst":
        dataset = SentenceClassificationDataset(sst_train, args)
        loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
        acc, preds, ids, metrics = evaluate_sst(model, loader, device, has_labels=True)
        print(f"Train accuracy (fit quality, NOT a generalization metric): {acc:.4f}")
        rows = list(zip(ids, (int(p) for p in preds)))

    elif args.task == "qqp":
        from tqdm import tqdm
        dataset = SentencePairDataset(quora_train, args)
        loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
        ids, preds, labels = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="eval-qqp-train"):
                b_ids1 = batch["token_ids_1"].to(device)
                b_mask1 = batch["attention_mask_1"].to(device)
                b_ids2 = batch["token_ids_2"].to(device)
                b_mask2 = batch["attention_mask_2"].to(device)
                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                p = logits.sigmoid().round().flatten().cpu().numpy().astype(int)
                ids.extend(batch["sent_ids"])
                preds.extend(p.tolist())
                labels.extend(batch["labels"].flatten().cpu().numpy().tolist())
        acc = sum(int(p == l) for p, l in zip(preds, labels)) / len(labels)
        print(f"Train accuracy (fit quality, NOT a generalization metric): {acc:.4f}")
        rows = list(zip(ids, preds))

    elif args.task == "sts":
        from tqdm import tqdm
        dataset = SentencePairDataset(sts_train, args, isRegression=True)
        loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
        ids, preds, labels = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="eval-sts-train"):
                b_ids1 = batch["token_ids_1"].to(device)
                b_mask1 = batch["attention_mask_1"].to(device)
                b_ids2 = batch["token_ids_2"].to(device)
                b_mask2 = batch["attention_mask_2"].to(device)
                logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                p = logits.flatten().cpu().numpy()
                ids.extend(batch["sent_ids"])
                preds.extend(p.tolist())
                labels.extend(batch["labels"].flatten().cpu().numpy().tolist())
        import numpy as np
        corr = float(np.corrcoef(preds, labels)[0, 1])
        print(f"Train correlation (fit quality, NOT a generalization metric): {corr:.4f}")
        rows = list(zip(ids, preds))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", out_col])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
