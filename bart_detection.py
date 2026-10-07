import argparse
import csv
import datetime
import difflib
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel, get_cosine_schedule_with_warmup

try:
    from sklearn.metrics import matthews_corrcoef
except ImportError:
    matthews_corrcoef = None

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None


TQDM_DISABLE = False


def safe_train_test_split(df, test_size=0.2, random_state=11711):
    if train_test_split is not None:
        return train_test_split(df, test_size=test_size, random_state=random_state)
    rng = np.random.RandomState(random_state)
    shuffled_indices = rng.permutation(len(df))
    split_point = int(len(df) * (1.0 - test_size))
    train_idx = shuffled_indices[:split_point]
    dev_idx = shuffled_indices[split_point:]
    return df.iloc[train_idx].copy(), df.iloc[dev_idx].copy()


def extract_linguistic_features(sentence1_list, sentence2_list):
    features = []
    for s1, s2 in zip(sentence1_list, sentence2_list):
        s1 = str(s1).lower().strip()
        s2 = str(s2).lower().strip()

        words1 = set(s1.split())
        words2 = set(s2.split())

        # 1. Jaccard word overlap
        union = words1.union(words2)
        intersection = words1.intersection(words2)
        jaccard = len(intersection) / max(1, len(union))

        # 2. Character edit similarity
        edit_sim = difflib.SequenceMatcher(None, s1, s2).ratio()

        # 3. Length ratio
        l1, l2 = len(s1), len(s2)
        len_ratio = min(l1, l2) / max(1, max(l1, l2))

        # 4. Token count difference
        n1, n2 = len(words1), len(words2)
        count_diff = abs(n1 - n2) / max(1, n1 + n2)

        features.append([jaccard, edit_sim, len_ratio, count_diff])

    return torch.tensor(features, dtype=torch.float32)


class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma=1.0):
        super(FocalLossWithLogits, self).__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * bce
        return loss.mean()


class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=26, dropout_p=0.3):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large", local_files_only=True)
        hidden_size = self.bart.config.hidden_size  # 1024

        # Multi-scale (3072) + Linguistic features (4) = 3076
        feature_dim = (hidden_size * 3) + 4

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask=None, ling_features=None):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, 1024]

        # 1. CLS token representation
        cls_output = last_hidden_state[:, 0, :]

        # 2. Mask-aware Mean-Pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_output = sum_embeddings / sum_mask

            # 3. Mask-aware Max-Pooling
            hidden_masked = last_hidden_state.clone()
            hidden_masked[mask_expanded == 0] = -1e9
            max_output = torch.max(hidden_masked, dim=1)[0]
        else:
            mean_output = torch.mean(last_hidden_state, dim=1)
            max_output = torch.max(last_hidden_state, dim=1)[0]

        # Combine neural multi-scale features with explicit linguistic features
        if ling_features is not None:
            combined_features = torch.cat([cls_output, mean_output, max_output, ling_features], dim=-1)
        else:
            dummy_ling = torch.zeros(cls_output.size(0), 4, device=cls_output.device)
            combined_features = torch.cat([cls_output, mean_output, max_output, dummy_ling], dim=-1)

        logits = self.classifier(combined_features)
        return logits


def transform_data(dataset, max_length=512, is_training=False, batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)
    sentence1 = dataset["sentence1"].astype(str).tolist()
    sentence2 = dataset["sentence2"].astype(str).tolist()

    encodings = tokenizer(
        sentence1,
        sentence2,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # Compute linguistic features
    ling_features = extract_linguistic_features(sentence1, sentence2)

    if "paraphrase_type_ids" in dataset.columns:
        unused_ids = [12, 19, 20, 23, 27]
        target_ids = list(set(range(1, 32)) - set(unused_ids))
        target_ids.sort()

        binary_labels = []
        for val in dataset["paraphrase_type_ids"]:
            if isinstance(val, str):
                try:
                    label_list = eval(val)
                except Exception:
                    label_list = []
            elif isinstance(val, list):
                label_list = val
            else:
                label_list = []

            label_set = set(label_list)
            binary_label = [int(i in label_set) for i in target_ids]
            binary_labels.append(binary_label)

        labels_tensor = torch.FloatTensor(binary_labels)
        tensor_dataset = TensorDataset(input_ids, attention_mask, ling_features, labels_tensor)
        dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=is_training)
    else:
        tensor_dataset = TensorDataset(input_ids, attention_mask, ling_features)
        dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

    return dataloader


def compute_mcc(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    numerator = float(tp * tn - fp * fn)
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator <= 0.0:
        return 0.0
    return numerator / np.sqrt(denominator)


def compute_ptd_metrics(true_labels_np, probs_np, thresholds=None):
    if thresholds is None:
        preds = (probs_np >= 0.5).astype(int)
    else:
        thresholds = np.asarray(thresholds)
        preds = (probs_np >= thresholds).astype(int)

    num_samples, num_labels = true_labels_np.shape

    # 1. Exact Match (Subset Accuracy)
    exact_match = float(np.mean(np.all(preds == true_labels_np, axis=1)))

    # 2. Mean Accuracy (Hamming Accuracy)
    mean_accuracy = float(np.mean(preds == true_labels_np))

    # 3. Hamming Loss
    hamming_loss = float(np.mean(preds != true_labels_np))

    # Per-label metrics
    f1_list = []
    prec_list = []
    rec_list = []
    mcc_list = []
    support_list = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for j in range(num_labels):
        yt = true_labels_np[:, j]
        yp = preds[:, j]

        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        support = np.sum(yt == 1)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / max(1e-12, prec + rec) if (prec + rec) > 0 else 0.0
        mcc = compute_mcc(yt, yp)

        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)
        mcc_list.append(mcc)
        support_list.append(support)

    macro_f1 = float(np.mean(f1_list))
    macro_prec = float(np.mean(prec_list))
    macro_rec = float(np.mean(rec_list))
    macro_mcc = float(np.mean(mcc_list))

    # Micro F1
    micro_prec = total_tp / max(1, total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / max(1, total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2.0 * micro_prec * micro_rec) / max(1e-12, micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    # Weighted F1
    total_support = max(1, sum(support_list))
    weighted_f1 = float(sum(f1 * sup for f1, sup in zip(f1_list, support_list)) / total_support)

    return {
        "mean_accuracy": mean_accuracy,
        "exact_match": exact_match,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "macro_mcc": macro_mcc,
        "hamming_loss": hamming_loss,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
    }


def find_optimal_thresholds(y_true, y_probs):
    num_classes = y_true.shape[1]
    optimal_thresholds = np.full(num_classes, 0.5)

    for label_idx in range(num_classes):
        true_col = y_true[:, label_idx]
        probs_col = y_probs[:, label_idx]

        best_thresh = 0.5
        best_acc = np.mean((probs_col >= 0.5).astype(int) == true_col)
        best_mcc = -1.0

        for thresh in np.linspace(0.35, 0.65, 31):
            preds = (probs_col >= thresh).astype(int)
            acc = np.mean(preds == true_col)
            mcc = compute_mcc(true_col, preds)

            if acc > best_acc or (acc == best_acc and mcc > best_mcc):
                best_acc = acc
                best_mcc = mcc
                best_thresh = thresh

        optimal_thresholds[label_idx] = best_thresh

    return optimal_thresholds


def log_epoch_metrics(history_csv, epoch, train_loss, dev_loss, metrics, lr, batch_size, seed):
    file_exists = os.path.exists(history_csv)
    fieldnames = [
        "epoch",
        "train_loss",
        "dev_loss",
        "mean_accuracy",
        "exact_match",
        "macro_f1",
        "micro_f1",
        "weighted_f1",
        "macro_mcc",
        "hamming_loss",
        "macro_precision",
        "macro_recall",
        "lr",
        "batch_size",
        "seed",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(history_csv)), exist_ok=True)
    with open(history_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "dev_loss": f"{dev_loss:.6f}",
            "mean_accuracy": f"{metrics['mean_accuracy']:.6f}",
            "exact_match": f"{metrics['exact_match']:.6f}",
            "macro_f1": f"{metrics['macro_f1']:.6f}",
            "micro_f1": f"{metrics['micro_f1']:.6f}",
            "weighted_f1": f"{metrics['weighted_f1']:.6f}",
            "macro_mcc": f"{metrics['macro_mcc']:.6f}",
            "hamming_loss": f"{metrics['hamming_loss']:.6f}",
            "macro_precision": f"{metrics['macro_precision']:.6f}",
            "macro_recall": f"{metrics['macro_recall']:.6f}",
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
        })


def evaluate_model(model, dataloader, device, loss_fn=None, thresholds=None):
    all_probs = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval-ptd", leave=False):
            input_ids, attention_mask, ling_feats, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            ling_feats = ling_feats.to(device)
            labels = labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, ling_features=ling_feats)
            if loss_fn is not None:
                loss = loss_fn(logits, labels)
                total_loss += loss.item()
                num_batches += 1

            probs = torch.sigmoid(logits)
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs_np = torch.cat(all_probs, dim=0).cpu().numpy()
    true_labels_np = torch.cat(all_labels, dim=0).cpu().numpy()
    avg_dev_loss = (total_loss / max(1, num_batches)) if loss_fn is not None else 0.0

    metrics = compute_ptd_metrics(true_labels_np, all_probs_np, thresholds=thresholds)
    return avg_dev_loss, metrics, all_probs_np, true_labels_np


def train_model(model, train_data, dev_data, device, lr=1e-5, epochs=10, batch_size=8, seed=11711, history_csv="runs/ptd_history.csv"):
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    if os.path.exists(history_csv):
        try:
            os.remove(history_csv)
        except OSError:
            pass

    # Focal loss for hard borderline cases
    loss_fn = FocalLossWithLogits(gamma=1.0)

    # Differential learning rates
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    head_params = list(model.classifier.parameters())
    head_param_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for p in backbone_params
                if not any(nd in n for n, p_named in model.named_parameters() if id(p_named) == id(p) for nd in no_decay)
            ],
            "weight_decay": 0.01,
            "lr": lr,
        },
        {
            "params": [
                p for p in backbone_params
                if any(nd in n for n, p_named in model.named_parameters() if id(p_named) == id(p) for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": head_params,
            "weight_decay": 0.01,
            "lr": lr * 3.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    total_steps = len(train_data) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    top_checkpoints = []

    print("\n" + "=" * 80)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Dev Loss':^10} | {'Mean Acc':^10} | {'Exact Match':^11} | {'Macro F1':^10} | {'MCC':^8}")
    print("=" * 80)

    best_acc = 0.0
    best_epoch = 1

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}", disable=TQDM_DISABLE)
        for batch in progress_bar:
            input_ids, attention_mask, ling_feats, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            ling_feats = ling_feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask, ling_features=ling_feats)
            loss = loss_fn(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / max(1, num_batches)
        avg_dev_loss, dev_metrics, dev_probs, dev_labels = evaluate_model(model, dev_data, device, loss_fn=loss_fn)

        acc = dev_metrics["mean_accuracy"]
        em = dev_metrics["exact_match"]
        f1 = dev_metrics["macro_f1"]
        mcc = dev_metrics["macro_mcc"]

        print(f"{epoch+1:^7d} | {avg_train_loss:^10.4f} | {avg_dev_loss:^10.4f} | {acc*100:^9.2f}% | {em*100:^10.2f}% | {f1:^10.4f} | {mcc:^8.4f}")

        # Log epoch metrics to CSV
        log_epoch_metrics(history_csv, epoch + 1, avg_train_loss, avg_dev_loss, dev_metrics, lr, batch_size, seed)

        # Save checkpoint for this epoch
        epoch_path = f"models/bart_ptd_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), epoch_path)

        score = acc + (0.01 * mcc)
        top_checkpoints.append((score, acc, mcc, em, f1, epoch + 1, epoch_path))
        top_checkpoints.sort(key=lambda x: x[0], reverse=True)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1

    print("=" * 80)
    print(f"\n[PTD Training Complete] Best Single Checkpoint Peaked at Epoch {best_epoch} with Dev Acc = {best_acc*100:.2f}%")
    print(f"Metrics history logged to: {history_csv}")

    best_3 = top_checkpoints[:3]
    print("\n" + "-" * 65)
    print(" Top 3 Checkpoints for Model Ensembling:")
    for rank, (score, acc, mcc, em, f1, ep, path) in enumerate(best_3, 1):
        print(f"  Rank {rank}: Epoch {ep:02d} (Mean Acc = {acc*100:.2f}%, Exact Match = {em*100:.2f}%, Macro F1 = {f1:.4f}, MCC = {mcc:.4f}) -> {path}")
    print("-" * 65)

    return best_3


def predict_ensemble(model_paths, dataloader, device):

    ensemble_probs = None
    all_labels = []
    labels_collected = False

    for path in model_paths:
        if not os.path.exists(path):
            continue
        model = BartWithClassifier().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        probs_list = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"eval-ptd-{os.path.basename(path)}"):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                ling_feats = batch[2].to(device)

                if len(batch) >= 4 and not labels_collected:
                    all_labels.append(batch[3])

                logits = model(input_ids=input_ids, attention_mask=attention_mask, ling_features=ling_feats)
                probs = torch.sigmoid(logits)
                probs_list.append(probs)

        labels_collected = True
        model_probs = torch.cat(probs_list, dim=0).cpu().numpy()
        if ensemble_probs is None:
            ensemble_probs = model_probs
        else:
            ensemble_probs += model_probs

    ensemble_probs /= max(1, len(model_paths))
    true_labels_np = torch.cat(all_labels, dim=0).cpu().numpy() if all_labels else None
    return ensemble_probs, true_labels_np


def test_ensemble(model_paths, test_data, test_ids, device, thresholds=None):
    ensemble_probs, _ = predict_ensemble(model_paths, test_data, device)

    if thresholds is not None:
        preds = (ensemble_probs >= thresholds).astype(int).tolist()
    else:
        preds = (ensemble_probs >= 0.5).astype(int).tolist()

    df = pd.DataFrame({
        "id": test_ids,
        "Predicted_Paraphrase_Types": preds,
    })
    return df


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--history_out", type=str, default="runs/ptd_history.csv", help="Path to output CSV history")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run ensembling on saved checkpoints")
    args = parser.parse_args()
    return args


def finetune_paraphrase_detection(args):
    print("=" * 75)
    print(" Part 2: Paraphrase Type Detection (PTD) — Multi-Metric Evaluation")
    print("=" * 75)
    print(f"Config: Epochs={args.epochs}, LR={args.lr}, Batch Size={args.batch_size}, Seed={args.seed}, Eval Only={args.eval_only}")

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartWithClassifier().to(device)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")

    # 80/20 train/dev split
    train_df, dev_df = safe_train_test_split(train_dataset, test_size=0.2, random_state=args.seed)

    print("Preprocessing data and extracting linguistic features...")
    train_data = transform_data(train_df, is_training=True, batch_size=args.batch_size)
    dev_data = transform_data(dev_df, is_training=False, batch_size=args.batch_size)
    test_data = transform_data(test_dataset, is_training=False, batch_size=args.batch_size)

    print(f"Loaded {len(train_df)} training samples and {len(dev_df)} validation samples.")

    if args.eval_only:
        print("\n[Fast Mode] Skipping training. Loading saved top-3 checkpoints:")
        best_3_paths = ["models/bart_ptd_epoch6.pt", "models/bart_ptd_epoch10.pt", "models/bart_ptd_epoch7.pt"]
        for p in best_3_paths:
            print(f"  -> {p}")
    else:
        best_3_info = train_model(
            model,
            train_data,
            dev_data,
            device,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            history_csv=args.history_out,
        )
        best_3_paths = [info[6] for info in best_3_info]

    print("\nRunning Top-3 Checkpoint Ensembling on validation set...")
    dev_ens_probs, dev_labels = predict_ensemble(best_3_paths, dev_data, device)

    # 1. Ensemble with standard 0.5 threshold
    std_metrics = compute_ptd_metrics(dev_labels, dev_ens_probs, thresholds=None)
    print("\n" + "=" * 65)
    print(f" Ensemble Results (Standard 0.5 Threshold):")
    print(f"  -> Mean Binary Accuracy:        {std_metrics['mean_accuracy']*100:.2f}%  (Baseline Target: 91.30%)")
    print(f"  -> Exact Match (Subset Acc):    {std_metrics['exact_match']*100:.2f}%")
    print(f"  -> Macro-Averaged F1:           {std_metrics['macro_f1']:.4f}")
    print(f"  -> Micro-Averaged F1:           {std_metrics['micro_f1']:.4f}")
    print(f"  -> Weighted F1:                 {std_metrics['weighted_f1']:.4f}")
    print(f"  -> Matthews Correlation (MCC):  {std_metrics['macro_mcc']:.4f}")
    print(f"  -> Hamming Loss:                {std_metrics['hamming_loss']:.4f}")
    print("=" * 65)

    # 2. Optimal threshold calibration on ensemble probabilities
    print("\nCalibrating per-class optimal thresholds on ensemble predictions...")
    optimal_thresholds = find_optimal_thresholds(dev_labels, dev_ens_probs)

    tuned_metrics = compute_ptd_metrics(dev_labels, dev_ens_probs, thresholds=optimal_thresholds)
    print("\n" + "=" * 65)
    print(f" Ensemble Results (Optimized Per-Class Thresholds):")
    print(f"  -> Mean Binary Accuracy:        {tuned_metrics['mean_accuracy']*100:.2f}%  (Target: 91.30%)")
    print(f"  -> Exact Match (Subset Acc):    {tuned_metrics['exact_match']*100:.2f}%")
    print(f"  -> Macro-Averaged F1:           {tuned_metrics['macro_f1']:.4f}")
    print(f"  -> Micro-Averaged F1:           {tuned_metrics['micro_f1']:.4f}")
    print(f"  -> Weighted F1:                 {tuned_metrics['weighted_f1']:.4f}")
    print(f"  -> Matthews Correlation (MCC):  {tuned_metrics['macro_mcc']:.4f}")
    print(f"  -> Hamming Loss:                {tuned_metrics['hamming_loss']:.4f}")
    print("=" * 65)

    # Save test predictions using the ensembled model with optimal thresholds
    os.makedirs("predictions/bart", exist_ok=True)
    test_ids = test_dataset["id"]
    test_results = test_ensemble(best_3_paths, test_data, test_ids, device, thresholds=optimal_thresholds)
    output_path = "predictions/bart/etpc-paraphrase-detection-test-output.csv"
    test_results.to_csv(output_path, index=False)
    print(f"\nEnsemble test predictions successfully written to: {output_path}")


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
