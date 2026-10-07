import argparse
import csv
import datetime
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration, get_cosine_schedule_with_warmup

try:
    from sacrebleu.metrics import BLEU
except ImportError:
    BLEU = None

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


def tokenize_words(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def get_ngrams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def lcs_length(tokens1, tokens2):
    m, n = len(tokens1), len(tokens2)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_rouge_scores(predictions, references):
    r1_f1_list, r1_p_list, r1_r_list = [], [], []
    r2_f1_list, r2_p_list, r2_r_list = [], [], []
    rl_f1_list, rl_p_list, rl_r_list = [], [], []

    for pred, ref in zip(predictions, references):
        pred_toks = tokenize_words(pred)
        ref_toks = tokenize_words(ref)

        len_p = len(pred_toks)
        len_r = len(ref_toks)

        if len_p == 0 or len_r == 0:
            r1_f1_list.append(0.0)
            r1_p_list.append(0.0)
            r1_r_list.append(0.0)
        else:
            p_counts = Counter(pred_toks)
            r_counts = Counter(ref_toks)
            overlap = sum(min(p_counts[w], r_counts[w]) for w in p_counts if w in r_counts)
            p = overlap / len_p
            r = overlap / len_r
            f1 = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0
            r1_f1_list.append(f1)
            r1_p_list.append(p)
            r1_r_list.append(r)

        p_ngrams = get_ngrams(pred_toks, 2)
        r_ngrams = get_ngrams(ref_toks, 2)
        if len(p_ngrams) == 0 or len(r_ngrams) == 0:
            r2_f1_list.append(0.0)
            r2_p_list.append(0.0)
            r2_r_list.append(0.0)
        else:
            p_ng_counts = Counter(p_ngrams)
            r_ng_counts = Counter(r_ngrams)
            overlap2 = sum(min(p_ng_counts[ng], r_ng_counts[ng]) for ng in p_ng_counts if ng in r_ng_counts)
            p2 = overlap2 / len(p_ngrams)
            r2 = overlap2 / len(r_ngrams)
            f2 = (2.0 * p2 * r2) / (p2 + r2) if (p2 + r2) > 0 else 0.0
            r2_f1_list.append(f2)
            r2_p_list.append(p2)
            r2_r_list.append(r2)

        if len_p == 0 or len_r == 0:
            rl_f1_list.append(0.0)
            rl_p_list.append(0.0)
            rl_r_list.append(0.0)
        else:
            lcs = lcs_length(ref_toks, pred_toks)
            pl = lcs / len_p
            rl = lcs / len_r
            fl = (2.0 * pl * rl) / (pl + rl) if (pl + rl) > 0 else 0.0
            rl_f1_list.append(fl)
            rl_p_list.append(pl)
            rl_r_list.append(rl)

    return {
        "rouge_1_f1": float(np.mean(r1_f1_list)),
        "rouge_1_p": float(np.mean(r1_p_list)),
        "rouge_1_r": float(np.mean(r1_r_list)),
        "rouge_2_f1": float(np.mean(r2_f1_list)),
        "rouge_2_p": float(np.mean(r2_p_list)),
        "rouge_2_r": float(np.mean(r2_r_list)),
        "rouge_l_f1": float(np.mean(rl_f1_list)),
        "rouge_l_p": float(np.mean(rl_p_list)),
        "rouge_l_r": float(np.mean(rl_r_list)),
    }


def compute_bleu_score(predictions, references):
    if BLEU is not None:
        bleu = BLEU()
        return float(bleu.corpus_score(predictions, [references]).score)
    scores = []
    for pred, ref in zip(predictions, references):
        p_toks = tokenize_words(pred)
        r_toks = tokenize_words(ref)
        if len(p_toks) == 0 or len(r_toks) == 0:
            scores.append(0.0)
            continue
        p_counts = Counter(p_toks)
        r_counts = Counter(r_toks)
        overlap = sum(min(p_counts[w], r_counts[w]) for w in p_counts if w in r_counts)
        precision = overlap / max(1, len(p_toks))
        bp = min(1.0, np.exp(1.0 - len(r_toks) / max(1, len(p_toks))))
        scores.append(bp * precision * 100.0)
    return float(np.mean(scores))


def clean_type_ids(raw_types):
    if pd.isna(raw_types):
        return "none"
    if isinstance(raw_types, str):
        try:
            val = eval(raw_types)
            if isinstance(val, (list, tuple, set)):
                cleaned = [str(x).strip() for x in val if str(x).strip()]
                return ", ".join(sorted(set(cleaned), key=lambda x: int(x) if x.isdigit() else x))
            return str(val)
        except Exception:
            clean_str = raw_types.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
            return clean_str if clean_str else "none"
    elif isinstance(raw_types, (list, tuple, set)):
        cleaned = [str(x).strip() for x in raw_types if str(x).strip()]
        return ", ".join(sorted(set(cleaned), key=lambda x: int(x) if x.isdigit() else x))
    return str(raw_types)


def transform_data(dataset, tokenizer, max_length=128, shuffle=True, batch_size=8):
    input_texts = []
    target_texts = []

    for _, row in dataset.iterrows():
        s1 = str(row["sentence1"]).strip()
        type_ids_str = clean_type_ids(row.get("paraphrase_type_ids", ""))

        # Structured linguistic prefix conditioning
        input_text = f"paraphrase types: {type_ids_str} | source: {s1}"
        input_texts.append(input_text)

        if "sentence2" in dataset.columns and pd.notna(row["sentence2"]):
            target_texts.append(str(row["sentence2"]).strip())

    encodings = tokenizer(
        input_texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    if target_texts and len(target_texts) == len(input_texts):
        target_encodings = tokenizer(
            target_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encodings["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
    else:
        labels = torch.zeros_like(input_ids)

    tensor_dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def compute_generation_metrics(predictions, dev_df):
    inputs = dev_df["sentence1"].astype(str).tolist()
    references = dev_df["sentence2"].astype(str).tolist()

    # 1. BLEU metrics
    ref_bleu = compute_bleu_score(predictions, references)
    input_bleu = compute_bleu_score(predictions, inputs)
    input_diff = max(0.0, 100.0 - input_bleu)

    # Penalized BLEU formula
    penalized_bleu = (ref_bleu * input_diff) / 52.0

    # 2. ROUGE metrics
    rouge_metrics = compute_rouge_scores(predictions, references)

    # 3. Length ratio & copy diagnostics
    pred_lens = [len(tokenize_words(p)) for p in predictions]
    ref_lens = [len(tokenize_words(r)) for r in references]
    len_ratio = float(np.mean(pred_lens) / max(1.0, np.mean(ref_lens)))

    exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
    exact_match_rate = float(exact_matches / max(1, len(predictions)))

    copies = sum(1 for p, inp in zip(predictions, inputs) if p.strip().lower() == inp.strip().lower())
    copy_rate = float(copies / max(1, len(predictions)))

    metrics = {
        "penalized_bleu": penalized_bleu,
        "ref_bleu": ref_bleu,
        "input_bleu": input_bleu,
        "input_novelty": input_diff,
        "length_ratio": len_ratio,
        "exact_match_rate": exact_match_rate,
        "copy_rate": copy_rate,
    }
    metrics.update(rouge_metrics)
    return metrics


def log_epoch_metrics(history_csv, epoch, train_loss, dev_loss, metrics, lr, batch_size, seed):
    file_exists = os.path.exists(history_csv)
    fieldnames = [
        "epoch",
        "train_loss",
        "dev_loss",
        "penalized_bleu",
        "ref_bleu",
        "input_bleu",
        "input_novelty",
        "rouge_1_f1",
        "rouge_2_f1",
        "rouge_l_f1",
        "length_ratio",
        "exact_match_rate",
        "copy_rate",
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
            "penalized_bleu": f"{metrics['penalized_bleu']:.6f}",
            "ref_bleu": f"{metrics['ref_bleu']:.6f}",
            "input_bleu": f"{metrics['input_bleu']:.6f}",
            "input_novelty": f"{metrics['input_novelty']:.6f}",
            "rouge_1_f1": f"{metrics['rouge_1_f1']:.6f}",
            "rouge_2_f1": f"{metrics['rouge_2_f1']:.6f}",
            "rouge_l_f1": f"{metrics['rouge_l_f1']:.6f}",
            "length_ratio": f"{metrics['length_ratio']:.6f}",
            "exact_match_rate": f"{metrics['exact_match_rate']:.6f}",
            "copy_rate": f"{metrics['copy_rate']:.6f}",
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
        })


def evaluate_model(model, dev_df, device, tokenizer):
    model.eval()
    predictions = []
    total_loss = 0.0
    num_batches = 0

    dataloader = transform_data(dev_df, tokenizer, max_length=128, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs_loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs_loss.loss.item()
            num_batches += 1

            # Anti-copy Diverse Beam Search decoding
            gen_outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=5,
                num_beam_groups=5,
                diversity_penalty=0.5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                length_penalty=1.0,
                early_stopping=True,
            )
            pred_text = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions.extend(pred_text)

    avg_dev_loss = total_loss / max(1, num_batches)
    metrics = compute_generation_metrics(predictions, dev_df)
    return avg_dev_loss, metrics, predictions


def train_model(model, train_data, dev_df, device, tokenizer, lr=1e-5, epochs=10, batch_size=8, seed=11711, history_csv="runs/ptg_history.csv"):
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    best_checkpoint_path = "models/bart_generation_best.pt"

    if os.path.exists(history_csv):
        try:
            os.remove(history_csv)
        except OSError:
            pass

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
            "lr": lr,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    total_steps = len(train_data) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_penalized_bleu = 0.0
    best_epoch = 1

    print("\n" + "=" * 90)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Dev Loss':^9} | {'Ref BLEU':^9} | {'Novelty':^8} | {'Penalized':^10} | {'ROUGE-1':^8} | {'ROUGE-L':^8}")
    print("=" * 90)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}", disable=TQDM_DISABLE)
        for batch in progress_bar:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / max(1, num_batches)

        # Evaluate validation set with full metrics
        avg_dev_loss, dev_metrics, _ = evaluate_model(model, dev_df, device, tokenizer)

        p_bleu = dev_metrics["penalized_bleu"]
        r_bleu = dev_metrics["ref_bleu"]
        nov = dev_metrics["input_novelty"]
        r1 = dev_metrics["rouge_1_f1"]
        rl = dev_metrics["rouge_l_f1"]

        print(f"{epoch+1:^7d} | {avg_train_loss:^10.4f} | {avg_dev_loss:^9.4f} | {r_bleu:^9.2f} | {nov:^7.1f}% | {p_bleu:^10.4f} | {r1:^8.4f} | {rl:^8.4f}")

        # Log epoch metrics to CSV
        log_epoch_metrics(history_csv, epoch + 1, avg_train_loss, avg_dev_loss, dev_metrics, lr, batch_size, seed)

        if p_bleu > best_penalized_bleu:
            best_penalized_bleu = p_bleu
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_checkpoint_path)

    print("=" * 90)
    print(f"\n[PTG Training Complete] Best Checkpoint Saved at Epoch {best_epoch} (Penalized BLEU = {best_penalized_bleu:.4f})")
    print(f"Metrics history logged to: {history_csv}")

    # Reload best model
    if os.path.exists(best_checkpoint_path):
        print(f"Reloading best checkpoint from {best_checkpoint_path}...")
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    return model


def test_model(test_data, test_ids, device, model, tokenizer):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Generating test paraphrases", disable=TQDM_DISABLE):
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=5,
                num_beam_groups=5,
                diversity_penalty=0.5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                length_penalty=1.0,
                early_stopping=True,
            )
            pred_text = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions.extend(pred_text)

    results = pd.DataFrame({
        "id": test_ids.values,
        "Generated_sentence2": predictions,
    })
    return results


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
    parser.add_argument("--history_out", type=str, default="runs/ptg_history.csv", help="Path to output CSV history")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and evaluate saved checkpoint")
    args = parser.parse_args()
    return args


def finetune_paraphrase_generation(args):
    print("=" * 75)
    print(" Part 2: Paraphrase Type Generation (PTG) — Multi-Metric Evaluation")
    print("=" * 75)
    print(f"Config: Epochs={args.epochs}, LR={args.lr}, Batch Size={args.batch_size}, Seed={args.seed}")

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv")

    # 80/20 train/dev split
    train_df, dev_df = safe_train_test_split(train_dataset, test_size=0.2, random_state=args.seed)
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)

    print(f"Loaded {len(train_df)} training samples and {len(dev_df)} validation samples.")

    train_data = transform_data(train_df, tokenizer, max_length=128, shuffle=True, batch_size=args.batch_size)
    test_data = transform_data(test_dataset, tokenizer, max_length=128, shuffle=False, batch_size=args.batch_size)

    if not args.eval_only:
        model = train_model(
            model,
            train_data,
            dev_df,
            device,
            tokenizer,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            history_csv=args.history_out,
        )
        print("\nTraining completed.")
    else:
        best_checkpoint_path = "models/bart_generation_best.pt"
        if os.path.exists(best_checkpoint_path):
            print(f"\n[Eval Only] Loading checkpoint from {best_checkpoint_path}...")
            model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        else:
            print("Warning: models/bart_generation_best.pt not found. Using pre-trained weights.")

    # Final validation evaluation
    print("\nRunning full validation evaluation with Diverse Beam Search...")
    _, final_metrics, _ = evaluate_model(model, dev_df, device, tokenizer)

    print("\n" + "=" * 65)
    print(" Final Validation Multi-Metric Report:")
    print(f"  -> Reference BLEU (Fidelity):        {final_metrics['ref_bleu']:.2f}  (Target: 47.50)")
    print(f"  -> Penalized BLEU (Novelty-Adjusted): {final_metrics['penalized_bleu']:.4f}")
    print(f"  -> Input Novelty (100 - Copy BLEU):  {final_metrics['input_novelty']:.2f}%")
    print(f"  -> ROUGE-1 F1 / Precision / Recall:  {final_metrics['rouge_1_f1']:.4f} / {final_metrics['rouge_1_p']:.4f} / {final_metrics['rouge_1_r']:.4f}")
    print(f"  -> ROUGE-2 F1 / Precision / Recall:  {final_metrics['rouge_2_f1']:.4f} / {final_metrics['rouge_2_p']:.4f} / {final_metrics['rouge_2_r']:.4f}")
    print(f"  -> ROUGE-L F1 / Precision / Recall:  {final_metrics['rouge_l_f1']:.4f} / {final_metrics['rouge_l_p']:.4f} / {final_metrics['rouge_l_r']:.4f}")
    print(f"  -> Generation Length Ratio:          {final_metrics['length_ratio']:.3f}")
    print(f"  -> Verbatim Input Copy Rate:         {final_metrics['copy_rate']*100:.2f}%")
    print("=" * 65)

    # Generate test predictions
    os.makedirs("predictions/bart", exist_ok=True)
    test_ids = test_dataset["id"]
    test_results = test_model(test_data, test_ids, device, model, tokenizer)

    output_path = "predictions/bart/etpc-paraphrase-generation-test-output.csv"
    test_results.to_csv(output_path, index=False)
    print(f"\nTest predictions successfully saved to: {output_path}")

    # Display sample predictions
    print("\nSample Generated Paraphrases:")
    for i in range(min(3, len(test_results))):
        print(f"[{i+1}] ID:     {test_results['id'].iloc[i]}")
        print(f"    INPUT:  {test_dataset['sentence1'].iloc[i]}")
        print(f"    OUTPUT: {test_results['Generated_sentence2'].iloc[i]}")
        print("-" * 55)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)