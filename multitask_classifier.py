from __future__ import annotations

import argparse
import atexit
import csv
import faulthandler
import io
import math
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from pprint import pformat
import random
import re
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Iterable, Optional, Sequence, cast
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BertModel
from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    load_multitask_data,
)
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW


TQDM_DISABLE = False
BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

P1_BASELINE_REFERENCE = {
    "sst": {
        "dev_accuracy": 0.524,
        "architecture": "BERT pooler_output -> Dropout(0.3) -> Linear(768,5)",
    },
    "qqp": {
        "dev_accuracy": 0.844,
        "architecture": (
            "BERT(u), BERT(v) -> [u,v,|u-v|] -> "
            "Dropout(0.3) -> Linear(2304,1)"
        ),
    },
}



LOG_DIR = Path(".runs")


def _ensure_gitignore_contains(path: Path) -> None:
    gitignore = Path(".gitignore")
    try:
        lines = gitignore.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []

    wanted = {f"/{path.as_posix()}", f"/{path.as_posix()}/"}
    if any(line.strip() in wanted for line in lines):
        return

    with gitignore.open("a", encoding="utf-8") as handle:
        if lines and lines[-1] != "":
            handle.write("\n")
        handle.write(f"/{path.as_posix()}/\n")


class Tee(io.TextIOBase):

    def __init__(self, *streams: Any):
        super().__init__()
        self.streams = tuple(streams)

    def write(self, message: str) -> int:
        for stream in self.streams:
            try:
                if not getattr(stream, "closed", False):
                    stream.write(message)
                    stream.flush()
            except (ValueError, OSError):
                pass
        return len(message)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                if not getattr(stream, "closed", False):
                    stream.flush()
            except (ValueError, OSError):
                pass

    def isatty(self) -> bool:
        for stream in self.streams:
            try:
                if stream.isatty():
                    return True
            except Exception:
                pass
        return False


class RunLogHandle:


    def __init__(
        self,
        metrics_path,
        progress_path,
        metrics_file,
        progress_file,
        original_stdout,
        original_stderr,
        original_excepthook,
        original_unraisablehook,
        original_showwarning,
    ):
        self.metrics_path = Path(metrics_path)
        self.progress_path = Path(progress_path)
        self.metrics_file = metrics_file
        self.progress_file = progress_file
        self.original_stdout = original_stdout
        self.original_stderr = original_stderr
        self.original_excepthook = original_excepthook
        self.original_unraisablehook = original_unraisablehook
        self.original_showwarning = original_showwarning
        self.closed = False

    @property
    def paths(self):
        return {
            "metrics": self.metrics_path,
            "progress": self.progress_path,
        }

    def __getitem__(self, key):
        return self.paths[key]

    def __repr__(self):
        return repr(self.paths)

    def close(self):
        if self.closed:
            return
        self.closed = True

        # Restore global streams/hooks FIRST.
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        sys.excepthook = self.original_excepthook
        if self.original_unraisablehook is not None:
            sys.unraisablehook = self.original_unraisablehook
        warnings.showwarning = self.original_showwarning

        try:
            if faulthandler.is_enabled():
                faulthandler.disable()
        except Exception:
            pass

        for handle in (self.metrics_file, self.progress_file):
            try:
                handle.flush()
            except Exception:
                pass
            try:
                handle.close()
            except Exception:
                pass


def start_runlog(prefix: str = "run") -> RunLogHandle:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_gitignore_contains(LOG_DIR)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_path = LOG_DIR / f"{prefix}-metrics-{timestamp}.txt"
    progress_path = LOG_DIR / f"{prefix}-progress-{timestamp}.txt"

    metrics_file = metrics_path.open("a", encoding="utf-8", buffering=1)
    progress_file = progress_path.open("a", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_excepthook = sys.excepthook
    original_unraisablehook = getattr(sys, "unraisablehook", None)
    original_showwarning = warnings.showwarning

    header = (
        "=== RUN START =====================================\n"
        f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"cwd:  {Path.cwd()}\n"
        f"cmd:  {' '.join(sys.argv)}\n"
        "stdout -> metrics log | stderr -> progress log\n"
        "===================================================\n"
    )
    metrics_file.write(header)
    progress_file.write(header)

    handle = RunLogHandle(
        metrics_path,
        progress_path,
        metrics_file,
        progress_file,
        original_stdout,
        original_stderr,
        original_excepthook,
        original_unraisablehook,
        original_showwarning,
    )
    atexit.register(handle.close)

    try:
        faulthandler.enable(file=progress_file, all_threads=True)
    except Exception:
        try:
            if not faulthandler.is_enabled():
                faulthandler.enable(file=original_stderr)
        except Exception:
            pass

    sys.stdout = Tee(original_stdout, metrics_file)
    sys.stderr = Tee(original_stderr, progress_file)

    def _excepthook(exc_type, exc, tb):
        target = sys.stderr if not handle.closed else original_stderr
        try:
            target.write("\n=== UNCAUGHT EXCEPTION ============================\n")
            traceback.print_exception(exc_type, exc, tb, file=target)
            target.write("===================================================\n")
            target.flush()
        except Exception:
            original_excepthook(exc_type, exc, tb)

    sys.excepthook = _excepthook

    def _unraisable_hook(unraisable):
        target = sys.stderr if not handle.closed else original_stderr
        try:
            target.write("\n=== UNRAISABLE EXCEPTION ==========================\n")
            if getattr(unraisable, "err_msg", None):
                target.write(f"{unraisable.err_msg}\n")
            obj = getattr(unraisable, "object", None)
            if obj is not None:
                try:
                    target.write(f"in object: {obj!r}\n")
                except Exception:
                    target.write("in object: <unrepr-able>\n")
            traceback.print_exception(
                unraisable.exc_type,
                unraisable.exc_value,
                unraisable.exc_traceback,
                file=target,
            )
            target.write("===================================================\n")
            target.flush()
        except Exception as logger_error:
            try:
                original_stderr.write(
                    f"\n[runlogger] unraisable hook failed: {logger_error!r}\n"
                )
            except Exception:
                pass

    if hasattr(sys, "unraisablehook"):
        sys.unraisablehook = _unraisable_hook

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        target = sys.stderr if not handle.closed else original_stderr
        try:
            target.write(
                warnings.formatwarning(
                    message,
                    category,
                    filename,
                    lineno,
                    line,
                )
            )
            target.flush()
        except Exception:
            original_showwarning(
                message,
                category,
                filename,
                lineno,
                file=file,
                line=line,
            )

    warnings.showwarning = _showwarning
    warnings.simplefilter("default")
    return handle


def _default_runlog_prefix(args) -> str:
    if args.task == "sst":
        if getattr(args, "ensemble_seeds", "").strip():
            return f"sst-ensemble-{args.sst_profile}"
        return f"sst-{args.sst_profile}-seed{args.seed}"
    if args.task == "qqp":
        return f"qqp-{args.qqp_profile}-seed{args.seed}"
    return f"{args.task}-seed{args.seed}"



def seed_everything(seed: int = 11711) -> None:
    """Fix random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is useful for fair ablations.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(use_gpu: bool) -> torch.device:
    """Return CUDA when requested and available, otherwise CPU."""
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    if use_gpu:
        print(
            "Warning: --use_gpu was provided, but CUDA is unavailable. "
            "Using CPU."
        )

    return torch.device("cpu")


def load_sst_data(filename: str, split: str):

    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unsupported SST split: {split!r}")

    examples = []

    with open(filename, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        required_columns = {"sentence", "id"}
        if split != "test":
            required_columns.add("sentiment")

        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns
        if missing_columns:
            raise ValueError(
                f"Missing columns in {filename}: {sorted(missing_columns)}"
            )

        for record in reader:
            sentence = record["sentence"].lower().strip()
            sentence_id = record["id"].lower().strip()

            if split == "test":
                examples.append((sentence, sentence_id))
                continue

            label = int(record["sentiment"].strip())
            if label not in range(N_SENTIMENT_CLASSES):
                raise ValueError(
                    f"Invalid SST label {label} in {filename}; expected 0..4."
                )
            examples.append((sentence, label, sentence_id))

    if not examples:
        raise RuntimeError(f"No SST examples were loaded from {filename}")

    print(f"Loaded {len(examples)} {split} SST examples from {filename}")
    return examples



def sst_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = N_SENTIMENT_CLASSES,
) -> dict[str, Any]:

    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)

    if true.shape != pred.shape:
        raise ValueError("SST y_true and y_pred must have the same shape.")
    if true.size == 0:
        raise ValueError("Cannot compute SST metrics on an empty set.")

    accuracy = float(np.mean(true == pred))
    mae = float(np.mean(np.abs(true - pred)))

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gold, guess in zip(true, pred):
        if 0 <= gold < num_classes and 0 <= guess < num_classes:
            confusion[gold, guess] += 1

    f1_per_class: list[float] = []
    for cls in range(num_classes):
        tp = int(confusion[cls, cls])
        fp = int(confusion[:, cls].sum() - tp)
        fn = int(confusion[cls, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_per_class.append(float(f1))

    error_mask = true != pred
    adjacent_error_rate = (
        float(np.mean(np.abs(true[error_mask] - pred[error_mask]) == 1))
        if error_mask.any()
        else 0.0
    )

    # Quadratic weighted kappa is useful for SST because labels 0<1<2<3<4
    # are ordered. It is an analysis metric only; official model selection
    # remains based on dev accuracy.
    hist_true = confusion.sum(axis=1).astype(np.float64)
    hist_pred = confusion.sum(axis=0).astype(np.float64)
    n = float(confusion.sum())
    if n > 0:
        observed = confusion.astype(np.float64) / n
        expected = np.outer(hist_true, hist_pred) / (n * n)
        idx = np.arange(num_classes, dtype=np.float64)
        weights = (idx[:, None] - idx[None, :]) ** 2 / max(1.0, (num_classes - 1) ** 2)
        denominator = float((weights * expected).sum())
        qwk = 1.0 - float((weights * observed).sum()) / denominator if denominator > 0 else 0.0
    else:
        qwk = 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_per_class)),
        "f1_per_class": f1_per_class,
        "mae": mae,
        "qwk": float(qwk),
        "adjacent_error_rate": adjacent_error_rate,
        "confusion_matrix": confusion,
    }


class MultiSampleClassificationHead(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        dropout_prob: float,
        num_samples: int = 5,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.projection = nn.Linear(input_size, hidden_size)
        self.activation = nn.GELU()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_prob) for _ in range(max(1, int(num_samples)))]
        )
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, features):
        hidden = self.input_norm(features)
        hidden = self.projection(hidden)
        hidden = self.activation(hidden)
        hidden = self.hidden_norm(hidden)

        logits = [self.output(dropout(hidden)) for dropout in self.dropouts]
        return torch.stack(logits, dim=0).mean(dim=0)


class RawCLSClassificationHead(nn.Module):

    def __init__(self, hidden_size: int, num_classes: int, dropout_prob: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, cls_vector):
        return self.output(self.dropout(self.norm(cls_vector)))


class GatedAttentionSentimentHead(nn.Module):


    def __init__(self, hidden_size: int, num_classes: int, dropout_prob: float):
        super().__init__()
        self.token_score = nn.Linear(hidden_size, 1, bias=False)
        self.gate = nn.Linear(2 * hidden_size, 1)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden, content_mask):
        scores = self.token_score(hidden).squeeze(-1)
        scores = scores.masked_fill(content_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1)
        attentive = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        raw_cls = hidden[:, 0]
        gate = torch.sigmoid(self.gate(torch.cat([raw_cls, attentive], dim=-1)))
        fused = gate * raw_cls + (1.0 - gate) * attentive
        return self.output(self.dropout(self.norm(fused)))


class ModelEMA:


    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.decay <= 0.0:
            return
        self.num_updates += 1

        # Bias-correct the very early EMA so it does not stay too close to the
        # initialization during the first updates.
        dynamic_decay = min(
            self.decay,
            (1.0 + self.num_updates) / (10.0 + self.num_updates),
        )

        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(dynamic_decay).add_(
                parameter.detach(),
                alpha=1.0 - dynamic_decay,
            )

    @contextmanager
    def average_parameters(self, model: nn.Module):
        if self.decay <= 0.0:
            yield
            return

        backup = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name not in self.shadow:
                    continue
                backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])

        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in backup:
                        parameter.copy_(backup[name])


class MultitaskBERT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        bert_output = BertModel.from_pretrained(
            "bert-base-uncased",
            local_files_only=config.local_files_only,
            output_loading_info=False,
        )
        self.bert = (
            bert_output[0] if isinstance(bert_output, tuple) else bert_output
        )

        self.sts_diff_head = nn.Sequential(
        nn.Linear(BERT_HIDDEN_SIZE + 1, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
        )

        for parameter in self.bert.parameters():
            if config.option == "pretrain":
                parameter.requires_grad = False
            elif config.option == "finetune":
                parameter.requires_grad = True
            else:
                raise ValueError(
                    f"Unsupported option {config.option!r}; expected "
                    "'pretrain' or 'finetune'."
                )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        strict_p1 = bool(getattr(config, "strict_p1_baseline", False))

        if strict_p1:
            self.sentiment_classifier = nn.Linear(
                config.hidden_size,
                N_SENTIMENT_CLASSES,
            )
            self.similarity_linear = nn.Linear(
                BERT_HIDDEN_SIZE * 2,
                1,
            )
            self.paraphrase_classifier = nn.Linear(
                3 * config.hidden_size,
                1,
            )
            self.paraphrase_type_classifier = nn.Linear(
                3 * config.hidden_size,
                26,
            )

            self.sst_head_type = "baseline"
            self.qqp_mode = "p1"
            self.qqp_apply_dropout = True

        else:

            self.sentiment_classifier = nn.Linear(
                config.hidden_size,
                N_SENTIMENT_CLASSES,
            )

            self.sentiment_raw_cls_classifier = RawCLSClassificationHead(
                hidden_size=config.hidden_size,
                num_classes=N_SENTIMENT_CLASSES,
                dropout_prob=float(
                    getattr(config, "sst_head_dropout", 0.15)
                ),
            )
            self.sentiment_gated_attention_classifier = (
                GatedAttentionSentimentHead(
                    hidden_size=config.hidden_size,
                    num_classes=N_SENTIMENT_CLASSES,
                    dropout_prob=float(
                        getattr(config, "sst_head_dropout", 0.15)
                    ),
                )
            )

            self.sst_head_type = getattr(
                config,
                "sst_head",
                "baseline",
            )
            sst_head_dropout = float(
                getattr(
                    config,
                    "sst_head_dropout",
                    config.hidden_dropout_prob,
                )
            )
            sst_head_hidden = int(
                getattr(config, "sst_head_hidden", 192)
            )
            sst_multi_sample = int(
                getattr(config, "sst_multi_sample_dropout", 5)
            )

            self.sentiment_mean_classifier = MultiSampleClassificationHead(
                input_size=config.hidden_size,
                hidden_size=sst_head_hidden,
                num_classes=N_SENTIMENT_CLASSES,
                dropout_prob=sst_head_dropout,
                num_samples=sst_multi_sample,
            )

            self.sentiment_hybrid_classifier = MultiSampleClassificationHead(
                input_size=2 * config.hidden_size,
                hidden_size=sst_head_hidden,
                num_classes=N_SENTIMENT_CLASSES,
                dropout_prob=sst_head_dropout,
                num_samples=sst_multi_sample,
            )

            self.sentiment_tri_pool_classifier = MultiSampleClassificationHead(
                input_size=3 * config.hidden_size,
                hidden_size=sst_head_hidden,
                num_classes=N_SENTIMENT_CLASSES,
                dropout_prob=sst_head_dropout,
                num_samples=sst_multi_sample,
            )

            # Original/current Part-2 team-task initialization order.
            self.similarity_linear = nn.Linear(
                BERT_HIDDEN_SIZE * 2,
                1,
            )
            self.sts_diff_head = nn.Sequential(
                nn.Linear(BERT_HIDDEN_SIZE + 1, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            self.paraphrase_classifier = nn.Linear(
                3 * config.hidden_size,
                1,
            )
            self.paraphrase_siamese_classifier = nn.Linear(
                2 * config.hidden_size,
                1,
            )
            self.paraphrase_joint_classifier = nn.Linear(
                config.hidden_size,
                1,
            )

            self.qqp_mode = getattr(
                config,
                "qqp_mode",
                "p1",
            )
            self.qqp_apply_dropout = bool(
                getattr(config, "qqp_apply_dropout", False)
            )

            self.paraphrase_type_classifier = nn.Linear(
                3 * config.hidden_size,
                26,
            )

    def forward(self, input_ids, attention_mask):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    @staticmethod
    def _content_mask(attention_mask):
        mask = attention_mask.clone()
        if mask.ndim != 2:
            raise ValueError("SST attention mask must have shape [batch, length].")

        mask[:, 0] = 0
        lengths = attention_mask.long().sum(dim=1).clamp(min=1)
        sep_positions = (lengths - 1).unsqueeze(1)
        mask.scatter_(1, sep_positions, 0)

        # Extremely short/malformed input fallback: keep the first attended token
        # after CLS so pooling never operates on an empty set.
        empty_rows = mask.sum(dim=1) == 0
        if empty_rows.any() and mask.size(1) > 1:
            mask[empty_rows, 1] = attention_mask[empty_rows, 1]
        return mask

    @classmethod
    def _masked_mean_and_max(cls, hidden, attention_mask):
        content_mask = cls._content_mask(attention_mask)
        float_mask = content_mask.unsqueeze(-1).type_as(hidden)

        token_count = float_mask.sum(dim=1).clamp(min=1.0)
        mean_vector = (hidden * float_mask).sum(dim=1) / token_count

        very_negative = torch.finfo(hidden.dtype).min
        max_input = hidden.masked_fill(float_mask == 0, very_negative)
        max_vector = max_input.max(dim=1).values

        # Numerical fallback for pathological empty rows.
        max_vector = torch.where(
            torch.isfinite(max_vector),
            max_vector,
            hidden[:, 0],
        )
        return mean_vector, max_vector

    def mean_pooling(self, input_ids, attention_mask):

        outputs = self.forward(input_ids, attention_mask)
        token_embeddings = outputs["last_hidden_state"]        # [batch, seq_len, hidden]
        mask = attention_mask.unsqueeze(-1).float()             # [batch, seq_len, 1]
        summed = torch.sum(token_embeddings * mask, dim=1)       # [batch, hidden]
        counted = torch.clamp(mask.sum(dim=1), min=1e-9)         # [batch, 1] avoid divide-by-zero
        return summed / counted                                  # [batch, hidden]

    def predict_sentiment(self, input_ids, attention_mask):
        outputs = self.forward(input_ids, attention_mask)

        if self.sst_head_type == "baseline":
            pooled_output = outputs["pooler_output"]
            pooled_output = self.dropout(pooled_output)
            return self.sentiment_classifier(pooled_output)

        hidden = outputs["last_hidden_state"]

        if self.sst_head_type == "raw_cls":
            return self.sentiment_raw_cls_classifier(hidden[:, 0])

        if self.sst_head_type == "gated_attn":
            content_mask = self._content_mask(attention_mask)
            return self.sentiment_gated_attention_classifier(hidden, content_mask)

        mean_vector, max_vector = self._masked_mean_and_max(
            hidden,
            attention_mask,
        )

        if self.sst_head_type == "mean":
            return self.sentiment_mean_classifier(mean_vector)

        if self.sst_head_type == "hybrid":
            cls_vector = hidden[:, 0]
            features = torch.cat([cls_vector, mean_vector], dim=-1)
            return self.sentiment_hybrid_classifier(features)

        if self.sst_head_type == "tri_pool":
            pooled_cls = outputs["pooler_output"]
            features = torch.cat(
                [pooled_cls, mean_vector, max_vector],
                dim=-1,
            )
            return self.sentiment_tri_pool_classifier(features)

        raise ValueError(
            f"Unknown SST head {self.sst_head_type!r}; "
            "use baseline, raw_cls, gated_attn, mean, hybrid, or tri_pool."
        )

    @staticmethod
    def _build_qqp_joint_inputs(
        input_ids_1,
        attention_mask_1,
        input_ids_2,
        attention_mask_2,
        max_length: int = 512,
    ):

        batch_size = input_ids_1.size(0)
        pad_id = 0  # bert-base-uncased PAD id

        len1 = attention_mask_1.long().sum(dim=1)
        len2 = attention_mask_2.long().sum(dim=1)
        len2_without_cls = torch.clamp(len2 - 1, min=0)

        target_lengths = torch.clamp(
            len1 + len2_without_cls,
            max=max_length,
        )
        batch_max = max(1, int(target_lengths.max().item()))

        concat_ids = input_ids_1.new_full(
            (batch_size, batch_max),
            pad_id,
        )
        concat_mask = attention_mask_1.new_zeros(
            (batch_size, batch_max)
        )

        for row in range(batch_size):
            first_len = int(len1[row].item())
            second_len = int(len2_without_cls[row].item())
            keep = min(batch_max, first_len + second_len)

            first_take = min(first_len, keep)
            if first_take > 0:
                concat_ids[row, :first_take] = (
                    input_ids_1[row, :first_take]
                )
                concat_mask[row, :first_take] = (
                    attention_mask_1[row, :first_take]
                )

            second_take = keep - first_take
            if second_take > 0:
                concat_ids[row, first_take:keep] = (
                    input_ids_2[row, 1:1 + second_take]
                )
                concat_mask[row, first_take:keep] = (
                    attention_mask_2[row, 1:1 + second_take]
                )

        return concat_ids, concat_mask

    def predict_paraphrase(
        self,
        input_ids_1,
        attention_mask_1,
        input_ids_2,
        attention_mask_2,
    ):

        if self.qqp_mode == "p1":
            outputs_1 = self.forward(input_ids_1, attention_mask_1)
            cls_1 = outputs_1["pooler_output"]

            outputs_2 = self.forward(input_ids_2, attention_mask_2)
            cls_2 = outputs_2["pooler_output"]

            diff = torch.abs(cls_1 - cls_2)
            combined = torch.cat([cls_1, cls_2, diff], dim=-1)
            combined = self.dropout(combined)
            return self.paraphrase_classifier(combined)

        if self.qqp_mode == "siamese":
            outputs_1 = self.forward(input_ids_1, attention_mask_1)
            outputs_2 = self.forward(input_ids_2, attention_mask_2)
            cls_1 = outputs_1["pooler_output"]
            cls_2 = outputs_2["pooler_output"]
            combined = torch.cat([cls_1, cls_2], dim=-1)
            if self.qqp_apply_dropout:
                combined = self.dropout(combined)
            return self.paraphrase_siamese_classifier(combined)

        if self.qqp_mode == "joint":
            concat_ids, concat_mask = self._build_qqp_joint_inputs(
                input_ids_1,
                attention_mask_1,
                input_ids_2,
                attention_mask_2,
            )
            outputs = self.forward(concat_ids, concat_mask)
            pooled = outputs["pooler_output"]
            if self.qqp_apply_dropout:
                pooled = self.dropout(pooled)
            return self.paraphrase_joint_classifier(pooled)

        raise ValueError(
            f"Unknown QQP mode {self.qqp_mode!r}; "
            "use 'p1', 'siamese', or 'joint'."
        )

    def predict_similarity(
        self,
        input_ids_1,
        attention_mask_1,
        input_ids_2,
        attention_mask_2,
    ):
        u = self.mean_pooling(input_ids_1, attention_mask_1)
        v = self.mean_pooling(input_ids_2, attention_mask_2)

        cos_sim = F.cosine_similarity(u, v, dim=1) # range: [-1, 1]

        diff = torch.abs(u - v) # element wise difference

        combined = torch.cat([diff, cos_sim.unsqueeze(1)], dim=1)

        adjustment = self.sts_diff_head(combined).squeeze(-1)

        base_score = 2.5 * (cos_sim + 1)
        final_score = base_score + adjustment

        return final_score

    def predict_paraphrase_types(
        self,
        input_ids_1,
        attention_mask_1,
        input_ids_2,
        attention_mask_2,
    ):
        outputs_1 = self.forward(input_ids_1, attention_mask_1)
        cls_1 = outputs_1["pooler_output"]
        outputs_2 = self.forward(input_ids_2, attention_mask_2)
        cls_2 = outputs_2["pooler_output"]
        diff = torch.abs(cls_1 - cls_2)
        combined = torch.cat([cls_1, cls_2, diff], dim=-1)
        combined = self.dropout(combined)
        return self.paraphrase_type_classifier(combined)


def ordinal_neighbour_cross_entropy(
    logits,
    labels,
    smoothing: float,
    class_weights=None,
):

    epsilon = float(smoothing)
    if epsilon <= 0.0:
        return F.cross_entropy(logits, labels, weight=class_weights)

    num_classes = logits.size(-1)
    with torch.no_grad():
        target = torch.zeros_like(logits)
        target.scatter_(1, labels.view(-1, 1), 1.0 - epsilon)

        left = labels - 1
        right = labels + 1
        has_left = left >= 0
        has_right = right < num_classes
        neighbour_count = has_left.long() + has_right.long()
        neighbour_mass = epsilon / neighbour_count.clamp(min=1).to(logits.dtype)

        rows = torch.arange(labels.numel(), device=labels.device)
        if has_left.any():
            target[rows[has_left], left[has_left]] = neighbour_mass[has_left]
        if has_right.any():
            target[rows[has_right], right[has_right]] = neighbour_mass[has_right]

    log_probs = F.log_softmax(logits, dim=-1)
    per_example = -(target * log_probs).sum(dim=-1)
    if class_weights is not None:
        per_example = per_example * class_weights[labels]
    return per_example.mean()


def sst_primary_loss(logits, labels, args, class_weights=None):

    if args.loss_mode == "ordinal_smoothing":
        return ordinal_neighbour_cross_entropy(
            logits,
            labels,
            smoothing=args.label_smoothing,
            class_weights=class_weights,
        )
    if args.loss_mode == "uniform_smoothing":
        return F.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )
    if args.loss_mode == "ce":
        return F.cross_entropy(logits, labels, weight=class_weights)
    raise ValueError(f"Unknown SST loss mode: {args.loss_mode}")


def ordinal_auxiliary_loss(logits, labels):

    probabilities = F.softmax(logits, dim=-1)
    class_values = torch.arange(
        N_SENTIMENT_CLASSES,
        device=logits.device,
        dtype=probabilities.dtype,
    )
    expected_score = (probabilities * class_values).sum(dim=-1)
    return F.smooth_l1_loss(expected_score, labels.float().view(-1))


def symmetric_kl_loss(logits_a, logits_b):
    log_p = F.log_softmax(logits_a, dim=-1)
    log_q = F.log_softmax(logits_b, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    kl_pq = F.kl_div(log_p, q, reduction="batchmean")
    kl_qp = F.kl_div(log_q, p, reduction="batchmean")
    return 0.5 * (kl_pq + kl_qp)


def make_class_weights(train_examples, mode: str, device: torch.device):

    if mode == "none":
        return None

    counts = np.zeros(N_SENTIMENT_CLASSES, dtype=np.float64)
    for _, label, _ in train_examples:
        counts[int(label)] += 1.0

    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    weights = total / (N_SENTIMENT_CLASSES * counts)

    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    elif mode != "balanced":
        raise ValueError(f"Unknown class weighting mode: {mode}")

    weights = weights / weights.mean()
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"SST class weights ({mode}): {tensor.detach().cpu().tolist()}")
    return tensor


def _is_no_decay_parameter(name: str) -> bool:
    lower_name = name.lower()
    return (
        lower_name.endswith("bias")
        or "layer_norm" in lower_name
        or "layernorm" in lower_name
    )


def _bert_lr_for_parameter(model, name: str, args) -> float:

    base_lr = float(args.lr)
    decay = float(args.layerwise_lr_decay)

    if decay >= 0.999999:
        return base_lr

    num_layers = len(model.bert.bert_layers)
    match = re.search(r"bert\.bert_layers\.(\d+)\.", name)
    if match:
        layer_index = int(match.group(1))
        distance_from_top = max(0, num_layers - 1 - layer_index)
        return base_lr * (decay ** distance_from_top)

    # Embeddings receive the smallest BERT LR.
    if any(
        part in name
        for part in (
            "bert.word_embedding",
            "bert.pos_embedding",
            "bert.tk_type_embedding",
            "bert.embed_layer_norm",
        )
    ):
        return base_lr * (decay ** num_layers)

    # Pooler and any unmatched BERT parameter use the base LR.
    return base_lr


def freeze_bottom_bert_layers(model, number_to_freeze: int) -> None:

    if number_to_freeze <= 0:
        return

    max_layers = len(model.bert.bert_layers)
    number_to_freeze = min(number_to_freeze, max_layers)
    for layer_index in range(number_to_freeze):
        for parameter in model.bert.bert_layers[layer_index].parameters():
            parameter.requires_grad = False

    print(
        f"Frozen bottom {number_to_freeze}/{max_layers} BERT encoder layers."
    )


def set_gradual_bert_trainability(model, epoch_index: int, args) -> None:

    if not args.gradual_unfreeze or args.option != "finetune":
        return

    schedule = [
        int(part.strip())
        for part in str(args.gradual_unfreeze_schedule).split(",")
        if part.strip()
    ]
    if not schedule:
        return

    num_layers = len(model.bert.bert_layers)
    top_n = min(num_layers, schedule[min(epoch_index, len(schedule) - 1)])
    first_trainable = num_layers - top_n

    for index, layer in enumerate(model.bert.bert_layers):
        trainable = index >= first_trainable
        for parameter in layer.parameters():
            parameter.requires_grad = trainable

    # Keep embeddings conservative until every encoder layer is open.
    embeddings_trainable = top_n >= num_layers
    for name, parameter in model.bert.named_parameters():
        if "bert_layers" in name:
            continue
        if any(token in name for token in (
            "word_embedding", "pos_embedding", "tk_type_embedding", "embed_layer_norm"
        )):
            parameter.requires_grad = embeddings_trainable

    print(
        f"Gradual unfreezing epoch {epoch_index + 1}: top {top_n}/{num_layers} "
        f"BERT encoder layers trainable"
    )


def build_sst_optimizer(model, args):
    grouped: dict[tuple[str, float, float], list[torch.nn.Parameter]] = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and not args.gradual_unfreeze:
            continue

        if name.startswith("bert."):
            group_name = "bert"
            learning_rate = _bert_lr_for_parameter(model, name, args)
        else:
            group_name = "head"
            learning_rate = float(args.lr) * float(args.head_lr_multiplier)

        weight_decay = (
            0.0 if _is_no_decay_parameter(name) else float(args.weight_decay)
        )
        grouped.setdefault(
            (group_name, learning_rate, weight_decay), []
        ).append(parameter)

    parameter_groups = [
        {
            "params": params,
            "lr": lr,
            "base_lr": lr,
            "weight_decay": wd,
            "group_name": group_name,
        }
        for (group_name, lr, wd), params in grouped.items()
    ]

    if not parameter_groups:
        raise RuntimeError("No trainable SST parameters were found.")

    # Runtime-compatible with torch.optim.Optimizer parameter-group dictionaries.
    typed_groups = cast(
        Iterable[torch.nn.parameter.Parameter],
        parameter_groups,
    )
    return AdamW(typed_groups, lr=float(args.lr))


def get_optimizer_lrs(optimizer, model, args):

    del model, args  # group_name tags make this robust after scheduler updates.
    bert_lrs = [
        float(group["lr"])
        for group in optimizer.param_groups
        if group.get("group_name") == "bert"
    ]
    head_lrs = [
        float(group["lr"])
        for group in optimizer.param_groups
        if group.get("group_name") == "head"
    ]
    all_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if not all_lrs:
        return 0.0, 0.0
    bert_lr = max(bert_lrs) if bert_lrs else min(all_lrs)
    head_lr = max(head_lrs) if head_lrs else max(all_lrs)
    return bert_lr, head_lr



def save_sst_history(history, filepath: str) -> None:

    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "dev_loss",
        "train_accuracy",
        "dev_accuracy",
        "train_macro_f1",
        "dev_macro_f1",
        "dev_mae",
        "dev_qwk",
        "dev_adjacent_error_rate",
        "generalization_gap",
        "bert_lr",
        "head_lr",
        "grad_norm",
        "checkpoint_variant",
    ]

    with open(filepath, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_model(
    model,
    optimizer,
    args,
    config,
    filepath: str,
    epoch: Optional[int] = None,
    best_metrics: Optional[dict[str, Any]] = None,
) -> None:
    
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "model_config": config,
        "epoch": epoch,
        "best_metrics": best_metrics,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }
    torch.save(save_info, filepath)
    print(f"Saving the model to {filepath}.")


def safe_torch_load(filepath: str, device: torch.device):
    
    try:
        return torch.load(
            filepath,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(filepath, map_location=device)



def evaluate_sst(
    model,
    dataloader,
    device,
    has_labels: bool = True,
):

    model.eval()

    all_predictions: list[int] = []
    all_labels: list[int] = []
    all_sentence_ids: list[str] = []
    total_ce = 0.0
    total_labeled_examples = 0

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="eval-sst",
            disable=TQDM_DISABLE,
        ):
            token_ids = batch["token_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model.predict_sentiment(token_ids, attention_mask)

            predictions = logits.argmax(dim=-1).flatten().cpu().tolist()
            all_predictions.extend(int(x) for x in predictions)
            all_sentence_ids.extend(batch["sent_ids"])

            if has_labels:
                labels_tensor = batch["labels"].to(device).view(-1)
                batch_ce = F.cross_entropy(
                    logits,
                    labels_tensor,
                    reduction="sum",
                )
                total_ce += float(batch_ce.item())
                total_labeled_examples += int(labels_tensor.numel())
                all_labels.extend(
                    int(x)
                    for x in labels_tensor.detach().cpu().flatten().tolist()
                )

    accuracy = None
    metrics = None
    if has_labels:
        metrics = sst_classification_metrics(all_labels, all_predictions)
        metrics["loss"] = total_ce / max(1, total_labeled_examples)
        accuracy = float(metrics["accuracy"])

    return accuracy, all_predictions, all_sentence_ids, metrics


def write_sst_predictions(
    filepath: str,
    sentence_ids: Sequence[str],
    predictions: Sequence[int],
) -> None:
    
    if len(sentence_ids) != len(predictions):
        raise ValueError(
            "Number of sentence IDs does not match number of predictions."
        )

    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filepath, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "Predicted_Sentiment"])
        writer.writerows(zip(sentence_ids, predictions))



def _checkpoint_is_better(
    current: dict[str, Any],
    best: Optional[dict[str, Any]],
    min_acc_delta: float,
) -> bool:

    if best is None:
        return True

    current_acc = float(current["accuracy"])
    best_acc = float(best["accuracy"])

    if current_acc > best_acc + min_acc_delta:
        return True
    if current_acc < best_acc - min_acc_delta:
        return False

    current_f1 = float(current["macro_f1"])
    best_f1 = float(best["macro_f1"])
    if current_f1 > best_f1 + 1e-5:
        return True
    if current_f1 < best_f1 - 1e-5:
        return False

    return float(current["loss"]) < float(best["loss"]) - 1e-5


def _scheduler_value(dev_metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name == "dev_loss":
        return float(dev_metrics["loss"])
    if metric_name == "dev_accuracy":
        return float(dev_metrics["accuracy"])
    if metric_name == "dev_macro_f1":
        return float(dev_metrics["macro_f1"])
    raise ValueError(f"Unsupported scheduler metric: {metric_name}")


def _set_linear_warmup_lr(
    optimizer,
    global_step: int,
    warmup_steps: int,
) -> None:
    """Linearly warm each parameter group from 10% to its target LR."""
    if warmup_steps <= 0 or global_step > warmup_steps:
        return
    scale = max(0.1, float(global_step) / float(max(1, warmup_steps)))
    for group in optimizer.param_groups:
        target = float(group.get("base_lr", group["lr"]))
        group["lr"] = target * scale



def train_sst(args):

    device = get_device(args.use_gpu)

    train_examples = load_sst_data(args.sst_train, split="train")
    dev_examples = load_sst_data(args.sst_dev, split="dev")

    train_dataset = SentenceClassificationDataset(train_examples, args)
    dev_dataset = SentenceClassificationDataset(dev_examples, args)

    use_pin_memory = device.type == "cuda"
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=dev_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )

    config = SimpleNamespace(
        hidden_dropout_prob=args.hidden_dropout_prob,
        hidden_size=BERT_HIDDEN_SIZE,
        data_dir=".",
        option=args.option,
        local_files_only=args.local_files_only,
        sst_head=args.sst_head,
        sst_head_dropout=args.sst_head_dropout,
        sst_head_hidden=args.sst_head_hidden,
        sst_multi_sample_dropout=args.sst_multi_sample_dropout,
    )

    print("-" * 70)
    print("ADVANCED PART-2 SST CONFIGURATION")
    print("-" * 70)
    print(pformat(vars(args)))
    print("-" * 70)

    model = MultitaskBERT(config).to(device)

    if args.option == "finetune":
        if args.gradual_unfreeze:
            set_gradual_bert_trainability(model, 0, args)
        else:
            freeze_bottom_bert_layers(model, args.freeze_bottom_layers)

    optimizer = build_sst_optimizer(model, args)
    class_weights = make_class_weights(
        train_examples,
        args.class_weighting,
        device,
    )

    scheduler = None
    if args.scheduler == "plateau":
        scheduler_mode = (
            "min" if args.scheduler_metric == "dev_loss" else "max"
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            threshold=args.scheduler_threshold,
            threshold_mode="abs",
            cooldown=args.scheduler_cooldown,
            min_lr=args.min_lr,
        )

    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0.0 else None

    total_training_steps = max(1, len(train_dataloader) * args.epochs)
    warmup_steps = int(round(args.warmup_ratio * total_training_steps))
    global_step = 0
    if warmup_steps > 0:
        print(f"Linear LR warmup: {warmup_steps} optimizer steps.")

    best_metrics: Optional[dict[str, Any]] = None
    best_epoch = 0
    best_variant = "raw"
    epochs_without_checkpoint_improvement = 0
    best_dev_loss = float("inf")
    epochs_without_loss_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(args.epochs):
        if epoch > 0 and args.gradual_unfreeze and args.option == "finetune":
            set_gradual_bert_trainability(model, epoch, args)
        model.train()
        running_loss = 0.0
        num_batches = 0
        train_predictions: list[int] = []
        train_labels: list[int] = []
        grad_norm_sum = 0.0
        grad_norm_count = 0

        for batch in tqdm(
            train_dataloader,
            desc=f"sst-train-{epoch + 1:02}",
            disable=TQDM_DISABLE,
        ):
            token_ids = batch["token_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).view(-1)

            optimizer.zero_grad()
            global_step += 1
            _set_linear_warmup_lr(optimizer, global_step, warmup_steps)

            if args.rdrop_alpha > 0.0:
                logits_a = model.predict_sentiment(token_ids, attention_mask)
                logits_b = model.predict_sentiment(token_ids, attention_mask)
                ce_a = sst_primary_loss(
                    logits_a, labels, args, class_weights
                )
                ce_b = sst_primary_loss(
                    logits_b, labels, args, class_weights
                )
                ce_loss = 0.5 * (ce_a + ce_b)
                averaged_logits = 0.5 * (logits_a + logits_b)
                consistency = symmetric_kl_loss(logits_a, logits_b)
                loss = ce_loss + args.rdrop_alpha * consistency

                if args.ordinal_lambda > 0.0:
                    ordinal = 0.5 * (
                        ordinal_auxiliary_loss(logits_a, labels)
                        + ordinal_auxiliary_loss(logits_b, labels)
                    )
                    loss = loss + args.ordinal_lambda * ordinal
                logits_for_metrics = averaged_logits
            else:
                logits = model.predict_sentiment(token_ids, attention_mask)
                ce_loss = sst_primary_loss(
                    logits, labels, args, class_weights
                )
                loss = ce_loss
                if args.ordinal_lambda > 0.0:
                    loss = loss + args.ordinal_lambda * ordinal_auxiliary_loss(
                        logits,
                        labels,
                    )
                logits_for_metrics = logits

            loss.backward()

            if args.max_grad_norm > 0.0:
                total_norm = clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )
                grad_norm_sum += float(total_norm)
                grad_norm_count += 1

            optimizer.step()
            if ema is not None:
                ema.update(model)

            running_loss += float(loss.item())
            num_batches += 1
            train_predictions.extend(
                int(x)
                for x in logits_for_metrics.detach()
                .argmax(dim=-1)
                .cpu()
                .flatten()
                .tolist()
            )
            train_labels.extend(
                int(x) for x in labels.detach().cpu().flatten().tolist()
            )

        train_loss = running_loss / max(1, num_batches)
        train_metrics = sst_classification_metrics(
            train_labels,
            train_predictions,
        )

        raw_dev_acc, _, _, raw_dev_metrics = evaluate_sst(
            model,
            dev_dataloader,
            device,
            has_labels=True,
        )
        assert raw_dev_metrics is not None
        assert raw_dev_acc is not None

        candidate_metrics = dict(raw_dev_metrics)
        candidate_variant = "raw"

        if ema is not None and (epoch + 1) >= args.ema_eval_start_epoch:
            with ema.average_parameters(model):
                ema_dev_acc, _, _, ema_dev_metrics = evaluate_sst(
                    model,
                    dev_dataloader,
                    device,
                    has_labels=True,
                )
            assert ema_dev_metrics is not None
            assert ema_dev_acc is not None

            if _checkpoint_is_better(
                ema_dev_metrics,
                candidate_metrics,
                min_acc_delta=0.0,
            ):
                candidate_metrics = dict(ema_dev_metrics)
                candidate_variant = "ema"

        dev_metrics = candidate_metrics
        dev_acc = float(dev_metrics["accuracy"])
        generalization_gap = float(train_metrics["accuracy"] - dev_acc)
        average_grad_norm = (
            grad_norm_sum / grad_norm_count if grad_norm_count else 0.0
        )

        improved_checkpoint = _checkpoint_is_better(
            dev_metrics,
            best_metrics,
            args.early_stop_min_delta,
        )
        if improved_checkpoint:
            best_metrics = dict(dev_metrics)
            best_epoch = epoch + 1
            best_variant = candidate_variant
            epochs_without_checkpoint_improvement = 0

            if candidate_variant == "ema" and ema is not None:
                with ema.average_parameters(model):
                    save_model(
                        model,
                        optimizer,
                        args,
                        config,
                        args.filepath,
                        epoch=epoch + 1,
                        best_metrics=best_metrics,
                    )
            else:
                save_model(
                    model,
                    optimizer,
                    args,
                    config,
                    args.filepath,
                    epoch=epoch + 1,
                    best_metrics=best_metrics,
                )
        else:
            epochs_without_checkpoint_improvement += 1

        current_dev_loss = float(dev_metrics["loss"])
        if current_dev_loss < best_dev_loss - args.early_stop_loss_min_delta:
            best_dev_loss = current_dev_loss
            epochs_without_loss_improvement = 0
        else:
            epochs_without_loss_improvement += 1

        if scheduler is not None:
            scheduler.step(
                _scheduler_value(dev_metrics, args.scheduler_metric)
            )

        bert_lr, head_lr = get_optimizer_lrs(optimizer, model, args)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "dev_loss": current_dev_loss,
                "train_accuracy": float(train_metrics["accuracy"]),
                "dev_accuracy": float(dev_metrics["accuracy"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "dev_macro_f1": float(dev_metrics["macro_f1"]),
                "dev_mae": float(dev_metrics["mae"]),
                "dev_qwk": float(dev_metrics["qwk"]),
                "dev_adjacent_error_rate": float(
                    dev_metrics["adjacent_error_rate"]
                ),
                "generalization_gap": generalization_gap,
                "bert_lr": bert_lr,
                "head_lr": head_lr,
                "grad_norm": average_grad_norm,
                "checkpoint_variant": candidate_variant,
            }
        )
        save_sst_history(history, args.sst_history_out)

        print(
            f"Epoch {epoch + 1:02} (sst): "
            f"train_loss={train_loss:.4f}, "
            f"dev_loss={current_dev_loss:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}, "
            f"dev_acc={dev_metrics['accuracy']:.4f}, "
            f"dev_macro_F1={dev_metrics['macro_f1']:.4f}, "
            f"dev_MAE={dev_metrics['mae']:.4f}, "
            f"dev_QWK={dev_metrics['qwk']:.4f}, "
            f"adjacent_error={dev_metrics['adjacent_error_rate']:.4f}, "
            f"gap={generalization_gap:.4f}, "
            f"variant={candidate_variant}, "
            f"bert_lr={bert_lr:.3e}, head_lr={head_lr:.3e}"
        )

        if generalization_gap >= args.overfit_warn_gap:
            print(
                "OVERFITTING WARNING: train-dev accuracy gap is "
                f"{generalization_gap:.3f}. The best checkpoint remains "
                "protected and early stopping will prevent needless later "
                "epochs from becoming the final model."
            )

        enough_epochs = (epoch + 1) >= args.min_epochs_before_stop
        accuracy_stop = (
            args.early_stop_patience > 0
            and epochs_without_checkpoint_improvement
            >= args.early_stop_patience
        )
        loss_stop = (
            args.early_stop_loss_patience > 0
            and epochs_without_loss_improvement
            >= args.early_stop_loss_patience
        )

        if enough_epochs and (accuracy_stop or loss_stop):
            reason = []
            if accuracy_stop:
                reason.append("no useful dev-accuracy/checkpoint improvement")
            if loss_stop:
                reason.append("dev loss stopped improving")
            print(
                "Early stopping to limit overfitting: " + " and ".join(reason)
            )
            break

    if best_metrics is None:
        raise RuntimeError("Training finished without a valid SST checkpoint.")

    print("-" * 70)
    print(
        f"Best SST checkpoint: epoch {best_epoch} ({best_variant}), "
        f"dev accuracy={best_metrics['accuracy']:.4f}, "
        f"macro-F1={best_metrics['macro_f1']:.4f}, "
        f"dev loss={best_metrics['loss']:.4f}, "
        f"MAE={best_metrics['mae']:.4f}"
    )
    print(f"Checkpoint: {args.filepath}")
    print(f"Training history: {args.sst_history_out}")
    print("-" * 70)
    return float(best_metrics["accuracy"])



def test_sst_model(args):
    with torch.no_grad():
        device = get_device(args.use_gpu)
        saved = safe_torch_load(args.filepath, device)
        config = saved["model_config"]

        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        model.eval()
        print(f"Loaded best SST model from {args.filepath}")

        dev_examples = load_sst_data(args.sst_dev, split="dev")
        dev_dataset = SentenceClassificationDataset(dev_examples, args)
        dev_loader = DataLoader(
            dev_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=dev_dataset.collate_fn,
            num_workers=args.num_workers,
        )
        dev_acc, dev_pred, dev_ids, dev_metrics = evaluate_sst(
            model,
            dev_loader,
            device,
            has_labels=True,
        )
        write_sst_predictions(args.sst_dev_out, dev_ids, dev_pred)

        test_examples = load_sst_data(args.sst_test, split="test")
        test_dataset = SentenceClassificationTestDataset(test_examples, args)
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=test_dataset.collate_fn,
            num_workers=args.num_workers,
        )
        _, test_pred, test_ids, _ = evaluate_sst(
            model,
            test_loader,
            device,
            has_labels=False,
        )
        write_sst_predictions(args.sst_test_out, test_ids, test_pred)

        assert dev_metrics is not None
        print(
            f"SST best-checkpoint dev accuracy: {dev_acc:.4f}\n"
            f"SST dev macro-F1: {dev_metrics['macro_f1']:.4f}\n"
            f"SST dev loss: {dev_metrics['loss']:.4f}\n"
            f"SST dev MAE: {dev_metrics['mae']:.4f}\n"
            f"SST dev QWK: {dev_metrics['qwk']:.4f}\n"
            f"SST adjacent-error rate: "
            f"{dev_metrics['adjacent_error_rate']:.4f}\n"
            f"Wrote dev predictions to {args.sst_dev_out}\n"
            f"Wrote test predictions to {args.sst_test_out}"
        )


def _sst_checkpoint_path(args) -> str:
    return (
        "models/"
        f"{args.option}-sst-{args.sst_profile}-{args.sst_head}-"
        f"lr{args.lr}-{args.loss_mode}-ord{args.ordinal_lambda}-seed{args.seed}.pt"
    )


def _suffix_path(path: str, suffix: str) -> str:
    root, extension = os.path.splitext(path)
    return f"{root}-{suffix}{extension}"


def sst_probabilities_from_checkpoint(
    args,
    checkpoint_path: str,
    split: str,
):
    
    if split not in {"dev", "test"}:
        raise ValueError("Ensemble prediction split must be 'dev' or 'test'.")

    device = get_device(args.use_gpu)
    saved = safe_torch_load(checkpoint_path, device)
    config = saved["model_config"]

    model = MultitaskBERT(config)
    model.load_state_dict(saved["model"])
    model = model.to(device)
    model.eval()

    if split == "dev":
        examples = load_sst_data(args.sst_dev, split="dev")
        dataset = SentenceClassificationDataset(examples, args)
        has_labels = True
    else:
        examples = load_sst_data(args.sst_test, split="test")
        dataset = SentenceClassificationTestDataset(examples, args)
        has_labels = False

    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    all_probabilities = []
    all_ids: list[str] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=f"ensemble-{split}",
            disable=TQDM_DISABLE,
        ):
            token_ids = batch["token_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model.predict_sentiment(token_ids, attention_mask)
            probabilities = F.softmax(logits, dim=-1)

            all_probabilities.append(probabilities.cpu())
            all_ids.extend(batch["sent_ids"])

            if has_labels:
                all_labels.extend(
                    int(x)
                    for x in batch["labels"].cpu().flatten().tolist()
                )

    probabilities_tensor = torch.cat(all_probabilities, dim=0)
    return probabilities_tensor, all_ids, all_labels


def train_sst_ensemble(args):

    seed_strings = [part.strip() for part in args.ensemble_seeds.split(",")]
    seeds = []
    for part in seed_strings:
        if not part:
            continue
        seed = int(part)
        if seed not in seeds:
            seeds.append(seed)

    if not seeds:
        raise ValueError("--ensemble_seeds did not contain a valid integer seed.")

    print("=" * 70)
    print(f"SST SOFT-VOTING ENSEMBLE: seeds={seeds}")
    print("=" * 70)

    dev_probability_members = []
    test_probability_members = []
    reference_dev_ids = None
    reference_test_ids = None
    dev_labels = None
    member_rows = []

    for member_index, seed in enumerate(seeds, start=1):
        member_args = deepcopy(args)
        member_args.ensemble_seeds = ""
        member_args.seed = seed
        member_args.filepath = _sst_checkpoint_path(member_args)
        member_args.sst_history_out = _suffix_path(
            args.sst_history_out,
            f"seed{seed}",
        )

        print("-" * 70)
        print(
            f"Training ensemble member {member_index}/{len(seeds)} "
            f"with seed={seed}"
        )
        print("-" * 70)

        seed_everything(seed)
        best_accuracy = train_sst(member_args)

        dev_probs, dev_ids, labels = sst_probabilities_from_checkpoint(
            member_args,
            member_args.filepath,
            "dev",
        )
        test_probs, test_ids, _ = sst_probabilities_from_checkpoint(
            member_args,
            member_args.filepath,
            "test",
        )

        if reference_dev_ids is None:
            reference_dev_ids = list(dev_ids)
            reference_test_ids = list(test_ids)
            dev_labels = list(labels)
        else:
            if dev_ids != reference_dev_ids:
                raise RuntimeError(
                    "SST dev IDs differ across ensemble members."
                )
            if test_ids != reference_test_ids:
                raise RuntimeError(
                    "SST test IDs differ across ensemble members."
                )
            if labels != dev_labels:
                raise RuntimeError(
                    "SST dev labels differ across ensemble members."
                )

        dev_probability_members.append(dev_probs)
        test_probability_members.append(test_probs)
        member_rows.append(
            {
                "seed": seed,
                "checkpoint": member_args.filepath,
                "best_dev_accuracy": best_accuracy,
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_dev_probs = torch.stack(dev_probability_members, dim=0).mean(dim=0)
    mean_test_probs = torch.stack(test_probability_members, dim=0).mean(dim=0)

    dev_predictions = mean_dev_probs.argmax(dim=-1).tolist()
    test_predictions = mean_test_probs.argmax(dim=-1).tolist()

    assert reference_dev_ids is not None
    assert reference_test_ids is not None
    assert dev_labels is not None

    ensemble_metrics = sst_classification_metrics(
        dev_labels,
        [int(x) for x in dev_predictions],
    )

    write_sst_predictions(
        args.sst_dev_out,
        reference_dev_ids,
        [int(x) for x in dev_predictions],
    )
    write_sst_predictions(
        args.sst_test_out,
        reference_test_ids,
        [int(x) for x in test_predictions],
    )

    summary_path = "runs/sst_06_accuracy-ensemble-members.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=("seed", "checkpoint", "best_dev_accuracy"),
        )
        writer.writeheader()
        writer.writerows(member_rows)

    print("=" * 70)
    print(
        f"ENSEMBLE DEV ACCURACY: {ensemble_metrics['accuracy']:.4f}\n"
        f"ENSEMBLE MACRO-F1: {ensemble_metrics['macro_f1']:.4f}\n"
        f"ENSEMBLE MAE: {ensemble_metrics['mae']:.4f}\n"
        f"ENSEMBLE ADJACENT-ERROR RATE: "
        f"{ensemble_metrics['adjacent_error_rate']:.4f}\n"
        f"Dev predictions: {args.sst_dev_out}\n"
        f"Test predictions: {args.sst_test_out}\n"
        f"Member summary: {summary_path}"
    )
    print("=" * 70)
    return float(ensemble_metrics["accuracy"])


def _ensure_etpc_split(args) -> None:
    if args.task not in {"etpc", "multitask"}:
        return
    if args.etpc_train != "data/etpc-paraphrase-train-split.csv":
        return
    if os.path.exists(args.etpc_train) and os.path.exists(args.etpc_dev):
        return

    source = "data/etpc-paraphrase-train.csv"
    print("Creating ETPC 80/20 train/dev split...")
    with open(source, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError(f"Could not read ETPC header from {source}")

    rng = random.Random(11711)
    rng.shuffle(rows)
    cutoff = int(round(0.8 * len(rows)))
    train_rows = rows[:cutoff]
    dev_rows = rows[cutoff:]

    for path, selected_rows in (
        (args.etpc_train, train_rows),
        (args.etpc_dev, dev_rows),
    ):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_rows)
    print("ETPC split complete.")


def qqp_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, Any]:

    y_true_arr = np.asarray(y_true, dtype=np.int64).flatten()
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).flatten()

    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    total = len(y_true_arr)

    accuracy = float(tp + tn) / max(1, total)
    precision_pos = float(tp) / max(1, tp + fp)
    recall_pos = float(tp) / max(1, tp + fn)
    f1_pos = 2.0 * precision_pos * recall_pos / max(1e-9, precision_pos + recall_pos)

    precision_neg = float(tn) / max(1, tn + fn)
    recall_neg = float(tn) / max(1, tn + fp)
    f1_neg = 2.0 * precision_neg * recall_neg / max(1e-9, precision_neg + recall_neg)

    macro_f1 = (f1_pos + f1_neg) / 2.0
    weighted_f1 = (f1_pos * (tp + fn) + f1_neg * (tn + fp)) / max(1, total)

    mcc_denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "precision_neg": precision_neg,
        "recall_neg": recall_neg,
        "f1_neg": f1_neg,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


def evaluate_qqp_metrics(model, dataloader, device):
    if dataloader is None:
        return float("nan"), None

    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            ids1 = batch["token_ids_1"].to(device)
            mask1 = batch["attention_mask_1"].to(device)
            ids2 = batch["token_ids_2"].to(device)
            mask2 = batch["attention_mask_2"].to(device)
            labels = batch["labels"].to(device).float().view(-1)

            logits = model.predict_paraphrase(
                ids1, mask1, ids2, mask2
            ).view(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="sum"
            )
            total_loss += float(loss.item())
            total_examples += int(labels.numel())

            preds = logits.sigmoid().round().long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.long().cpu().numpy())

    mean_loss = total_loss / max(1, total_examples)
    metrics = qqp_classification_metrics(all_labels, all_preds)
    metrics["loss"] = mean_loss
    return mean_loss, metrics


def evaluate_qqp_loss(model, dataloader, device):
    loss, _ = evaluate_qqp_metrics(model, dataloader, device)
    return loss



def train_multitask(args):
    
    device = get_device(args.use_gpu)
    _ensure_etpc_split(args)

    sst_train_data, _, quora_train_data, sts_train_data, etpc_train_data = (
        load_multitask_data(
            args.sst_train,
            args.quora_train,
            args.sts_train,
            args.etpc_train,
            split="train",
        )
    )
    sst_dev_data, _, quora_dev_data, sts_dev_data, etpc_dev_data = (
        load_multitask_data(
            args.sst_dev,
            args.quora_dev,
            args.sts_dev,
            args.etpc_dev,
            split="dev",
        )
    )

    sst_train_dataloader = None
    sst_dev_dataloader = None
    quora_train_dataloader = None
    quora_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    etpc_train_dataloader = None
    etpc_dev_dataloader = None

    if args.task in {"sst", "multitask"}:
        sst_train_dataset = SentenceClassificationDataset(sst_train_data, args)
        sst_dev_dataset = SentenceClassificationDataset(sst_dev_data, args)
        sst_train_dataloader = DataLoader(
            sst_train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sst_train_dataset.collate_fn,
        )
        sst_dev_dataloader = DataLoader(
            sst_dev_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sst_dev_dataset.collate_fn,
        )

    if args.task in {"qqp", "multitask"}:
        quora_train_dataset = SentencePairDataset(quora_train_data, args)
        quora_dev_dataset = SentencePairDataset(quora_dev_data, args)
        quora_train_dataloader = DataLoader(
            quora_train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=quora_train_dataset.collate_fn,
        )
        quora_dev_dataloader = DataLoader(
            quora_dev_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=quora_dev_dataset.collate_fn,
        )

    if args.task in {"sts", "multitask"}:
        sts_train_dataset = SentencePairDataset(
            sts_train_data,
            args,
            isRegression=True,
        )
        sts_dev_dataset = SentencePairDataset(
            sts_dev_data,
            args,
            isRegression=True,
        )
        sts_train_dataloader = DataLoader(
            sts_train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sts_train_dataset.collate_fn,
        )
        sts_dev_dataloader = DataLoader(
            sts_dev_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sts_dev_dataset.collate_fn,
        )

    if args.task in {"etpc", "multitask"}:
        etpc_train_dataset = SentencePairDataset(etpc_train_data, args)
        etpc_dev_dataset = SentencePairDataset(etpc_dev_data, args)
        etpc_train_dataloader = DataLoader(
            etpc_train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=etpc_train_dataset.collate_fn,
        )
        etpc_dev_dataloader = DataLoader(
            etpc_dev_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=etpc_dev_dataset.collate_fn,
        )

    config = SimpleNamespace(
        hidden_dropout_prob=args.hidden_dropout_prob,
        hidden_size=BERT_HIDDEN_SIZE,
        data_dir=".",
        option=args.option,
        local_files_only=args.local_files_only,
        sst_head="baseline",
        sst_head_dropout=args.hidden_dropout_prob,
        sst_head_hidden=256,
        qqp_mode=getattr(args, "qqp_mode", "p1"),
        qqp_apply_dropout=getattr(args, "qqp_apply_dropout", False),
        strict_p1_baseline=(
            (args.task == "sst" and args.sst_profile in {"baseline", "readme_baseline"})
            or (args.task == "qqp" and args.qqp_profile == "baseline")
        ),
    )

    print("-" * 30)
    print("BERT Model Configuration")
    print("-" * 30)
    print(pformat(vars(args)))
    print("-" * 30)

    model = MultitaskBERT(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_dev_acc = float("-inf")
    qqp_history = []
    p1_sst_history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0

        if args.task in {"sst", "multitask"}:
            assert sst_train_dataloader is not None
            for batch in tqdm(
                sst_train_dataloader,
                desc=f"train-{epoch + 1:02}",
                disable=TQDM_DISABLE,
            ):
                b_ids = batch["token_ids"].to(device)
                b_mask = batch["attention_mask"].to(device)
                b_labels = batch["labels"].to(device)
                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1))
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                num_batches += 1

        if args.task in {"sts", "multitask"}:
            assert sts_train_dataloader is not None
            for batch in tqdm(
                sts_train_dataloader,
                desc=f"train-{epoch + 1:02}",
                disable=TQDM_DISABLE,
            ):
                b_ids1 = batch["token_ids_1"].to(device)
                b_mask1 = batch["attention_mask_1"].to(device)
                b_ids2 = batch["token_ids_2"].to(device)
                b_mask2 = batch["attention_mask_2"].to(device)
                b_labels = batch["labels"].to(device).float()
                optimizer.zero_grad()
                logits = model.predict_similarity(
                    b_ids1,
                    b_mask1,
                    b_ids2,
                    b_mask2,
                )
                loss = F.mse_loss(logits.flatten(), b_labels.flatten())
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                num_batches += 1

        if args.task in {"qqp", "multitask"}:
            assert quora_train_dataloader is not None
            for batch in tqdm(
                quora_train_dataloader,
                desc=f"train-{epoch + 1:02}",
                disable=TQDM_DISABLE,
            ):
                b_ids1 = batch["token_ids_1"].to(device)
                b_mask1 = batch["attention_mask_1"].to(device)
                b_ids2 = batch["token_ids_2"].to(device)
                b_mask2 = batch["attention_mask_2"].to(device)
                b_labels = batch["labels"].to(device).float()

                optimizer.zero_grad()

                if getattr(args, "qqp_bidirectional", False):
                    logits_12 = model.predict_paraphrase(
                        b_ids1,
                        b_mask1,
                        b_ids2,
                        b_mask2,
                    )
                    logits_21 = model.predict_paraphrase(
                        b_ids2,
                        b_mask2,
                        b_ids1,
                        b_mask1,
                    )
                    loss_12 = F.binary_cross_entropy_with_logits(
                        logits_12.view(-1),
                        b_labels.view(-1),
                    )
                    loss_21 = F.binary_cross_entropy_with_logits(
                        logits_21.view(-1),
                        b_labels.view(-1),
                    )
                    loss = 0.5 * (loss_12 + loss_21)
                else:
                    logits = model.predict_paraphrase(
                        b_ids1,
                        b_mask1,
                        b_ids2,
                        b_mask2,
                    )
                    loss = F.binary_cross_entropy_with_logits(
                        logits.view(-1),
                        b_labels.view(-1),
                    )

                loss.backward()

                qqp_clip = float(
                    getattr(args, "qqp_max_grad_norm", 0.0)
                )
                if qqp_clip > 0.0:
                    clip_grad_norm_(model.parameters(), qqp_clip)

                optimizer.step()
                train_loss += float(loss.item())
                num_batches += 1

        if args.task in {"etpc", "multitask"}:
            assert etpc_train_dataloader is not None
            for batch in tqdm(
                etpc_train_dataloader,
                desc=f"train-{epoch + 1:02}",
                disable=TQDM_DISABLE,
            ):
                b_ids1 = batch["token_ids_1"].to(device)
                b_mask1 = batch["attention_mask_1"].to(device)
                b_ids2 = batch["token_ids_2"].to(device)
                b_mask2 = batch["attention_mask_2"].to(device)
                b_labels = batch["labels"].to(device).float()
                optimizer.zero_grad()
                logits = model.predict_paraphrase_types(
                    b_ids1,
                    b_mask1,
                    b_ids2,
                    b_mask2,
                )
                loss = F.binary_cross_entropy_with_logits(logits, b_labels)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                num_batches += 1

        train_loss /= max(1, num_batches)

        (
            quora_train_acc,
            _,
            _,
            sst_train_acc,
            _,
            _,
            sts_train_corr,
            _,
            _,
            etpc_train_acc,
            _,
            _,
        ) = model_eval_multitask(
            sst_train_dataloader,
            quora_train_dataloader,
            sts_train_dataloader,
            etpc_train_dataloader,
            model=model,
            device=device,
            task=args.task,
        )

        (
            quora_dev_acc,
            _,
            _,
            sst_dev_acc,
            _,
            _,
            sts_dev_corr,
            _,
            _,
            etpc_dev_acc,
            _,
            _,
        ) = model_eval_multitask(
            sst_dev_dataloader,
            quora_dev_dataloader,
            sts_dev_dataloader,
            etpc_dev_dataloader,
            model=model,
            device=device,
            task=args.task,
        )

        train_acc, dev_acc = {
            "sst": (sst_train_acc, sst_dev_acc),
            "sts": (sts_train_corr, sts_dev_corr),
            "qqp": (quora_train_acc, quora_dev_acc),
            "etpc": (etpc_train_acc, etpc_dev_acc),
            "multitask": (0.0, 0.0),
        }[args.task]

        qqp_dev_loss = float("nan")
        qqp_dev_metrics = None
        if args.task == "qqp":
            qqp_dev_loss, qqp_dev_metrics = evaluate_qqp_metrics(
                model, quora_dev_dataloader, device
            )

        p1_sst_dev_loss = float("nan")
        p1_sst_train_metrics = None
        p1_sst_dev_metrics = None
        if args.task == "sst" and args.sst_profile in {"baseline", "readme_baseline"}:
            (
                _,
                _,
                _,
                p1_sst_train_metrics,
            ) = evaluate_sst(
                model,
                sst_train_dataloader,
                device,
                has_labels=True,
            )
            (
                _,
                _,
                _,
                p1_sst_dev_metrics,
            ) = evaluate_sst(
                model,
                sst_dev_dataloader,
                device,
                has_labels=True,
            )
            if p1_sst_dev_metrics is not None:
                p1_sst_dev_loss = float(p1_sst_dev_metrics["loss"])

        if args.task == "qqp" and qqp_dev_metrics is not None:
            print(
                f"Epoch {epoch + 1:02} (qqp): "
                f"train loss={train_loss:.3f}, dev loss={qqp_dev_loss:.3f}, "
                f"train_acc={train_acc:.3f}, dev_acc={dev_acc:.3f}, "
                f"dev_f1={qqp_dev_metrics['f1_pos']:.3f}, dev_mcc={qqp_dev_metrics['mcc']:.3f}"
            )
        else:
            print(
                f"Epoch {epoch + 1:02} ({args.task}): "
                f"train loss={train_loss:.3f}, "
                f"train={train_acc:.3f}, dev={dev_acc:.3f}"
            )

        if (
            args.task == "sst"
            and args.sst_profile in {"baseline", "readme_baseline"}
            and p1_sst_train_metrics is not None
            and p1_sst_dev_metrics is not None
        ):
            p1_sst_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "dev_loss": float(p1_sst_dev_loss),
                    "train_accuracy": float(
                        p1_sst_train_metrics["accuracy"]
                    ),
                    "dev_accuracy": float(
                        p1_sst_dev_metrics["accuracy"]
                    ),
                    "train_macro_f1": float(
                        p1_sst_train_metrics["macro_f1"]
                    ),
                    "dev_macro_f1": float(
                        p1_sst_dev_metrics["macro_f1"]
                    ),
                    "dev_mae": float(p1_sst_dev_metrics["mae"]),
                    "dev_qwk": float(p1_sst_dev_metrics["qwk"]),
                    "dev_adjacent_error_rate": float(
                        p1_sst_dev_metrics["adjacent_error_rate"]
                    ),
                    "generalization_gap": float(
                        p1_sst_train_metrics["accuracy"]
                        - p1_sst_dev_metrics["accuracy"]
                    ),
                    "bert_lr": float(optimizer.param_groups[0]["lr"]),
                    "head_lr": float(optimizer.param_groups[0]["lr"]),
                }
            )
            save_sst_history(
                p1_sst_history,
                args.sst_history_out,
            )

        if args.task == "qqp":
            record = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "dev_loss": float(qqp_dev_loss),
                "train_accuracy": float(train_acc),
                "dev_accuracy": float(dev_acc),
                "dev_f1_pos": float(qqp_dev_metrics["f1_pos"]) if qqp_dev_metrics else float("nan"),
                "dev_macro_f1": float(qqp_dev_metrics["macro_f1"]) if qqp_dev_metrics else float("nan"),
                "dev_weighted_f1": float(qqp_dev_metrics["weighted_f1"]) if qqp_dev_metrics else float("nan"),
                "dev_mcc": float(qqp_dev_metrics["mcc"]) if qqp_dev_metrics else float("nan"),
                "dev_precision_pos": float(qqp_dev_metrics["precision_pos"]) if qqp_dev_metrics else float("nan"),
                "dev_recall_pos": float(qqp_dev_metrics["recall_pos"]) if qqp_dev_metrics else float("nan"),
                "dev_precision_neg": float(qqp_dev_metrics["precision_neg"]) if qqp_dev_metrics else float("nan"),
                "dev_recall_neg": float(qqp_dev_metrics["recall_neg"]) if qqp_dev_metrics else float("nan"),
                "tp": int(qqp_dev_metrics["tp"]) if qqp_dev_metrics else 0,
                "tn": int(qqp_dev_metrics["tn"]) if qqp_dev_metrics else 0,
                "fp": int(qqp_dev_metrics["fp"]) if qqp_dev_metrics else 0,
                "fn": int(qqp_dev_metrics["fn"]) if qqp_dev_metrics else 0,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "qqp_profile": args.qqp_profile,
                "qqp_mode": args.qqp_mode,
                "bidirectional": int(args.qqp_bidirectional),
                "max_grad_norm": float(args.qqp_max_grad_norm),
            }
            qqp_history.append(record)
            qqp_history_dir = os.path.dirname(args.qqp_history_out)
            if qqp_history_dir:
                os.makedirs(qqp_history_dir, exist_ok=True)
            with open(
                args.qqp_history_out,
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                fieldnames = list(record.keys())
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(qqp_history)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(
                model,
                optimizer,
                args,
                config,
                args.filepath,
                epoch=epoch + 1,
                best_metrics={"accuracy": float(dev_acc)},
            )

    return best_dev_acc


def test_model(args):
    if args.task == "sst":
        return test_sst_model(args)

    with torch.no_grad():
        device = get_device(args.use_gpu)
        saved = safe_torch_load(args.filepath, device)
        config = saved["model_config"]
        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")
        result = test_model_multitask(args, model, device)

        if args.task == "qqp":
            quora_dev_data = load_multitask_data(
                args.sst_dev,
                args.quora_dev,
                args.sts_dev,
                args.etpc_dev,
                split="dev",
            )[2]
            quora_dev_dataset = SentencePairDataset(quora_dev_data, args)
            quora_dev_dataloader = DataLoader(
                quora_dev_dataset,
                shuffle=False,
                batch_size=args.batch_size,
                collate_fn=quora_dev_dataset.collate_fn,
            )
            _, qqp_metrics = evaluate_qqp_metrics(model, quora_dev_dataloader, device)
            if qqp_metrics is not None:
                print("=" * 70)
                print(f"FINAL QQP VALIDATION EVALUATION REPORT ({args.qqp_profile})")
                print("=" * 70)
                print(f"Accuracy:                  {qqp_metrics['accuracy']:.4f} ({qqp_metrics['accuracy']*100:.2f}%)")
                print(f"Paraphrase F1 (Class 1):   {qqp_metrics['f1_pos']:.4f}")
                print(f"Macro-Averaged F1:         {qqp_metrics['macro_f1']:.4f}")
                print(f"Weighted-Averaged F1:      {qqp_metrics['weighted_f1']:.4f}")
                print(f"Matthews Corr. Coeff (MCC):{qqp_metrics['mcc']:.4f}")
                print(f"Paraphrase Precision:      {qqp_metrics['precision_pos']:.4f} ({qqp_metrics['precision_pos']*100:.2f}%)")
                print(f"Paraphrase Recall:         {qqp_metrics['recall_pos']:.4f} ({qqp_metrics['recall_pos']*100:.2f}%)")
                print(f"Non-Paraphrase Precision:  {qqp_metrics['precision_neg']:.4f} ({qqp_metrics['precision_neg']*100:.2f}%)")
                print(f"Non-Paraphrase Recall:     {qqp_metrics['recall_neg']:.4f} ({qqp_metrics['recall_neg']*100:.2f}%)")
                print(f"Confusion Matrix (TN={qqp_metrics['tn']}, FP={qqp_metrics['fp']}, FN={qqp_metrics['fn']}, TP={qqp_metrics['tp']})")
                print("=" * 70)

        return result


QQP_PROFILES = {
    # Experiment 1: EXACT Part-1 QQP baseline.
    # u=BERT(q1), v=BERT(q2), [u,v,|u-v|] -> Dropout(0.3) -> Linear(2304,1).
    "baseline": {
        "qqp_mode": "p1",
        "qqp_apply_dropout": True,
        "hidden_dropout_prob": 0.3,
        "qqp_bidirectional": False,
        "qqp_max_grad_norm": 0.0,
    },

    # Experiment 2: P2 self-attention / joint sentence-pair encoding.
    # [CLS] q1 [SEP] q2 [SEP] goes through one BERT forward pass.
    "self_attention": {
        "qqp_mode": "joint",
        "qqp_apply_dropout": True,
        "hidden_dropout_prob": 0.3,
        "qqp_bidirectional": False,
        "qqp_max_grad_norm": 0.0,
    },

    # Experiment 3: same joint encoding but train in both question orders.
    "bidirectional": {
        "qqp_mode": "joint",
        "qqp_apply_dropout": True,
        "hidden_dropout_prob": 0.3,
        "qqp_bidirectional": True,
        "qqp_max_grad_norm": 0.0,
    },

    # Experiment 4: Joint encoding with gradient clipping 1.0.
    "gradient_clip": {
        "qqp_mode": "joint",
        "qqp_apply_dropout": True,
        "hidden_dropout_prob": 0.3,
        "qqp_bidirectional": False,
        "qqp_max_grad_norm": 1.0,
    },
    # Backwards-compatible alias for gradient_clip
    "final_clip": {
        "qqp_mode": "joint",
        "qqp_apply_dropout": True,
        "hidden_dropout_prob": 0.3,
        "qqp_bidirectional": False,
        "qqp_max_grad_norm": 1.0,
    },
}


def apply_qqp_profile(args):
    if args.task != "qqp":
        return args

    profile = QQP_PROFILES[args.qqp_profile]

    # These controls are intentionally profile-owned so each experiment command
    # reproduces one clearly defined QQP experiment.
    args.qqp_mode = profile["qqp_mode"]
    args.qqp_apply_dropout = profile["qqp_apply_dropout"]
    args.qqp_bidirectional = profile["qqp_bidirectional"]
    args.qqp_max_grad_norm = profile["qqp_max_grad_norm"]

    # hidden_dropout_prob is shared by the model constructor.
    if args.hidden_dropout_prob is None:
        args.hidden_dropout_prob = profile["hidden_dropout_prob"]

    return args



SST_PROFILES = {

    "baseline": {
        "lr": 1e-5,
        "batch_size": 64,
        "sst_head": "baseline",
        "sst_head_hidden": 256,
        "sst_multi_sample_dropout": 1,
        "hidden_dropout_prob": 0.3,
        "sst_head_dropout": 0.3,
        "loss_mode": "ce",
        "label_smoothing": 0.0,
        "ordinal_lambda": 0.0,
        "weight_decay": 0.0,
        "head_lr_multiplier": 1.0,
        "layerwise_lr_decay": 1.0,
        "gradual_unfreeze": False,
        "gradual_unfreeze_schedule": "12",
        "freeze_bottom_layers": 0,
        "max_grad_norm": 0.0,
        "rdrop_alpha": 0.0,
        "class_weighting": "none",
        "ema_decay": 0.0,
        "ema_eval_start_epoch": 1,
        "warmup_ratio": 0.0,
        "scheduler": "none",
        "scheduler_metric": "dev_accuracy",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "scheduler_threshold": 0.001,
        "scheduler_cooldown": 0,
        "min_lr": 1e-8,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "early_stop_loss_patience": 0,
        "early_stop_loss_min_delta": 0.0,
        "min_epochs_before_stop": 1,
        "overfit_warn_gap": 0.20,
    },
    "regularized": {
        "lr": 1e-5,
        "batch_size": 64,
        "sst_head": "baseline",
        "sst_head_hidden": 256,
        "sst_multi_sample_dropout": 1,
        "hidden_dropout_prob": 0.2,
        "sst_head_dropout": 0.2,
        "loss_mode": "uniform_smoothing",
        "label_smoothing": 0.05,
        "ordinal_lambda": 0.0,
        "weight_decay": 0.01,
        "head_lr_multiplier": 1.0,
        "layerwise_lr_decay": 1.0,
        "gradual_unfreeze": False,
        "gradual_unfreeze_schedule": "12",
        "freeze_bottom_layers": 0,
        "max_grad_norm": 1.0,
        "rdrop_alpha": 0.0,
        "class_weighting": "none",
        "ema_decay": 0.0,
        "ema_eval_start_epoch": 1,
        "warmup_ratio": 0.05,
        "scheduler": "plateau",
        "scheduler_metric": "dev_loss",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "scheduler_threshold": 0.002,
        "scheduler_cooldown": 0,
        "min_lr": 1e-8,
        "early_stop_patience": 3,
        "early_stop_min_delta": 0.001,
        "early_stop_loss_patience": 3,
        "early_stop_loss_min_delta": 0.002,
        "min_epochs_before_stop": 3,
        "overfit_warn_gap": 0.15,
    },
    "advanced": {
        "lr": 8e-6,
        "batch_size": 32,
        "sst_head": "hybrid",
        "sst_head_hidden": 192,
        "sst_multi_sample_dropout": 3,
        "hidden_dropout_prob": 0.2,
        "sst_head_dropout": 0.25,
        "loss_mode": "uniform_smoothing",
        "label_smoothing": 0.04,
        "ordinal_lambda": 0.02,
        "weight_decay": 0.01,
        "head_lr_multiplier": 2.0,
        "layerwise_lr_decay": 0.85,
        "gradual_unfreeze": False,
        "gradual_unfreeze_schedule": "12",
        "freeze_bottom_layers": 2,
        "max_grad_norm": 1.0,
        "rdrop_alpha": 0.0,
        "class_weighting": "none",
        "ema_decay": 0.99,
        "ema_eval_start_epoch": 2,
        "warmup_ratio": 0.10,
        "scheduler": "plateau",
        "scheduler_metric": "dev_loss",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "scheduler_threshold": 0.002,
        "scheduler_cooldown": 0,
        "min_lr": 1e-8,
        "early_stop_patience": 3,
        "early_stop_min_delta": 0.001,
        "early_stop_loss_patience": 3,
        "early_stop_loss_min_delta": 0.002,
        "min_epochs_before_stop": 3,
        "overfit_warn_gap": 0.15,
    },
    "strong": {
        # Accuracy-first Part-2 configuration.  It combines richer pooling with
        # conservative BERT updates and several low-cost anti-overfitting tools.
        "lr": 8e-6,
        "batch_size": 32,
        "sst_head": "tri_pool",
        "sst_head_hidden": 192,
        "sst_multi_sample_dropout": 5,
        "hidden_dropout_prob": 0.15,
        "sst_head_dropout": 0.30,
        "loss_mode": "uniform_smoothing",
        "label_smoothing": 0.03,
        # Keep the ordinal loss off by default because official SST is 5-class
        # accuracy. It remains available as an explicit ablation.
        "ordinal_lambda": 0.0,
        "weight_decay": 0.01,
        "head_lr_multiplier": 2.0,
        "layerwise_lr_decay": 0.85,
        "gradual_unfreeze": False,
        "gradual_unfreeze_schedule": "12",
        "freeze_bottom_layers": 3,
        "max_grad_norm": 1.0,
        "rdrop_alpha": 0.10,
        "class_weighting": "none",
        "ema_decay": 0.995,
        "ema_eval_start_epoch": 2,
        "warmup_ratio": 0.10,
        "scheduler": "plateau",
        "scheduler_metric": "dev_loss",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "scheduler_threshold": 0.002,
        "scheduler_cooldown": 0,
        "min_lr": 1e-8,
        "early_stop_patience": 3,
        "early_stop_min_delta": 0.001,
        "early_stop_loss_patience": 3,
        "early_stop_loss_min_delta": 0.002,
        "min_epochs_before_stop": 3,
        "overfit_warn_gap": 0.12,
    },

    "readme_baseline": {
        "lr": 1e-5,
        "batch_size": 64,
        "sst_head": "baseline",
        "sst_head_hidden": 256,
        "sst_multi_sample_dropout": 1,
        "hidden_dropout_prob": 0.3,
        "sst_head_dropout": 0.3,
        "loss_mode": "ce",
        "label_smoothing": 0.0,
        "ordinal_lambda": 0.0,
        "weight_decay": 0.0,
        "head_lr_multiplier": 1.0,
        "layerwise_lr_decay": 1.0,
        "gradual_unfreeze": False,
        "gradual_unfreeze_schedule": "12",
        "freeze_bottom_layers": 0,
        "max_grad_norm": 0.0,
        "rdrop_alpha": 0.0,
        "class_weighting": "none",
        "ema_decay": 0.0,
        "ema_eval_start_epoch": 1,
        "warmup_ratio": 0.0,
        "scheduler": "none",
        "scheduler_metric": "dev_accuracy",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "scheduler_threshold": 0.001,
        "scheduler_cooldown": 0,
        "min_lr": 1e-8,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "early_stop_loss_patience": 0,
        "early_stop_loss_min_delta": 0.0,
        "min_epochs_before_stop": 1,
        "overfit_warn_gap": 0.20,
    },
    "readme_dropout": {
        "lr": 1e-5,
        "batch_size": 64,
        "sst_head": "baseline",
        "sst_head_hidden": 256,
        "sst_multi_sample_dropout": 1,
        "hidden_dropout_prob": 0.3,
        "sst_head_dropout": 0.3,
        "loss_mode": "ce",
        "label_smoothing": 0.0,
        "ordinal_lambda": 0.0,
        "weight_decay": 0.0,
        "head_lr_multiplier": 1.0,
        "layerwise_lr_decay": 1.0,
        "gradual_unfreeze": False,
        "gradual_unfreeze_schedule": "12",
        "freeze_bottom_layers": 0,
        "max_grad_norm": 0.0,
        "rdrop_alpha": 0.0,
        "class_weighting": "none",
        "ema_decay": 0.0,
        "ema_eval_start_epoch": 1,
        "warmup_ratio": 0.0,
        "scheduler": "none",
        "scheduler_metric": "dev_accuracy",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "scheduler_threshold": 0.001,
        "scheduler_cooldown": 0,
        "min_lr": 1e-8,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "early_stop_loss_patience": 0,
        "early_stop_loss_min_delta": 0.0,
        "min_epochs_before_stop": 1,
        "overfit_warn_gap": 0.20,
    },
    "readme_low_lr": {
        "lr": 1e-6,
        "batch_size": 64,
        "sst_head": "baseline",
        "sst_head_hidden": 256,
        "sst_multi_sample_dropout": 1,
        "hidden_dropout_prob": 0.1,
        "sst_head_dropout": 0.1,
        "loss_mode": "ce",
        "label_smoothing": 0.0,
        "ordinal_lambda": 0.0,
        "weight_decay": 0.0,
        "head_lr_multiplier": 1.0,
        "layerwise_lr_decay": 1.0,
        "gradual_unfreeze": False,
        "gradual_unfreeze_schedule": "12",
        "freeze_bottom_layers": 0,
        "max_grad_norm": 0.0,
        "rdrop_alpha": 0.0,
        "class_weighting": "none",
        "ema_decay": 0.0,
        "ema_eval_start_epoch": 1,
        "warmup_ratio": 0.0,
        "scheduler": "none",
        "scheduler_metric": "dev_accuracy",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "scheduler_threshold": 0.001,
        "scheduler_cooldown": 0,
        "min_lr": 1e-8,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "early_stop_loss_patience": 0,
        "early_stop_loss_min_delta": 0.0,
        "min_epochs_before_stop": 1,
        "overfit_warn_gap": 0.20,
    },
    "readme_smoothing": {
        "lr": 1e-6,
        "batch_size": 64,
        "sst_head": "baseline",
        "sst_head_hidden": 256,
        "sst_multi_sample_dropout": 1,
        "hidden_dropout_prob": 0.1,
        "sst_head_dropout": 0.1,
        "loss_mode": "uniform_smoothing",
        "label_smoothing": 0.1,
        "ordinal_lambda": 0.0,
        "weight_decay": 0.0,
        "head_lr_multiplier": 1.0,
        "layerwise_lr_decay": 1.0,
        "gradual_unfreeze": False,
        "gradual_unfreeze_schedule": "12",
        "freeze_bottom_layers": 0,
        "max_grad_norm": 0.0,
        "rdrop_alpha": 0.0,
        "class_weighting": "none",
        "ema_decay": 0.0,
        "ema_eval_start_epoch": 1,
        "warmup_ratio": 0.0,
        "scheduler": "none",
        "scheduler_metric": "dev_accuracy",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "scheduler_threshold": 0.001,
        "scheduler_cooldown": 0,
        "min_lr": 1e-8,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "early_stop_loss_patience": 0,
        "early_stop_loss_min_delta": 0.0,
        "min_epochs_before_stop": 1,
        "overfit_warn_gap": 0.20,
    },
    "readme_plateau": {

        "lr": 1e-6,
        "batch_size": 64,
        "sst_head": "baseline",
        "sst_head_hidden": 256,
        "sst_multi_sample_dropout": 1,
        "hidden_dropout_prob": 0.1,
        "sst_head_dropout": 0.1,
        "loss_mode": "uniform_smoothing",
        "label_smoothing": 0.1,
        "ordinal_lambda": 0.0,
        "weight_decay": 0.0,
        "head_lr_multiplier": 1.0,
        "layerwise_lr_decay": 1.0,
        "gradual_unfreeze": False,
        "gradual_unfreeze_schedule": "12",
        "freeze_bottom_layers": 0,
        "max_grad_norm": 1.0,
        "rdrop_alpha": 0.0,
        "class_weighting": "none",
        "ema_decay": 0.0,
        "ema_eval_start_epoch": 1,
        "warmup_ratio": 0.0,
        "scheduler": "plateau",
        "scheduler_metric": "dev_accuracy",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "scheduler_threshold": 0.001,
        "scheduler_cooldown": 0,
        "min_lr": 1e-8,
        "early_stop_patience": 3,
        "early_stop_min_delta": 0.001,
        "early_stop_loss_patience": 3,
        "early_stop_loss_min_delta": 0.002,
        "min_epochs_before_stop": 4,
        "overfit_warn_gap": 0.15,
    },
    "readme_final_tuned": {

        "lr": 1e-5,
        "batch_size": 64,
        "sst_head": "baseline",
        "sst_head_hidden": 256,
        "sst_multi_sample_dropout": 1,
        "hidden_dropout_prob": 0.1,
        "sst_head_dropout": 0.1,
        "loss_mode": "uniform_smoothing",
        "label_smoothing": 0.05,
        "ordinal_lambda": 0.0,
        "weight_decay": 0.0,
        "head_lr_multiplier": 1.0,
        "layerwise_lr_decay": 1.0,
        "gradual_unfreeze": False,
        "gradual_unfreeze_schedule": "12",
        "freeze_bottom_layers": 0,
        "max_grad_norm": 1.0,
        "rdrop_alpha": 0.0,
        "class_weighting": "none",
        "ema_decay": 0.0,
        "ema_eval_start_epoch": 1,
        "warmup_ratio": 0.0,
        "scheduler": "plateau",
        "scheduler_metric": "dev_accuracy",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "scheduler_threshold": 0.001,
        "scheduler_cooldown": 0,
        "min_lr": 1e-8,
        "early_stop_patience": 3,
        "early_stop_min_delta": 0.001,
        "early_stop_loss_patience": 3,
        "early_stop_loss_min_delta": 0.002,
        "min_epochs_before_stop": 4,
        "overfit_warn_gap": 0.15,
    },
    "accuracy": {

        "lr": 1e-5,
        "batch_size": 32,
        "sst_head": "gated_attn",
        "sst_head_hidden": 128,
        "sst_multi_sample_dropout": 1,
        "hidden_dropout_prob": 0.10,
        "sst_head_dropout": 0.15,
        "loss_mode": "ordinal_smoothing",
        "label_smoothing": 0.04,
        "ordinal_lambda": 0.0,
        "weight_decay": 0.01,
        "head_lr_multiplier": 2.0,
        "layerwise_lr_decay": 0.90,
        "gradual_unfreeze": True,
        "gradual_unfreeze_schedule": "8,12",
        "freeze_bottom_layers": 0,
        "max_grad_norm": 1.0,
        "rdrop_alpha": 0.0,
        "class_weighting": "none",
        "ema_decay": 0.0,
        "ema_eval_start_epoch": 1,
        "warmup_ratio": 0.06,
        "scheduler": "plateau",
        "scheduler_metric": "dev_accuracy",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "scheduler_threshold": 0.001,
        "scheduler_cooldown": 0,
        "min_lr": 1e-8,
        "early_stop_patience": 3,
        "early_stop_min_delta": 0.0005,
        "early_stop_loss_patience": 3,
        "early_stop_loss_min_delta": 0.001,
        "min_epochs_before_stop": 3,
        "overfit_warn_gap": 0.15,
    },
}


def apply_sst_profile(args):

    if args.task == "sst":
        profile = SST_PROFILES[args.sst_profile]
        for key, value in profile.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
    else:
        if args.hidden_dropout_prob is None:
            args.hidden_dropout_prob = 0.3

        # Harmless fallback values used only when model construction expects
        # the attributes; non-SST training does not consume the SST controls.
        fallback = SST_PROFILES["baseline"]
        for key, value in fallback.items():
            if key in {"lr", "batch_size"}:
                continue
            if getattr(args, key, None) is None:
                setattr(args, key, value)

    return args


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def write_p1_baseline_reference(output_root: str):
    
    del output_root
    return None


def configure_experiment_outputs(args):

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)


    if args.task == "sst" and not args.sst_history_out:
        default_name = {
            "baseline": "sst_01_baseline.csv",
            "readme_baseline": "sst_01_baseline.csv",
            "readme_dropout": "sst_02_dropout.csv",
            "readme_low_lr": "sst_03_low_lr.csv",
            "readme_smoothing": "sst_04_label_smoothing.csv",
            "readme_plateau": "sst_05_plateau.csv",
            "readme_final_tuned": "sst_06_final_tuned.csv",
            "accuracy": "sst_06_accuracy.csv",
            "regularized": "sst_regularized.csv",
            "advanced": "sst_advanced.csv",
            "strong": "sst_strong.csv",
        }.get(
            args.sst_profile,
            f"sst_{_safe_name(args.sst_profile)}.csv",
        )
        args.sst_history_out = str(Path("runs") / default_name)

    if args.task == "qqp" and not args.qqp_history_out:
        default_name = {
            "baseline": "qqp_01_baseline.csv",
            "self_attention": "qqp_02_self_attention.csv",
            "bidirectional": "qqp_03_bidirectional.csv",
            "final_clip": "qqp_04_final_clip.csv",
        }.get(
            args.qqp_profile,
            f"qqp_{_safe_name(args.qqp_profile)}.csv",
        )
        args.qqp_history_out = str(Path("runs") / default_name)


    args.sst_dev_out = str(
        root / "sst-sentiment-dev-output.csv"
    )
    args.sst_test_out = str(
        root / "sst-sentiment-test-output.csv"
    )
    args.quora_dev_out = str(
        root / "quora-paraphrase-dev-output.csv"
    )
    args.quora_test_out = str(
        root / "quora-paraphrase-test-output.csv"
    )
    args.sts_dev_out = str(
        root / "sts-similarity-dev-output.csv"
    )
    args.sts_test_out = str(
        root / "sts-similarity-test-output.csv"
    )


    if args.etpc_dev_out is None:
        args.etpc_dev_out = (
            "predictions/bart/etpc-paraphrase-detection-dev-output.csv"
        )
    if args.etpc_test_out is None:
        args.etpc_test_out = (
            "predictions/bart/etpc-paraphrase-detection-test-output.csv"
        )

    args.experiment_dir = str(root)
    return args


def write_experiment_summary(args, observed_best_dev_accuracy):

    del args, observed_best_dev_accuracy
    return None


def is_final_sst_run(args) -> bool:

    return (
        args.task == "sst"
        and args.sst_profile == "accuracy"
        and bool(args.ensemble_seeds.strip())
    )


def is_submission_qqp_run(args) -> bool:
    return args.task == "qqp" and args.qqp_profile in {"gradient_clip", "bidirectional", "self_attention", "final_clip"}


def print_p1_baseline_policy(args) -> None:
    if args.task == "sst" and args.sst_profile in {"baseline", "readme_baseline"}:
        ref = P1_BASELINE_REFERENCE["sst"]["dev_accuracy"]
        print(
            f"P1 SST baseline is fixed at {ref:.3f}. "
            "This new run is only a reproduction check."
        )
    if args.task == "qqp" and args.qqp_profile == "baseline":
        ref = P1_BASELINE_REFERENCE["qqp"]["dev_accuracy"]
        print(
            f"P1 QQP baseline is fixed at {ref:.3f}. "
            "This new run is only a reproduction check."
        )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=("sst", "sts", "qqp", "etpc", "multitask"),
        default="sst",
    )
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--option",
        choices=("pretrain", "finetune"),
        default="pretrain",
        help=(
            "pretrain: freeze BERT; finetune: update BERT parameters"
        ),
    )
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument(
        "--no_runlog",
        action="store_true",
        help="Disable automatic .runs metrics/progress logging.",
    )
    parser.add_argument(
        "--runlog_prefix",
        type=str,
        default=None,
        help="Optional prefix for the built-in .runs logger.",
    )

    # QQP Part-1 -> Part-2 controlled ablations.
    parser.add_argument(
        "--qqp_profile",
        choices=(
            "baseline",
            "self_attention",
            "bidirectional",
            "gradient_clip",
            "final_clip",
        ),
        default="gradient_clip",
        help=(
            "QQP ablation: exact P1 [u,v,|u-v|]+dropout baseline, "
            "joint self-attention, joint+bidirectional training, or "
            "joint+gradient clipping."
        ),
    )
    parser.add_argument(
        "--qqp_history_out",
        type=str,
        default=None,
        help="Default: predictions/bert/qqp/<profile>/history.csv",
    )

    parser.add_argument(
        "--sst_profile",
        choices=(
            "baseline", "regularized", "advanced", "strong", "accuracy",
            "readme_baseline", "readme_dropout", "readme_low_lr",
            "readme_smoothing", "readme_plateau", "readme_final_tuned",
        ),
        default="readme_final_tuned",
    )

    # Part-2 SST controls. None means "use profile value".
    parser.add_argument(
        "--sst_head",
        choices=("baseline", "raw_cls", "gated_attn", "mean", "hybrid", "tri_pool"),
        default=None,
    )
    parser.add_argument("--sst_head_hidden", type=int, default=None)
    parser.add_argument("--sst_multi_sample_dropout", type=int, default=None)
    parser.add_argument("--sst_head_dropout", type=float, default=None)
    parser.add_argument("--hidden_dropout_prob", type=float, default=None)
    parser.add_argument(
        "--loss_mode",
        choices=("ce", "uniform_smoothing", "ordinal_smoothing"),
        default=None,
    )
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--ordinal_lambda", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--head_lr_multiplier", type=float, default=None)
    parser.add_argument("--layerwise_lr_decay", type=float, default=None)
    parser.add_argument(
        "--gradual_unfreeze",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Gradually open lower BERT layers according to --gradual_unfreeze_schedule.",
    )
    parser.add_argument(
        "--gradual_unfreeze_schedule",
        type=str,
        default=None,
        help="Comma-separated number of top encoder layers trainable by epoch, e.g. 8,12.",
    )
    parser.add_argument("--freeze_bottom_layers", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--rdrop_alpha", type=float, default=None)
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--ema_eval_start_epoch", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument(
        "--class_weighting",
        choices=("none", "balanced", "sqrt_balanced"),
        default=None,
    )
    parser.add_argument(
        "--scheduler",
        choices=("none", "plateau"),
        default=None,
    )
    parser.add_argument(
        "--scheduler_metric",
        choices=("dev_loss", "dev_accuracy", "dev_macro_f1"),
        default=None,
    )
    parser.add_argument("--scheduler_factor", type=float, default=None)
    parser.add_argument("--scheduler_patience", type=int, default=None)
    parser.add_argument("--scheduler_threshold", type=float, default=None)
    parser.add_argument("--scheduler_cooldown", type=int, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=None)
    parser.add_argument(
        "--early_stop_loss_patience",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--early_stop_loss_min_delta",
        type=float,
        default=None,
    )
    parser.add_argument("--min_epochs_before_stop", type=int, default=None)
    parser.add_argument("--overfit_warn_gap", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--sst_history_out",
        type=str,
        default=None,
        help="Default: predictions/bert/sst/<profile>/history.csv",
    )

    # Determine official LR default from option before adding --lr.
    partial, _ = parser.parse_known_args()

    parser.add_argument(
        "--output_root",
        type=str,
        default="predictions/bert",
        help="Directory for canonical final BERT prediction CSVs.",
    )

    # Data paths.
    parser.add_argument(
        "--sst_train",
        default="data/sst-sentiment-train.csv",
    )
    parser.add_argument(
        "--sst_dev",
        default="data/sst-sentiment-dev.csv",
    )
    parser.add_argument(
        "--sst_test",
        default="data/sst-sentiment-test-student.csv",
    )
    parser.add_argument(
        "--quora_train",
        default="data/quora-paraphrase-train.csv",
    )
    parser.add_argument(
        "--quora_dev",
        default="data/quora-paraphrase-dev.csv",
    )
    parser.add_argument(
        "--quora_test",
        default="data/quora-paraphrase-test-student.csv",
    )
    parser.add_argument(
        "--sts_train",
        default="data/sts-similarity-train.csv",
    )
    parser.add_argument(
        "--sts_dev",
        default="data/sts-similarity-dev.csv",
    )
    parser.add_argument(
        "--sts_test",
        default="data/sts-similarity-test-student.csv",
    )
    parser.add_argument(
        "--etpc_train",
        default="data/etpc-paraphrase-train-split.csv",
    )
    parser.add_argument(
        "--etpc_dev",
        default="data/etpc-paraphrase-dev.csv",
    )
    parser.add_argument(
        "--etpc_test",
        default="data/etpc-paraphrase-detection-test-student.csv",
    )

    # Output paths. None means profile-specific output under --output_root.
    parser.add_argument("--sst_dev_out", default=None)
    parser.add_argument("--sst_test_out", default=None)
    parser.add_argument("--quora_dev_out", default=None)
    parser.add_argument("--quora_test_out", default=None)
    parser.add_argument("--sts_dev_out", default=None)
    parser.add_argument("--sts_test_out", default=None)
    parser.add_argument("--etpc_dev_out", default=None)
    parser.add_argument("--etpc_test_out", default=None)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="SST profile chooses 32/64 by default; override if necessary.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="SST profile chooses the fine-tuning LR; non-SST keeps P1 defaults.",
    )

    parser.add_argument(
        "--ensemble_seeds",
        type=str,
        default="",
        help=(
            "Optional comma-separated SST seeds, e.g. 11711,42,2026. "
            "Each seed is trained independently and probabilities are averaged."
        ),
    )

    args = parser.parse_args()
    args = apply_qqp_profile(args)
    args = apply_sst_profile(args)

    if args.task != "sst":
        if args.batch_size is None:
            args.batch_size = 64
        if args.lr is None:
            args.lr = 1e-3 if args.option == "pretrain" else 1e-5

    args = configure_experiment_outputs(args)
    return args


if __name__ == "__main__":
    # Parse arguments before starting the logger.
    args = get_args()

    if args.task == "sst":
        args.filepath = _sst_checkpoint_path(args)
    elif args.task == "qqp":
        args.filepath = (
            f"models/{args.option}-{args.epochs}-{args.lr}-"
            f"qqp-{args.qqp_profile}.pt"
        )
    else:
        args.filepath = (
            f"models/{args.option}-{args.epochs}-{args.lr}-{args.task}.pt"
        )

    # Logs are research artifacts, not submission predictions.
    LOG_DIR = Path(".runs")

    runlog = None
    if not args.no_runlog:
        prefix = args.runlog_prefix or _default_runlog_prefix(args)
        runlog = start_runlog(prefix=prefix)
        print(f"Run logs: {runlog.paths}")

    try:
        seed_everything(args.seed)
        print_p1_baseline_policy(args)

        if is_final_sst_run(args):
            train_sst_ensemble(args)

            print("Final SST predictions:")
            print(f"  {args.sst_dev_out}")
            print(f"  {args.sst_test_out}")

        elif args.task == "sst" and args.sst_profile in {
            "baseline",
            "readme_baseline",
        }:
            # Exact P1-style reproduction path.
            train_multitask(args)
            print(
                "SST baseline reproduction complete. "
                "History/checkpoint saved; canonical final predictions unchanged."
            )

        elif args.task == "sst":
            train_sst(args)
            print(
                "SST ablation complete. "
                "History/checkpoint saved; canonical final predictions unchanged."
            )

        elif args.task == "qqp":
            train_multitask(args)

            if is_submission_qqp_run(args):
                test_model(args)
                print("QQP prediction outputs written:")
                print(f"  {args.quora_dev_out}")
                print(f"  {args.quora_test_out}")
            else:
                print(
                    "QQP baseline complete. "
                    "History/checkpoint saved; prediction files untouched."
                )

        else:
            train_multitask(args)
            test_model(args)

    finally:
        if runlog is not None:
            runlog.close()