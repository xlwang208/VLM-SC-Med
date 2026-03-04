from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

# patient-level multilabel stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch
import torch.nn as nn
import torch.utils.data as tud

# Lightning import (supports both old/new package names)
try:
    import lightning.pytorch as pl
except Exception:
    import pytorch_lightning as pl

import re


LABELS = ["head", "neck", "chest", "abdomen", "pelvis"]
LABEL_TO_ID = {k: i for i, k in enumerate(LABELS)}
N_CLASSES = len(LABELS)
VALID = set(LABELS)


def strip_nii_suffix(s: str) -> str:
    s = (s or "").strip()
    if s.endswith(".nii.gz"):
        return s[:-7]
    if s.endswith(".nii"):
        return s[:-4]
    return s


def extract_patient_id(volume_id_norm: str) -> str:
    parts = str(volume_id_norm).split("/")
    if len(parts) >= 3:
        return parts[2]
    return str(volume_id_norm)


def parse_raw_label_set_from_cell(cell: Any) -> Set[str]:
    """
    exact_cases.csv gt_top3_set examples:
      chest
      chest,abdomen
      "chest,abdomen"
    We parse ALL tokens, lowercased, but DO NOT filter here.
    Filtering/drop happens later so we can count unknown labels.
    """
    if cell is None:
        return set()
    s = str(cell).strip().strip('"')
    if not s:
        return set()
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    return set(parts)


def parse_round3_label_set(cell: Any) -> Set[str]:
    if cell is None:
        return set()
    s = str(cell).strip().strip('"').strip()
    if not s:
        return set()

    s = s.lower()
    
    s = re.sub(r"\band\b", ",", s)
    s = s.replace("|", ",").replace(";", ",").replace("/", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    
    out = set()
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.add(p)
    return out


def multilabel_to_matrix(label_sets: List[Set[str]]) -> np.ndarray:
    Y = np.zeros((len(label_sets), N_CLASSES), dtype=np.int64)
    for i, s in enumerate(label_sets):
        for t in s:
            if t in VALID:
                Y[i, LABEL_TO_ID[t]] = 1
    return Y


def matrix_to_label_list(proba_row: np.ndarray, threshold: float, min_k: int, max_k: int) -> List[str]:
    assert proba_row.shape == (N_CLASSES,)
    max_k = min(max_k, N_CLASSES)
    min_k = max(0, min(min_k, N_CLASSES))

    order = np.argsort(-proba_row)  # desc
    chosen = [i for i in range(N_CLASSES) if proba_row[i] >= threshold]

    if len(chosen) < min_k:
        for i in order:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= min_k:
                break

    if len(chosen) > max_k:
        chosen = sorted(chosen, key=lambda i: -proba_row[i])[:max_k]

    chosen = sorted(chosen, key=lambda i: -proba_row[i])
    return [LABELS[i] for i in chosen]


def load_cases_from_vlm_image_jsonl(jsonl_path: str) -> pd.DataFrame:
    """
    Read VLM output_image.jsonl and return a DataFrame with:
      - volume_id_norm
      - label_raw (set[str])
    Using:
      obj["llm"]["parsed"]["labels"]
    Fallback:
      parse obj["llm"]["raw_text"] as JSON and read ["labels"].
    """

    rows = []
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"vlm_image_jsonl not found: {path}")

    n_bad = 0
    n_no_llm = 0
    n_no_labels = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                n_bad += 1
                continue

            vid = obj.get("volume_id_norm") or obj.get("volume_id")
            if not vid:
                continue

            volume_id_norm = strip_nii_suffix(str(vid))

            llm = obj.get("llm", None)
            if not isinstance(llm, dict):
                n_no_llm += 1
                continue

            labels = None

            # primary: llm.parsed.labels
            parsed = llm.get("parsed", None)
            if isinstance(parsed, dict):
                labels = parsed.get("labels", None)

            # fallback: llm.raw_text
            if labels is None:
                raw_text = llm.get("raw_text", None)
                if isinstance(raw_text, str) and raw_text.strip():
                    try:
                        raw_obj = json.loads(raw_text)
                        if isinstance(raw_obj, dict):
                            labels = raw_obj.get("labels", None)
                    except Exception:
                        pass

            if labels is None:
                n_no_labels += 1
                labels_set = set()
            else:
                if isinstance(labels, list):
                    labels_set = set(str(x).strip().lower() for x in labels if str(x).strip())
                else:
                    labels_set = parse_raw_label_set_from_cell(labels)

            rows.append({
                "volume_id_norm": volume_id_norm,
                "label_raw": labels_set,
            })

    df = pd.DataFrame(rows)

    print(
        f"[INFO] loaded vlm_image_jsonl: rows={len(df)} "
        f"bad_lines={n_bad} no_llm={n_no_llm} no_labels={n_no_labels}"
    )

    return df


def load_embeddings_from_paths(paths: List[str]) -> np.ndarray:
    X = []
    for p in tqdm(paths, desc="Loading .npy embeddings"):
        arr = np.load(p)
        X.append(arr)
    return np.stack(X, axis=0)


def load_cases_from_round3_csv(round3_csv_path: str, volume_col: str = "volume_id", label_col: str = "final_consistency_relabel") -> pd.DataFrame:
    path = Path(round3_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"round3 csv not found: {path}")

    df = pd.read_csv(path)

    if volume_col not in df.columns or label_col not in df.columns:
        raise KeyError(
            f"round3 csv missing required columns: {volume_col}, {label_col}. "
            f"Got: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["volume_id_norm"] = df[volume_col].astype(str).map(strip_nii_suffix)
    out["label_raw"] = df[label_col].map(parse_round3_label_set)

    print(f"[INFO] loaded round3_csv: rows={len(out)} from {path}")
    return out

@dataclass
class SplitResult:
    train_patients: List[str]
    val_patients: List[str]
    test_patients: List[str]


def patient_level_stratified_split(
    df: pd.DataFrame,
    patient_col: str,
    labelset_col: str,
    seed: int,
) -> SplitResult:
    patient_to_labels: Dict[str, Set[str]] = {}
    for pid, sub in df.groupby(patient_col):
        u: Set[str] = set()
        for s in sub[labelset_col]:
            u |= set(s)
        patient_to_labels[pid] = u

    patients = sorted(patient_to_labels.keys())
    Yp = multilabel_to_matrix([patient_to_labels[p] for p in patients])

    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(mskf.split(np.zeros((len(patients), 1)), Yp))

    test_idx = folds[0][1]
    val_idx = folds[1][1]
    train_idx = np.concatenate([folds[i][1] for i in [2, 3, 4]])

    return SplitResult(
        train_patients=[patients[i] for i in train_idx],
        val_patients=[patients[i] for i in val_idx],
        test_patients=[patients[i] for i in test_idx],
    )


class BinaryMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))  # 1 logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)


class LitFiveIndependent(pl.LightningModule):
    def __init__(self, in_dim: int, hidden_sizes: Tuple[int, ...], lr: float, pos_weight_1d: np.ndarray, weight_decay: float):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight_1d"])
        self.lr = lr

        # 5 independent MLPs (no shared weights)
        self.mlps = nn.ModuleList([BinaryMLP(in_dim, hidden_sizes) for _ in range(N_CLASSES)])

        # 5 independent losses (each has its own pos_weight scalar)
        self.loss_fns = nn.ModuleList([
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight_1d[i]), dtype=torch.float32))
            for i in range(N_CLASSES)
        ])

    def training_step(self, batch, batch_idx):
        x, y = batch  # y: (B,5)
        losses = []
        for i in range(N_CLASSES):
            logits = self.mlps[i](x)      # (B,)
            yi = y[:, i]                  # (B,)
            li = self.loss_fns[i](logits, yi)
            losses.append(li)
            self.log(f"train_loss_{LABELS[i]}", li, on_step=True, on_epoch=True, prog_bar=False)

        loss_total = torch.stack(losses).mean()
        self.log("train_loss", loss_total, on_step=True, on_epoch=True, prog_bar=True)
        return loss_total

    def validation_step(self, batch, batch_idx):
        x, y = batch
        losses = []
        for i in range(N_CLASSES):
            logits = self.mlps[i](x)
            yi = y[:, i]
            li = self.loss_fns[i](logits, yi)
            losses.append(li)
            self.log(f"val_loss_{LABELS[i]}", li, on_step=False, on_epoch=True, prog_bar=False)

        loss_total = torch.stack(losses).mean()
        self.log("val_loss", loss_total, on_step=False, on_epoch=True, prog_bar=True)
        return loss_total

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)


def compute_pos_weight(Y_train: np.ndarray, power: float, cap: float) -> np.ndarray:
    pos = Y_train.sum(axis=0).astype(np.float64)
    neg = (Y_train.shape[0] - pos).astype(np.float64)
    raw = neg / np.maximum(pos, 1.0)

    pw = np.power(raw, power)  
    pw = np.clip(pw, 1.0, cap) 
    return pw


def make_weighted_sampler(Y_train: np.ndarray) -> tud.WeightedRandomSampler:
    """
    Sample-level weights to increase presence of rare-label positive samples.
    Simple, stable heuristic:
      class_w[c] = 1 / sqrt(freq_pos[c])
      sample_w[i] = sum_c y[i,c] * class_w[c]
    """
    pos = Y_train.sum(axis=0).astype(np.float64)
    freq = pos / max(float(Y_train.shape[0]), 1.0)
    class_w = 1.0 / np.sqrt(np.maximum(freq, 1e-12))

    sample_w = (Y_train * class_w.reshape(1, -1)).sum(axis=1)
    sample_w = np.maximum(sample_w, 1e-8)

    weights = torch.tensor(sample_w, dtype=torch.double)
    sampler = tud.WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler


class LossCurvePngCallback(pl.Callback):
    def __init__(self, out_dir: Path):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.history: Dict[str, List[float]] = {}

    def _record(self, metrics: Dict[str, Any], key: str):
        v = metrics.get(key, None)
        if v is None:
            return
        try:
            fv = float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
        except Exception:
            return
        self.history.setdefault(key, []).append(fv)

    def on_validation_epoch_end(self, trainer, pl_module):
        # callback_metrics contains aggregated epoch metrics
        m = trainer.callback_metrics

        # total
        self._record(m, "train_loss_epoch")  
        self._record(m, "train_loss")   
        self._record(m, "val_loss")

        # per label
        for lab in LABELS:
            self._record(m, f"train_loss_{lab}")
            self._record(m, f"val_loss_{lab}")

        self._save_png()

    def _save_png(self):
        if not self.history:
            return
        import matplotlib.pyplot as plt

        png_path = self.out_dir / "loss_curves.png"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure()

        # ---- total loss ----
        if "train_loss" in self.history and len(self.history["train_loss"]) > 0:
            plt.plot(
                self.history["train_loss"],
                label="train_loss",
                linestyle="-",  
            )

        if "val_loss" in self.history and len(self.history["val_loss"]) > 0:
            plt.plot(
                self.history["val_loss"],
                label="val_loss",
                linestyle="--", 
            )

        # ---- per-label loss ----
        for lab in LABELS:
            tk = f"train_loss_{lab}"
            vk = f"val_loss_{lab}"

            if tk in self.history and len(self.history[tk]) > 0:
                plt.plot(
                    self.history[tk],
                    label=tk,
                    linestyle="-",
                )

            if vk in self.history and len(self.history[vk]) > 0:
                plt.plot(
                    self.history[vk],
                    label=vk,
                    linestyle="--",
                )

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()


def train_mode(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "round3", None):
        df_cases = load_cases_from_round3_csv(
            args.round3,
            volume_col="volume_id",
            label_col="final_consistency_relabel",
        )
    elif getattr(args, "vlm_image_jsonl", None):
        df_cases = load_cases_from_vlm_image_jsonl(args.vlm_image_jsonl)
    else:
        # CSV mode (original behavior)
        df_cases = pd.read_csv(args.exact_cases_csv)
        if args.csv_volume_col not in df_cases.columns or args.csv_label_col not in df_cases.columns:
            raise KeyError(
                f"exact_cases_csv missing required columns: {args.csv_volume_col}, {args.csv_label_col}. "
                f"Got: {list(df_cases.columns)}"
            )

        df_cases["volume_id_norm"] = df_cases[args.csv_volume_col].astype(str).map(strip_nii_suffix)
        df_cases["label_raw"] = df_cases[args.csv_label_col].map(parse_raw_label_set_from_cell)

    unknown_counter = Counter()
    keep_mask = []
    for s in df_cases["label_raw"].tolist():
        if len(s) == 0:
            keep_mask.append(False)
            continue
        unknown = set(s) - VALID
        if unknown:
            unknown_counter.update(list(unknown))
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    n_before = len(df_cases)
    df_cases = df_cases[np.array(keep_mask, dtype=bool)].copy()
    n_after = len(df_cases)
    n_dropped = n_before - n_after

    print(f"[INFO] dropped rows due to unknown labels or empty label_set: {n_dropped} / {n_before}")
    if unknown_counter:
        top = unknown_counter.most_common(20)
        print("[INFO] top unknown labels (dropped rows):")
        for k, v in top:
            print(f"  - {k}: {v}")

    df_cases["label_set"] = df_cases["label_raw"].map(lambda s: set(s))  
    df_cases = df_cases[df_cases["label_set"].map(len) > 0].copy()
    df_cases["patient_id"] = df_cases["volume_id_norm"].map(extract_patient_id)

    print(f"[INFO] cases rows (kept): {len(df_cases)}")
    print(f"[INFO] unique patients: {df_cases['patient_id'].nunique()}")

    df_emb = pd.read_parquet(args.embedding_index_parquet)
    need = {args.emb_volume_col, args.emb_path_col}
    missing = need - set(df_emb.columns)
    if missing:
        raise KeyError(f"embedding_index_parquet missing columns: {missing}, got={list(df_emb.columns)}")

    df_emb["volume_id_norm"] = df_emb[args.emb_volume_col].astype(str).map(strip_nii_suffix)

    # ---- join ----
    df = df_cases.merge(df_emb[["volume_id_norm", args.emb_path_col]], on="volume_id_norm", how="inner")
    df = df.drop_duplicates(subset=["volume_id_norm"]).copy()
    if len(df) == 0:
        raise RuntimeError("No matched rows after joining exact_cases_csv with embedding_index_parquet.")

    print(f"[INFO] joined rows (labels + embedding): {len(df)}")
    print(f"[INFO] joined unique patients: {df['patient_id'].nunique()}")

    # ---- patient-level stratified split ----
    split = patient_level_stratified_split(df, patient_col="patient_id", labelset_col="label_set", seed=args.seed)
    df["split"] = "train"
    df.loc[df["patient_id"].isin(split.val_patients), "split"] = "val"
    df.loc[df["patient_id"].isin(split.test_patients), "split"] = "test"

    manifest = df[["volume_id_norm", "patient_id", "split", args.emb_path_col]].copy()
    manifest["labels"] = df["label_set"].map(lambda s: ",".join([k for k in LABELS if k in s]))
    manifest.to_csv(out_dir / "split_manifest.csv", index=False)

    X = load_embeddings_from_paths(df[args.emb_path_col].tolist()).astype(np.float32)
    Y = multilabel_to_matrix(df["label_set"].tolist()).astype(np.float32)

    # split indices
    idx_train = df.index[df["split"] == "train"].to_numpy()
    idx_val = df.index[df["split"] == "val"].to_numpy()
    idx_test = df.index[df["split"] == "test"].to_numpy()

    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]
    X_test, Y_test = X[idx_test], Y[idx_test]

    # ---- StandardScaler (match sklearn pipeline behavior) ----
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # ---- Weighted BCE pos_weight ----
    pos_weight = compute_pos_weight(Y_train.astype(np.int64), args.pos_weight_power, args.pos_weight_cap)
    print("[INFO] pos_weight (Nneg/Npos) per class:")
    for lab, pw in zip(LABELS, pos_weight.tolist()):
        print(f"  - {lab}: {pw:.3f}")

    # ---- WeightedRandomSampler for balanced batches ----
    # sampler = make_weighted_sampler(Y_train.astype(np.int64))

    # ---- DataLoaders ----
    bs = args.batch_size
    train_ds = tud.TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(Y_train))
    val_ds = tud.TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(Y_val))
    test_ds = tud.TensorDataset(torch.from_numpy(X_test_s), torch.from_numpy(Y_test))

    # explode version
    # train_loader = tud.DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=args.num_workers)
    train_loader = tud.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=args.num_workers)

    val_loader = tud.DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=args.num_workers)
    test_loader = tud.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=args.num_workers)

    # ---- build Lightning model (hidden_sizes match old MLPClassifier) ----
    hidden_sizes = tuple(int(x.strip()) for x in args.hidden_sizes.split(",") if x.strip())
    in_dim = X_train_s.shape[1]
    lit = LitFiveIndependent(in_dim=in_dim, hidden_sizes=hidden_sizes, lr=args.lr, pos_weight_1d=pos_weight, weight_decay=args.weight_decay,)

    pl.seed_everything(args.seed, workers=True)
    csv_logger = pl.loggers.CSVLogger(save_dir=str(out_dir), name="lightning_logs")
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience,
        min_delta=args.min_delta,
    )

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best",
    )

    trainer = pl.Trainer(
        max_epochs=args.max_iter,
        accelerator="gpu",
        devices="1",
        logger=csv_logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        callbacks=[LossCurvePngCallback(out_dir), early_stop, checkpoint],
    )

    print(f"[INFO] Training Torch MLP: hidden_sizes={hidden_sizes}, max_epochs={args.max_iter}, batch_size={bs}, lr={args.lr}")
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ---- eval helper (threshold unchanged) ----
    @torch.no_grad()
    def predict_proba_numpy(model: LitFiveIndependent, Xs: np.ndarray) -> np.ndarray:
        model.eval()
        device = next(model.parameters()).device
        x = torch.from_numpy(Xs).to(device)

        probs = []
        for i in range(N_CLASSES):
            logits_i = model.mlps[i](x)
            probs.append(torch.sigmoid(logits_i))

        return torch.stack(probs, dim=1).detach().cpu().numpy()  # (n,5)


    def eval_split(name: str, Xs: np.ndarray, Ys: np.ndarray, model: LitFiveIndependent):
        P = predict_proba_numpy(model, Xs)
        pred = (P >= args.eval_threshold).astype(int)

        micro = f1_score(Ys, pred, average="micro", zero_division=0)
        macro = f1_score(Ys, pred, average="macro", zero_division=0)
        print(f"[{name}] f1_micro={micro:.4f} f1_macro={macro:.4f} (threshold={args.eval_threshold})")

        try:
            print(classification_report(Ys, pred, target_names=LABELS, zero_division=0))
        except Exception:
            pass

        return {"f1_micro": float(micro), "f1_macro": float(macro)}

    # also ensure last model is in eval
    lit.eval()
    print("\n========== EVAL: LAST (early-stopped / final weights) ==========")
    metrics_last = {
        "train": eval_split("train", X_train_s, Y_train.astype(int), lit),
        "val": eval_split("val", X_val_s, Y_val.astype(int), lit),
        "test": eval_split("test", X_test_s, Y_test.astype(int), lit),
    }

    best_path = checkpoint.best_model_path
    print(f"[INFO] best checkpoint: {best_path}")

    best_lit = LitFiveIndependent.load_from_checkpoint(
        best_path,
        in_dim=in_dim,
        hidden_sizes=hidden_sizes,
        lr=args.lr,
        pos_weight_1d=pos_weight, 
        weight_decay=args.weight_decay,
    )

    # move best model to same device as 'lit'
    device = next(lit.parameters()).device
    best_lit = best_lit.to(device)
    best_lit.eval()

    print("\n========== EVAL: BEST CHECKPOINT (min val_loss) ==========")
    metrics_best = {
        "train": eval_split("train", X_train_s, Y_train.astype(int), best_lit),
        "val": eval_split("val", X_val_s, Y_val.astype(int), best_lit),
        "test": eval_split("test", X_test_s, Y_test.astype(int), best_lit),
    }

    # ---- save model.joblib (keep filename/contract) ----
    model_pack = {
        "type": "torch_5indep_mlp_v1",
        "labels": LABELS,
        "hidden_sizes": list(hidden_sizes),
        "in_dim": int(in_dim),
        "state_dicts": [m.state_dict() for m in best_lit.mlps],
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "lr": float(args.lr),
        "pos_weight": pos_weight,
        "seed": int(args.seed),
        "best_ckpt_path": str(best_path),
        "best_val_loss": float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else None,
    }
    model_path = out_dir / "model.joblib"
    joblib.dump(model_pack, model_path)

    meta = {
        "labels": LABELS,
        "label_to_id": LABEL_TO_ID,
        "n_classes": N_CLASSES,
        "hidden_sizes": list(hidden_sizes),
        "max_iter": args.max_iter,
        "seed": args.seed,
        "eval_threshold": float(args.eval_threshold),
        "paths": {
            "round3": getattr(args, "round3", None),
            "exact_cases_csv": args.exact_cases_csv,
            "vlm_image_jsonl": args.vlm_image_jsonl,
            "embedding_index_parquet": args.embedding_index_parquet,
        },
        "counts": {
            "n_cases_rows_before_filter": int(n_before),
            "n_cases_rows_kept": int(len(df_cases)),
            "n_dropped_unknown_or_empty": int(n_dropped),
            "n_joined": int(len(df)),
            "n_patients": int(df["patient_id"].nunique()),
            "n_train": int((df["split"] == "train").sum()),
            "n_val": int((df["split"] == "val").sum()),
            "n_test": int((df["split"] == "test").sum()),
        },
        "metrics_last": metrics_last,
        "metrics_best": metrics_best,
        "best_checkpoint": {
            "path": str(best_path),
            "val_loss": float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else None,
        },
        "split_note": "Patient-level split via MultilabelStratifiedKFold on union labels per patient. fold0=test, fold1=val, others=train.",
        "unknown_labels_top20": unknown_counter.most_common(20),
    }
    with open(out_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved model: {model_path}")
    print(f"[OK] saved meta : {out_dir / 'model_meta.json'}")
    print(f"[OK] saved split: {out_dir / 'split_manifest.csv'}")


def summarize_prediction_label_sets(pred_csv_path: Path, out_dir: Path) -> None:
    """
    Read predictions.csv and summarize distribution of pred_labels as sorted label sets.
    Prints to stdout and writes to out_dir/label_set_stats.{csv,txt}.
    """
    import csv

    label_counter = Counter()
    total = 0

    with open(pred_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "pred_labels" not in reader.fieldnames:
            raise KeyError(f"predictions.csv missing column 'pred_labels'. columns={reader.fieldnames}")
        for row in reader:
            raw_label = (row.get("pred_labels") or "").strip()
            label_set = tuple(sorted(l.strip() for l in raw_label.split(",") if l.strip()))
            label_counter[label_set] += 1
            total += 1

    # print
    print(f"\n[INFO] Total predicted samples: {total}")
    print("[INFO] Label set statistics (percentage):")
    for label_set, count in label_counter.most_common():
        percent = (count * 100.0 / total) if total > 0 else 0.0
        label_name = ",".join(label_set) if label_set else "(empty)"
        print(f"{label_name:20s}  count={count:6d}  percent={percent:6.2f}%")

    # write txt
    out_txt = Path(out_dir) / "label_set_stats.txt"
    with open(out_txt, "w", encoding="utf-8") as w:
        w.write(f"Total samples: {total}\n\n")
        w.write("Label set statistics (percentage):\n")
        for label_set, count in label_counter.most_common():
            percent = (count * 100.0 / total) if total > 0 else 0.0
            label_name = ",".join(label_set) if label_set else "(empty)"
            w.write(f"{label_name:20s}  count={count:6d}  percent={percent:6.2f}%\n")

    # write csv (machine-friendly)
    out_csv = Path(out_dir) / "label_set_stats.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as w:
        writer = csv.writer(w)
        writer.writerow(["label_set", "count", "percent"])
        for label_set, count in label_counter.most_common():
            percent = (count * 100.0 / total) if total > 0 else 0.0
            label_name = ",".join(label_set) if label_set else ""
            writer.writerow([label_name, count, f"{percent:.6f}"])

    print(f"[OK] wrote label set stats: {out_txt}")
    print(f"[OK] wrote label set stats: {out_csv}\n")


@torch.no_grad()
def predict_mode(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pack = joblib.load(args.model)
    if not isinstance(pack, dict) or "state_dicts" not in pack:
        raise RuntimeError("Loaded --model is not a torch model pack created by this script (missing 'state_dicts').")

    # ---- self-check logs (optional but recommended) ----
    if "best_ckpt_path" in pack:
        print(f"[INFO] model pack best_ckpt_path: {pack['best_ckpt_path']}")
    if "best_val_loss" in pack:
        print(f"[INFO] model pack best_val_loss: {pack['best_val_loss']}")
    print(f"[INFO] model pack type: {pack.get('type', None)}")
    print(f"[INFO] model pack labels: {pack.get('labels', None)}")
    print(f"[INFO] model pack hidden_sizes: {pack.get('hidden_sizes', None)} in_dim: {pack.get('in_dim', None)}")
    # -----------------------------------------------

    labels = pack.get("labels", None)
    if labels != LABELS:
        raise RuntimeError(f"Model labels {labels} != script labels {LABELS}. Please use matching script/model.")

    hidden_sizes = tuple(pack["hidden_sizes"])
    in_dim = int(pack["in_dim"])
    scaler_mean = np.asarray(pack["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(pack["scaler_scale"], dtype=np.float64)
    state_dicts = pack["state_dicts"]
    if not isinstance(state_dicts, list) or len(state_dicts) != N_CLASSES:
        raise RuntimeError(f"'state_dicts' must be a list of length {N_CLASSES}, got {type(state_dicts)} len={len(state_dicts)}")

    # Build 5 independent MLPs
    mlps = nn.ModuleList([BinaryMLP(in_dim, hidden_sizes) for _ in range(N_CLASSES)])
    for i in range(N_CLASSES):
        mlps[i].load_state_dict(state_dicts[i])
        mlps[i].eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlps.to(device)

    # ---- load embedding index ----
    df_emb = pd.read_parquet(args.embedding_index_parquet)
    need = {args.emb_volume_col, args.emb_path_col}
    missing = need - set(df_emb.columns)
    if missing:
        raise KeyError(f"embedding_index_parquet missing columns: {missing}, got={list(df_emb.columns)}")

    df_emb["volume_id_norm"] = df_emb[args.emb_volume_col].astype(str).map(strip_nii_suffix)

    # choose query set
    if args.in_volume_ids_txt:
        vids = [
            strip_nii_suffix(x.strip())
            for x in Path(args.in_volume_ids_txt).read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        df_q = pd.DataFrame({"volume_id_norm": vids}).merge(
            df_emb[["volume_id_norm", args.emb_path_col]],
            on="volume_id_norm",
            how="left",
        )
    else:
        df_q = df_emb[["volume_id_norm", args.emb_path_col]].drop_duplicates("volume_id_norm").copy()

    # exclude manifest (same behavior as old script)
    if args.exclude_manifest_csv:
        df_ex = pd.read_csv(args.exclude_manifest_csv)
        if "volume_id_norm" not in df_ex.columns:
            raise KeyError(f"--exclude_manifest_csv must contain column 'volume_id_norm'. got={list(df_ex.columns)}")
        exclude = set(df_ex["volume_id_norm"].astype(str).map(strip_nii_suffix))
        n_before = len(df_q)
        df_q = df_q[~df_q["volume_id_norm"].isin(exclude)].copy()
        df_q.reset_index(drop=True, inplace=True)
        n_after = len(df_q)
        print(f"[INFO] excluded {n_before - n_after} volumes from manifest ({args.exclude_manifest_csv}); remaining={n_after}")

    # ----------------------------
    # Optional: filter by evaluation JSONL (only keep volume_id that appear in output.jsonl)
    # ----------------------------
    if getattr(args, "eval_jsonl", None):
        jsonl_path = Path(args.eval_jsonl)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"--eval_jsonl not found: {jsonl_path}")

        eval_ids: Set[str] = set()
        n_bad = 0
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    n_bad += 1
                    continue
                vid = obj.get(args.eval_jsonl_volume_key, None)
                if not vid:
                    continue
                eval_ids.add(strip_nii_suffix(str(vid)))

        n_before = len(df_q)
        df_q = df_q[df_q["volume_id_norm"].isin(eval_ids)].copy()
        df_q.reset_index(drop=True, inplace=True)
        n_after = len(df_q)

        print(
            f"[INFO] eval_jsonl filter: jsonl={jsonl_path} unique_ids={len(eval_ids)} "
            f"bad_lines={n_bad} kept={n_after}/{n_before} dropped={n_before-n_after}"
        )

    if args.limit is not None:
        df_q = df_q.head(args.limit).reset_index(drop=True)
        print(f"[INFO] limiting predict rows to {len(df_q)} (limit={args.limit})")

    # ---- load embeddings with progress bar ----
    ok_rows = []
    X_list = []
    for r in tqdm(df_q.itertuples(index=False), total=len(df_q), desc="Loading embeddings"):
        p = getattr(r, args.emb_path_col)
        if not isinstance(p, str) or not p:
            ok_rows.append(False)
            X_list.append(None)
            continue
        try:
            arr = np.load(p).astype(np.float32)
            ok_rows.append(True)
            X_list.append(arr)
        except Exception:
            ok_rows.append(False)
            X_list.append(None)

    ok_idx = [i for i, ok in enumerate(ok_rows) if ok]
    print(f"[INFO] predict rows: {len(df_q)}, ok embeddings: {len(ok_idx)}, missing/bad: {len(df_q)-len(ok_idx)}")
    if len(ok_idx) == 0:
        raise RuntimeError("No valid embeddings found to predict.")

    X_ok = np.stack([X_list[i] for i in ok_idx], axis=0)

    # ---- drop NaN/Inf (same behavior as old script) ----
    finite_mask = np.isfinite(X_ok).all(axis=1)
    n_drop_nan = int((~finite_mask).sum())

    if n_drop_nan > 0:
        drop_idx = [ok_idx[k] for k in np.where(~finite_mask)[0]]
        keep_idx = [ok_idx[k] for k in np.where(finite_mask)[0]]

        dropped = df_q.iloc[drop_idx][["volume_id_norm", args.emb_path_col]].copy()
        out_drop = out_dir / "dropped_nan_embeddings.csv"
        dropped.to_csv(out_drop, index=False)

        print(f"[WARN] embeddings contain NaN/Inf: dropped {n_drop_nan} / {len(ok_idx)} ok-rows. wrote: {out_drop}")

        ok_idx = keep_idx
        X_ok = X_ok[finite_mask]
    else:
        print("[INFO] embeddings are all finite (no NaN/Inf).")

    n_missing_bad = len(df_q) - len([i for i, ok in enumerate(ok_rows) if ok])
    print(
        f"[INFO] predict rows: {len(df_q)}, ok embeddings (loaded): {len([i for i, ok in enumerate(ok_rows) if ok])}, "
        f"missing/bad: {n_missing_bad}, dropped_nan: {n_drop_nan}, final_for_predict: {len(ok_idx)}"
    )
    if len(ok_idx) == 0:
        raise RuntimeError("No valid finite embeddings found to predict after dropping NaN/Inf rows.")

    # ---- StandardScaler transform (manual) ----
    X_ok_s = ((X_ok.astype(np.float64) - scaler_mean.reshape(1, -1)) / scaler_scale.reshape(1, -1)).astype(np.float32)

    # ---- forward with batching + progress bar ----
    batch_size = getattr(args, "predict_batch_size", None) or getattr(args, "batch_size", 256)
    P_ok = np.zeros((X_ok_s.shape[0], N_CLASSES), dtype=np.float32)

    n = X_ok_s.shape[0]
    for start in tqdm(range(0, n, batch_size), desc="Predicting", total=(n + batch_size - 1) // batch_size):
        end = min(start + batch_size, n)
        xb = torch.from_numpy(X_ok_s[start:end]).to(device)

        probs = []
        for i in range(N_CLASSES):
            logits_i = mlps[i](xb)                 # (b,)
            probs.append(torch.sigmoid(logits_i))  # (b,)

        pb = torch.stack(probs, dim=1).detach().cpu().numpy().astype(np.float32)  # (b,5)
        P_ok[start:end] = pb

    # ---- fill outputs (same column keywords as old script, minus other) ----
    for lab in LABELS:
        df_q[f"p_{lab}"] = np.nan
    df_q["pred_labels"] = ""
    df_q["pred_k"] = 0

    for out_row, i in enumerate(ok_idx):
        proba_row = P_ok[out_row]
        for j, lab in enumerate(LABELS):
            df_q.at[i, f"p_{lab}"] = float(proba_row[j])

        labs = matrix_to_label_list(
            proba_row=proba_row,
            threshold=args.threshold,
            min_k=args.min_k,
            max_k=args.max_k,
        )
        df_q.at[i, "pred_labels"] = ",".join(labs)
        df_q.at[i, "pred_k"] = int(len(labs))

    out_csv = out_dir / "predictions.csv"
    df_q.to_csv(out_csv, index=False)
    print(f"[OK] wrote predictions: {out_csv}")

    summarize_prediction_label_sets(out_csv, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "predict"], required=True)

    # common
    ap.add_argument("--out_dir", required=True)

    # data
    ap.add_argument("--embedding_index_parquet", required=True)
    ap.add_argument("--emb_volume_col", default="volume_id")
    ap.add_argument("--emb_path_col", default="embedding_abs")

    # train-only
    ap.add_argument("--exact_cases_csv", default=None)
    ap.add_argument("--csv_volume_col", default="volume_id")
    ap.add_argument("--csv_label_col", default="gt_top3_set")

    ap.add_argument("--hidden_sizes", default="512,256")
    ap.add_argument("--max_iter", type=int, default=200) 
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--weight_decay", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=0.0)

    ap.add_argument("--pos_weight_power", type=float, default=0.35) 
    ap.add_argument("--pos_weight_cap", type=float, default=90.0) 

    ap.add_argument(
        "--vlm_image_jsonl",
        default=None,
        help="Optional: JSONL path (e.g. output_image.jsonl). "
            "If provided, use llm labels as ground-truth instead of exact_cases_csv.",
    )

    ap.add_argument("--eval_threshold", type=float, default=0.5)

    ap.add_argument("--model", default=None)
    ap.add_argument("--in_volume_ids_txt", default=None, help="Optional: one volume_id per line. If omitted, predict for all embeddings in parquet.")
    ap.add_argument(
        "--exclude_manifest_csv",
        default=None,
        help="Optional: CSV with column volume_id_norm to exclude from prediction (e.g., train out_dir/split_manifest.csv)."
    )
    ap.add_argument("--predict_batch_size", type=int, default=None, help="Optional: batch size for predict forward pass.")
    ap.add_argument(
        "--eval_jsonl",
        default=None,
        help="Optional: JSONL path (e.g. Report_meta_match/output.jsonl). If provided, predict only on volume_id that appear in this file (after exclude_manifest_csv).",
    )
    ap.add_argument(
        "--eval_jsonl_volume_key",
        default="volume_id",
        help="JSONL key name for volume id. default=volume_id",
    )

    ap.add_argument(
        "--round3",
        default=None,
        help="Optional: round3_final_consistency.csv path. "
             "If provided, use final_consistency_relabel as ground-truth labels (replaces exact_cases_csv).",
    )

    ap.add_argument("--threshold", type=float, default=0.5, help="prob threshold for selecting labels")
    ap.add_argument("--min_k", type=int, default=1, help="minimum number of output labels")
    ap.add_argument("--max_k", type=int, default=6, help="maximum number of output labels") 

    ap.add_argument("--limit", type=int, default=None, help="Optional: limit number of rows for predict (e.g., 100 for smoke test)")

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)

    args = ap.parse_args()

    # ---- enforce mutual exclusivity among label sources (train) ----
    if args.mode == "train":
        sources = [("round3", args.round3), ("vlm_image_jsonl", args.vlm_image_jsonl), ("exact_cases_csv", args.exact_cases_csv)]
        used = [name for name, v in sources if v]
        if len(used) == 0:
            raise ValueError("--round3 or --vlm_image_jsonl or --exact_cases_csv is required for mode=train")
        if len(used) > 1:
            raise ValueError(f"Only one of --round3 / --vlm_image_jsonl / --exact_cases_csv can be provided. Got: {used}")

    if args.mode == "train":
        train_mode(args)
    else:
        if not args.model:
            raise ValueError("--model is required for mode=predict")
        predict_mode(args)


if __name__ == "__main__":
    main()
