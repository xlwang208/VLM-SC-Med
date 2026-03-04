from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


CLASSES = ["head", "neck", "chest", "abdomen", "pelvis"]
PROB_COLS = ["p_head", "p_neck", "p_chest", "p_abdomen", "p_pelvis"]


def norm_vid(vid: str) -> str:
    vid = (vid or "").strip()
    if vid.endswith(".nii.gz"):
        vid = vid[:-7]
    elif vid.endswith(".nii"):
        vid = vid[:-4]
    return vid


def parse_label_set(x: Any) -> Set[str]:
    """Parse a label field into a set of labels within CLASSES."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return set()
    s = str(x).strip()
    if not s or s.lower() == "none":
        return set()

    for sep in [";", "|", "/", "\t"]:
        s = s.replace(sep, ",")
    s = ",".join([t for t in s.replace(" ", ",").split(",") if t != ""])
    items = [t.strip().lower() for t in s.split(",") if t.strip()]
    items = [t for t in items if t in set(CLASSES)]
    return set(items)


def load_manual_gt(path: str, id_col: str = "volume_id", label_col: str = "manual_labels") -> pd.DataFrame:
    df = pd.read_csv(path)
    if id_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"manual GT csv must have columns: {id_col}, {label_col}")
    out = pd.DataFrame()
    out["volume_id_norm"] = df[id_col].astype(str).map(norm_vid)
    out["gt_set"] = df[label_col].apply(parse_label_set)
    return out


def load_predictions(path: str) -> pd.DataFrame:
    """Load prediction table.

    Supported formats:
      - CSV with columns: volume_id + either PROB_COLS or pred_labels
      - JSONL with per-line dict containing volume_id and either PROB_COLS or pred_labels

    If probabilities are provided, pred_labels are derived by threshold 0.5 by default.
    """
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(p)

    if "volume_id" not in df.columns:
        raise KeyError("prediction file must contain column 'volume_id'")
    df["volume_id_norm"] = df["volume_id"].astype(str).map(norm_vid)

    # If probabilities present, derive pred_set
    has_probs = all(c in df.columns for c in PROB_COLS)
    if has_probs:
        prob = df[PROB_COLS].to_numpy(dtype=float)
        pred = (prob >= 0.5).astype(int)
        pred_set = []
        for r in range(pred.shape[0]):
            labs = [CLASSES[i] for i in range(len(CLASSES)) if pred[r, i] == 1]
            pred_set.append(set(labs))
        df["pred_set"] = pred_set
        return df

    if "pred_labels" in df.columns:
        df["pred_set"] = df["pred_labels"].apply(parse_label_set)
        return df

    raise KeyError(f"prediction file must contain either {PROB_COLS} or 'pred_labels'")


def to_binary_matrix(label_sets: List[Set[str]]) -> np.ndarray:
    y = np.zeros((len(label_sets), len(CLASSES)), dtype=int)
    idx = {c: i for i, c in enumerate(CLASSES)}
    for r, s in enumerate(label_sets):
        for c in s:
            if c in idx:
                y[r, idx[c]] = 1
    return y


def main():
    ap = argparse.ArgumentParser(description="Evaluate predictions against manual GT (anonymous release).")
    ap.add_argument("--pred", required=True, help="Prediction CSV/JSONL.")
    ap.add_argument("--manual-gt", required=True, help="Manual GT CSV.")
    ap.add_argument("--id-col", default="volume_id", help="ID column in manual GT.")
    ap.add_argument("--label-col", default="manual_labels", help="Label column in manual GT.")
    args = ap.parse_args()

    gt = load_manual_gt(args.manual_gt, id_col=args.id_col, label_col=args.label_col)
    pr = load_predictions(args.pred)

    merged = gt.merge(pr[["volume_id_norm", "pred_set"]], on="volume_id_norm", how="inner")
    if len(merged) == 0:
        raise RuntimeError("No overlapping volume_id between pred and manual GT.")

    y_true = to_binary_matrix(merged["gt_set"].tolist())
    y_pred = to_binary_matrix(merged["pred_set"].tolist())

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="micro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="micro", zero_division=0)

    print(json.dumps({
        "n": int(len(merged)),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "micro_precision": float(prec),
        "micro_recall": float(rec),
        "per_class": classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0),
    }, indent=2))


if __name__ == "__main__":
    main()
