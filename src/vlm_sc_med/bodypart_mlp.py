from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# sklearn is used for metrics / scaling only (no private assets needed)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

# Optional: torch is only required if you actually want to train the MLP.
try:
    import torch
    import torch.nn as nn
    import torch.utils.data as tud
except Exception as e:  # pragma: no cover
    torch = None
    nn = None
    tud = None


CLASSES: List[str] = ["head", "neck", "chest", "abdomen", "pelvis"]


def strip_nii_suffix(volume_id: str) -> str:
    s = (volume_id or "").strip()
    if s.endswith(".nii.gz"):
        return s[:-7]
    if s.endswith(".nii"):
        return s[:-4]
    return s


def parse_label_set(x: Any) -> Set[str]:
    """Parse a label field into a set of labels within CLASSES.

    Accepts:
      - list like ["abdomen", "pelvis"]
      - string like "abdomen,pelvis" (also supports ; | / whitespace)
      - None/NaN -> empty set
    """
    if x is None:
        return set()
    try:
        import math
        if isinstance(x, float) and math.isnan(x):
            return set()
    except Exception:
        pass

    if isinstance(x, (list, tuple, set)):
        items = [str(t).strip().lower() for t in x]
    else:
        s = str(x).strip()
        if not s or s.lower() == "none":
            return set()
        for sep in [";", "|", "/", "\t"]:
            s = s.replace(sep, ",")
        s = ",".join([t for t in s.replace(" ", ",").split(",") if t != ""])
        items = [t.strip().lower() for t in s.split(",") if t.strip()]

    return set([t for t in items if t in set(CLASSES)])


def load_embeddings_from_npy_paths(paths: Sequence[str]) -> np.ndarray:
    """Load per-case embeddings from .npy paths and stack to (N, D).

    NOTE: Embeddings are NOT included in the anonymous release.
    You should generate/provide them yourself and point this function to those paths.
    """
    X: List[np.ndarray] = []
    for p in tqdm(list(paths), desc="Loading embeddings (.npy)"):
        arr = np.load(p)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        X.append(arr.astype(np.float32))
    return np.stack(X, axis=0)


def read_cases_csv(
    csv_path: str,
    volume_col: str = "volume_id",
    label_col: str = "labels",
    embedding_path_col: str = "embedding_path",
) -> pd.DataFrame:
    """Read a case table that links volume_id, labels, and embedding paths.

    Required columns:
      - volume_id (string)
      - labels (string/list)
      - embedding_path (path to .npy embedding)

    This function does NOT assume any dataset-specific schema beyond these.
    """
    df = pd.read_csv(csv_path)
    for c in [volume_col, label_col, embedding_path_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}'. Got columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["volume_id_norm"] = df[volume_col].astype(str).map(strip_nii_suffix)
    out["label_set"] = df[label_col].apply(parse_label_set)
    out["embedding_path"] = df[embedding_path_col].astype(str)
    return out


def multilabel_to_binary_matrix(label_sets: Sequence[Set[str]]) -> np.ndarray:
    """Convert list[set[str]] to a binary matrix (N, C)."""
    y = np.zeros((len(label_sets), len(CLASSES)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(CLASSES)}
    for r, s in enumerate(label_sets):
        for c in s:
            if c in idx:
                y[r, idx[c]] = 1.0
    return y


class FiveClassMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(CLASSES)),
        )

    def forward(self, x):
        return self.net(x)


class EmbeddingDataset(tud.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    hidden_dim: int = 512
    dropout: float = 0.1
    threshold: float = 0.5


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    device: Optional[str] = None,
) -> Tuple[FiveClassMLP, StandardScaler]:
    """Train a simple multilabel MLP on provided embeddings.

    This is a *reference* implementation to reproduce the evaluation pipeline.
    Exact training details in the original project may differ.
    """
    if torch is None:
        raise RuntimeError("torch is required to train the MLP, but it's not installed.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = FiveClassMLP(in_dim=Xs.shape[1], hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    ds = EmbeddingDataset(Xs, y)
    dl = tud.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    return model.eval(), scaler


def predict_proba(model: FiveClassMLP, scaler: StandardScaler, X: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required for model inference, but it's not installed.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        xb = torch.from_numpy(Xs).to(device)
        logits = model(xb).cpu().numpy()
    # sigmoid
    return 1.0 / (1.0 + np.exp(-logits))


def eval_multilabel(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    out: Dict[str, Any] = {}
    out["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["per_class_report"] = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return out


def main():
    ap = argparse.ArgumentParser(description="Train a 5-label MLP on user-provided embeddings (anonymous release).")
    ap.add_argument("--cases-csv", required=True, help="CSV with columns: volume_id, labels, embedding_path.")
    ap.add_argument("--out-model", required=True, help="Output path (joblib) to save scaler+state_dict+meta.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        threshold=args.threshold,
    )

    df = read_cases_csv(args.cases_csv)
    X = load_embeddings_from_npy_paths(df["embedding_path"].tolist())
    y = multilabel_to_binary_matrix(df["label_set"].tolist())

    model, scaler = train_mlp(X, y, cfg)

    # save as a light-weight artifact (no private data)
    import joblib
    payload = {
        "classes": CLASSES,
        "cfg": cfg.__dict__,
        "scaler": scaler,
        "state_dict": model.state_dict(),
        "in_dim": int(X.shape[1]),
    }
    joblib.dump(payload, args.out_model)

    y_prob = predict_proba(model, scaler, X)
    metrics = eval_multilabel(y, y_prob, threshold=cfg.threshold)
    print(json.dumps({k: v for k, v in metrics.items() if k in ["micro_f1", "macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
