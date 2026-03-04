import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


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


def parse_bpreg_tag(tag: Any) -> Set[str]:
    if tag is None:
        return set()
    s = str(tag).strip()
    if not s or s.lower() == "none":
        return set()

    for sep in [",", ";", "|", "/", " "]:
        s = s.replace(sep, "-")
    parts = [p.strip().lower() for p in s.split("-") if p.strip()]
    parts = [p for p in parts if p in set(CLASSES)]
    return set(parts)


def parse_list_or_str_labels(x: Any) -> Set[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return set()

    if isinstance(x, (list, tuple, set)):
        items = []
        for t in x:
            if t is None:
                continue
            s = str(t).strip().lower()
            if s:
                items.append(s)
        return set([t for t in items if t in set(CLASSES)])

    return parse_label_set(x)


def match_type(gt: Set[str], pred: Set[str]) -> str:
    if gt == pred:
        return "exact"
    if pred and gt and pred.issubset(gt):
        return "underpredict"
    if gt.intersection(pred):
        return "overlap"
    return "disjoint"


def per_class_counts(sets: List[Set[str]]) -> Dict[str, int]:
    d = {c: 0 for c in CLASSES}
    for s in sets:
        for c in s:
            d[c] += 1
    return d


def _compute_metrics_from_sets(gt_sets: List[Set[str]], pred_sets: List[Set[str]]) -> Dict[str, float]:
    mlb = MultiLabelBinarizer(classes=CLASSES)
    Y_true = mlb.fit_transform([sorted(s) for s in gt_sets])
    Y_pred = mlb.transform([sorted(s) for s in pred_sets])

    # per-class f1 (order matches CLASSES)
    f1_per_class = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    out = {f"f1_{c}": float(v) for c, v in zip(CLASSES, f1_per_class)}

    out["f1_macro"] = float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))
    out["precision_macro"] = float(precision_score(Y_true, Y_pred, average="macro", zero_division=0))
    out["recall_macro"] = float(recall_score(Y_true, Y_pred, average="macro", zero_division=0))
    return out


def _bootstrap_metrics(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    n = len(gt_sets)
    rng = np.random.default_rng(seed)

    keys = list(_compute_metrics_from_sets(gt_sets, pred_sets).keys())
    samples = {k: [] for k in keys}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n, endpoint=False)
        g = [gt_sets[i] for i in idx]
        p = [pred_sets[i] for i in idx]
        m = _compute_metrics_from_sets(g, p)
        for k in keys:
            samples[k].append(m[k])

    out = {}
    for k in keys:
        arr = np.asarray(samples[k], dtype=float)
        out[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }
    return out


def evaluate_sets(
    df: pd.DataFrame,
    id_col: str,
    gt_col: str,
    pred_col: str,
    outdir: Path,
    prefix: str,
    bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)

    gt_counts = per_class_counts(df[gt_col].tolist())
    pred_counts = per_class_counts(df[pred_col].tolist())
    mt = df["match_type"].value_counts().to_dict()

    gt_sets = df[gt_col].tolist()
    pred_sets = df[pred_col].tolist()

    point_metrics = _compute_metrics_from_sets(gt_sets, pred_sets)

    mlb = MultiLabelBinarizer(classes=CLASSES)
    Y_true = mlb.fit_transform([sorted(s) for s in gt_sets])
    Y_pred = mlb.transform([sorted(s) for s in pred_sets])

    report = classification_report(Y_true, Y_pred, target_names=CLASSES, digits=4, zero_division=0)
    f1_macro = float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))
    f1_micro = float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))
    f1_weighted = float(f1_score(Y_true, Y_pred, average="weighted", zero_division=0))

    bootstrap_metrics: Optional[Dict[str, Dict[str, float]]] = None
    if bootstrap and bootstrap > 0:
        bootstrap_metrics = _bootstrap_metrics(
            gt_sets=gt_sets,
            pred_sets=pred_sets,
            n_bootstrap=int(bootstrap),
            seed=int(bootstrap_seed),
        )

    lines = []
    lines.append(f"Rows evaluated: {len(df)}")
    lines.append("")
    lines.append("Per-class sample counts (membership-based):")
    lines.append("  GT:   " + " | ".join([f"{c}={gt_counts[c]}" for c in CLASSES]))
    lines.append("  Pred: " + " | ".join([f"{c}={pred_counts[c]}" for c in CLASSES]))
    lines.append("")
    lines.append("Per-volume GT vs Pred set comparison:")
    for k in ["exact", "underpredict", "overlap", "disjoint"]:
        lines.append(f"  {k}: {mt.get(k, 0)}")
    lines.append("")
    lines.append("F1 (point estimate on full set):")
    lines.append(f"  macro:    {f1_macro}")
    lines.append(f"  micro:    {f1_micro}")
    lines.append(f"  weighted: {f1_weighted}")
    lines.append("")
    lines.append("Table metrics (point):")
    for k in ["f1_head", "f1_neck", "f1_chest", "f1_abdomen", "f1_pelvis", "f1_macro", "precision_macro", "recall_macro"]:
        lines.append(f"  {k}: {point_metrics[k]}")
    lines.append("")

    if bootstrap_metrics is not None:
        lines.append(f"Bootstrap: n={bootstrap} (row-wise), seed={bootstrap_seed}")
        lines.append("Table metrics (bootstrap mean ± std):")
        for k in ["f1_head", "f1_neck", "f1_chest", "f1_abdomen", "f1_pelvis", "f1_macro", "precision_macro", "recall_macro"]:
            mu = bootstrap_metrics[k]["mean"]
            sd = bootstrap_metrics[k]["std"]
            lines.append(f"  {k}: {mu} ± {sd}")
        lines.append("")

    lines.append("==== classification_report ====")
    lines.append(report)
    lines.append("")

    (outdir / f"analysis_report_{prefix}.txt").write_text("\n".join(lines), encoding="utf-8")

    out_cols = [
        id_col,
        "gt_labels_joined",
        "pred_labels_joined",
        "match_type",
        "gt_size",
        "pred_size",
        "intersection",
    ]
    for c in PROB_COLS:
        if c in df.columns and c not in out_cols:
            out_cols.append(c)
    for c in ["pred_labels", "body_part_examined_tag"]:
        if c in df.columns and c not in out_cols:
            out_cols.append(c)

    df[out_cols].to_csv(outdir / f"result_{prefix}.csv", index=False)

    metrics_payload: Dict[str, Any] = {
        "rows_evaluated": int(len(df)),
        "point": point_metrics,
    }
    if bootstrap_metrics is not None:
        metrics_payload["bootstrap"] = {
            "n": int(bootstrap),
            "seed": int(bootstrap_seed),
            "mean_std": bootstrap_metrics,
        }

    (outdir / f"metrics_{prefix}.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metrics_payload


def load_manual(manual_csv: Path, id_col: str, label_col: str) -> pd.DataFrame:
    manual_df = pd.read_csv(manual_csv)

    s = manual_df[label_col].astype(str).replace("nan", "").str.strip()
    manual_df["_label_str"] = s
    manual_valid = manual_df[manual_df["_label_str"] != ""].copy()

    manual_valid["volume_id_norm"] = manual_valid[id_col].astype(str).str.strip()
    manual_valid["gt_set"] = manual_valid[label_col].apply(parse_label_set)
    return manual_valid


def eval_pred_csv(
    manual_valid: pd.DataFrame,
    pred_csv: Path,
    outdir: Path,
    bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    pred_df = pd.read_csv(pred_csv)

    merged = manual_valid.merge(
        pred_df,
        left_on="volume_id_norm",
        right_on="volume_id_norm",
        how="inner",
        suffixes=("_gt", "_pred"),
    ).copy()

    merged["pred_set"] = merged["pred_labels"].apply(parse_label_set)

    merged["match_type"] = merged.apply(lambda r: match_type(r["gt_set"], r["pred_set"]), axis=1)
    merged["gt_labels_joined"] = merged["gt_set"].apply(lambda s: ",".join(sorted(s)))
    merged["pred_labels_joined"] = merged["pred_set"].apply(lambda s: ",".join(sorted(s)))
    merged["gt_size"] = merged["gt_set"].apply(len)
    merged["pred_size"] = merged["pred_set"].apply(len)
    merged["intersection"] = merged.apply(lambda r: ",".join(sorted(r["gt_set"].intersection(r["pred_set"]))), axis=1)

    _ = evaluate_sets(
        df=merged,
        id_col="volume_id_norm",
        gt_col="gt_set",
        pred_col="pred_set",
        outdir=outdir,
        prefix="pred",
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
    )

    return {
        "manual_labeled": len(manual_valid),
        "pred_rows": len(pred_df),
        "matched": len(merged),
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def eval_dkfz_bpreg(
    manual_valid: pd.DataFrame,
    dkfz_bpreg: Path,
    outdir: Path,
    bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    rows = read_jsonl(dkfz_bpreg)
    bp_df = pd.DataFrame(rows)

    total_rows = len(bp_df)

    bp_df["volume_id_norm"] = bp_df["volume_id"].apply(norm_vid)
    manual_ids = set(manual_valid["volume_id_norm"].tolist())

    bp_matched = bp_df[bp_df["volume_id_norm"].isin(manual_ids)].copy()
    matched_to_manual_total = len(bp_matched)

    manual_labeled_matched_ids_count = len(set(bp_matched["volume_id_norm"].tolist()))
    manual_labeled_unmatched_ids_count = len(manual_ids) - manual_labeled_matched_ids_count

    if "body_part_examined_tag" not in bp_matched.columns:
        bp_matched["body_part_examined_tag"] = None

    bp_matched["pred_set"] = bp_matched["body_part_examined_tag"].apply(parse_bpreg_tag)
    matched_empty_or_none = int(bp_matched["pred_set"].apply(lambda s: len(s) == 0).sum())

    merged = manual_valid.merge(
        bp_matched[["volume_id_norm", "body_part_examined_tag", "pred_set"]],
        on="volume_id_norm",
        how="inner",
    ).copy()

    merged["match_type"] = merged.apply(lambda r: match_type(r["gt_set"], r["pred_set"]), axis=1)
    merged["gt_labels_joined"] = merged["gt_set"].apply(lambda s: ",".join(sorted(s)))
    merged["pred_labels_joined"] = merged["pred_set"].apply(lambda s: ",".join(sorted(s)))
    merged["gt_size"] = merged["gt_set"].apply(len)
    merged["pred_size"] = merged["pred_set"].apply(len)
    merged["intersection"] = merged.apply(
        lambda r: ",".join(sorted(r["gt_set"].intersection(r["pred_set"]))), axis=1
    )

    _ = evaluate_sets(
        df=merged,
        id_col="volume_id_norm",
        gt_col="gt_set",
        pred_col="pred_set",
        outdir=outdir,
        prefix="dkfz",
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
    )

    return {
        "dkfz_rows_total_read": total_rows,
        "manual_labeled": len(manual_valid),
        "matched_to_manual_total_rows": matched_to_manual_total,
        "manual_labeled_matched_unique_ids": manual_labeled_matched_ids_count,
        "manual_labeled_unmatched_unique_ids": manual_labeled_unmatched_ids_count,
        "matched_empty_or_none_counted_as_empty_pred": matched_empty_or_none,
        "evaluated_rows": len(merged),
    }


def eval_report_meta_match(
    manual_valid: pd.DataFrame,
    report_jsonl: Path,
    outdir: Path,
    bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    rows = read_jsonl(report_jsonl)
    df = pd.DataFrame(rows)
    total_rows = len(df)

    df["volume_id_norm"] = df["volume_id"].apply(norm_vid)

    manual_ids = set(manual_valid["volume_id_norm"].tolist())
    matched = df[df["volume_id_norm"].isin(manual_ids)].copy()
    matched_to_manual_total = len(matched)

    manual_labeled_matched_unique_ids_count = len(set(matched["volume_id_norm"].tolist()))
    manual_labeled_unmatched_ids_count = len(manual_ids) - manual_labeled_matched_unique_ids_count

    if "top3" not in matched.columns:
        matched["top3"] = None

    def _top3_to_set(x: Any) -> Set[str]:
        if isinstance(x, list):
            x = x[:3]
        return parse_list_or_str_labels(x)

    matched["pred_set_report"] = matched["top3"].apply(_top3_to_set)
    matched_empty_report = int(matched["pred_set_report"].apply(lambda s: len(s) == 0).sum())

    merged_report = manual_valid.merge(
        matched[["volume_id_norm", "top3", "pred_set_report"]],
        on="volume_id_norm",
        how="inner",
    ).copy()

    merged_report["pred_set"] = merged_report["pred_set_report"]
    merged_report["match_type"] = merged_report.apply(lambda r: match_type(r["gt_set"], r["pred_set"]), axis=1)
    merged_report["gt_labels_joined"] = merged_report["gt_set"].apply(lambda s: ",".join(sorted(s)))
    merged_report["pred_labels_joined"] = merged_report["pred_set"].apply(lambda s: ",".join(sorted(s)))
    merged_report["gt_size"] = merged_report["gt_set"].apply(len)
    merged_report["pred_size"] = merged_report["pred_set"].apply(len)
    merged_report["intersection"] = merged_report.apply(
        lambda r: ",".join(sorted(r["gt_set"].intersection(r["pred_set"]))), axis=1
    )

    _ = evaluate_sets(
        df=merged_report,
        id_col="volume_id_norm",
        gt_col="gt_set",
        pred_col="pred_set",
        outdir=outdir,
        prefix="report",
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
    )

    def _get_norm_cands(md: Any) -> Any:
        if not isinstance(md, dict):
            return None
        return md.get("normalized_candidates", None)

    if "match_detail" not in matched.columns:
        matched["match_detail"] = None
    matched["normalized_candidates"] = matched["match_detail"].apply(_get_norm_cands)

    matched["pred_set_meta"] = matched["normalized_candidates"].apply(parse_list_or_str_labels)
    matched_empty_meta = int(matched["pred_set_meta"].apply(lambda s: len(s) == 0).sum())

    merged_meta = manual_valid.merge(
        matched[["volume_id_norm", "normalized_candidates", "pred_set_meta"]],
        on="volume_id_norm",
        how="inner",
    ).copy()

    merged_meta["pred_set"] = merged_meta["pred_set_meta"]
    merged_meta["match_type"] = merged_meta.apply(lambda r: match_type(r["gt_set"], r["pred_set"]), axis=1)
    merged_meta["gt_labels_joined"] = merged_meta["gt_set"].apply(lambda s: ",".join(sorted(s)))
    merged_meta["pred_labels_joined"] = merged_meta["pred_set"].apply(lambda s: ",".join(sorted(s)))
    merged_meta["gt_size"] = merged_meta["gt_set"].apply(len)
    merged_meta["pred_size"] = merged_meta["pred_set"].apply(len)
    merged_meta["intersection"] = merged_meta.apply(
        lambda r: ",".join(sorted(r["gt_set"].intersection(r["pred_set"]))), axis=1
    )

    _ = evaluate_sets(
        df=merged_meta,
        id_col="volume_id_norm",
        gt_col="gt_set",
        pred_col="pred_set",
        outdir=outdir,
        prefix="meta",
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
    )

    return {
        "report_meta_rows_total_read": total_rows,
        "manual_labeled": len(manual_valid),
        "matched_to_manual_total_rows": matched_to_manual_total,
        "manual_labeled_matched_unique_ids": manual_labeled_matched_unique_ids_count,
        "manual_labeled_unmatched_unique_ids": manual_labeled_unmatched_ids_count,
        "matched_empty_report_counted_as_empty_pred": matched_empty_report,
        "matched_empty_meta_counted_as_empty_pred": matched_empty_meta,
        "evaluated_rows_report": len(merged_report),
        "evaluated_rows_meta": len(merged_meta),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_csv", required=True, help="Path to manuel_label_expert.csv")
    parser.add_argument("--pred_csv", required=True, help="Path to predictions.csv")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--dkfz_bpreg", default=None, help="Path to bpreg_results_existing.jsonl (DKFZ BodyPartRegression)")
    parser.add_argument(
        "--report_meta_match",
        default=None,
        help="Path to Report_meta_match/output.jsonl (contains top3 and match_detail.normalized_candidates)",
    )
    parser.add_argument("--id_col_manual", default="volume_id")
    parser.add_argument("--label_col_manual", default="label")

    # NEW: control std via bootstrap
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="If >0, run row-wise bootstrap with N resamples and report mean±std. Default 0 = no std.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
        help="Random seed for bootstrap (only used when --bootstrap > 0).",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manual_valid = load_manual(Path(args.manual_csv), args.id_col_manual, args.label_col_manual)

    pred_stats = eval_pred_csv(
        manual_valid,
        Path(args.pred_csv),
        outdir,
        bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )

    dkfz_stats = None
    if args.dkfz_bpreg:
        dkfz_stats = eval_dkfz_bpreg(
            manual_valid,
            Path(args.dkfz_bpreg),
            outdir,
            bootstrap=args.bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )

    report_meta_stats = None
    if args.report_meta_match:
        report_meta_stats = eval_report_meta_match(
            manual_valid,
            Path(args.report_meta_match),
            outdir,
            bootstrap=args.bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )

    lines = []
    lines.append("==== SUMMARY ====")
    lines.append("")
    lines.append(f"Bootstrap: {'OFF' if args.bootstrap <= 0 else f'ON (n={args.bootstrap}, seed={args.bootstrap_seed})'}")
    lines.append("")

    lines.append("[Pred CSV]")
    for k, v in pred_stats.items():
        lines.append(f"{k}: {v}")
    lines.append(f"analysis_report: {outdir / 'analysis_report_pred.txt'}")
    lines.append(f"metrics_json:    {outdir / 'metrics_pred.json'}")
    lines.append(f"result_csv:      {outdir / 'result_pred.csv'}")
    lines.append("")

    if dkfz_stats is not None:
        lines.append("[DKFZ BPREG]")
        for k, v in dkfz_stats.items():
            lines.append(f"{k}: {v}")
        lines.append(f"analysis_report: {outdir / 'analysis_report_dkfz.txt'}")
        lines.append(f"metrics_json:    {outdir / 'metrics_dkfz.json'}")
        lines.append(f"result_csv:      {outdir / 'result_dkfz.csv'}")
        lines.append("")

    if report_meta_stats is not None:
        lines.append("[REPORT_META_MATCH]")
        for k, v in report_meta_stats.items():
            lines.append(f"{k}: {v}")
        lines.append(f"analysis_report_report: {outdir / 'analysis_report_report.txt'}")
        lines.append(f"metrics_report_json:    {outdir / 'metrics_report.json'}")
        lines.append(f"result_report_csv:      {outdir / 'result_report.csv'}")
        lines.append(f"analysis_report_meta:   {outdir / 'analysis_report_meta.txt'}")
        lines.append(f"metrics_meta_json:      {outdir / 'metrics_meta.json'}")
        lines.append(f"result_meta_csv:        {outdir / 'result_meta.csv'}")
        lines.append("")

    (outdir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
