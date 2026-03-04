from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ANATOMY_ORDER = ["head", "neck", "chest", "abdomen", "pelvis", "other"]
ANATOMY_SET = set(ANATOMY_ORDER)


def extract_patient_id(volume_id: str) -> str:
    """
    Heuristic: split a series identifier/path and pick a stable patient token.
    You SHOULD adapt this to your dataset.
    """
    parts = str(volume_id).split("/")
    if len(parts) >= 3:
        return parts[2]
    return str(volume_id)


def extract_study_date(volume_id: str) -> str:
    """
    Heuristic: parse YYYYMMDD[HHMMSS] token from the series identifier.
    You SHOULD adapt this to your dataset.
    """
    base = str(volume_id).split("/")[-1]
    toks = base.split("_")
    if len(toks) >= 2:
        cand = toks[-2]
        if re.fullmatch(r"\d{8,14}", cand):
            return cand
    return ""


@dataclass
class ConsistencyResult:
    final_label: str
    is_outlier: bool
    reason: str


def enforce_study_level_consistency(items: List[Dict]) -> List[Dict]:
    """
    Given a list of series dicts (same schema as the pipeline output),
    add:
      - patient_id
      - study_date
      - intra_study_outlier
      - outlier_reason
      - final_consistency_label (may override curated label)

    Logic:
      - group by (patient_id, study_date)
      - if a label is disjoint from the majority set, mark as outlier
      - otherwise keep label

    This module is intentionally conservative and dataset-agnostic.
    """
    # precompute ids
    for it in items:
        vid = it.get("volume_id", "")
        it["patient_id"] = extract_patient_id(vid)
        it["study_date"] = extract_study_date(vid)

    # group
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for it in items:
        key = (it["patient_id"], it["study_date"])
        groups.setdefault(key, []).append(it)

    for key, g in groups.items():
        if len(g) < 2:
            for it in g:
                it["intra_study_outlier"] = False
                it["outlier_reason"] = ""
                it["final_consistency_label"] = it.get("curated_label", it.get("label", "other"))
            continue

        labels = [str(it.get("curated_label", it.get("label", "other"))).lower() for it in g]
        cnt = Counter(labels)
        majority, maj_n = cnt.most_common(1)[0]

        for it in g:
            lab = str(it.get("curated_label", it.get("label", "other"))).lower()
            it["intra_study_outlier"] = (lab != majority and maj_n >= 2)
            if it["intra_study_outlier"]:
                it["outlier_reason"] = f"label '{lab}' differs from study majority '{majority}'"
                it["final_consistency_label"] = majority
            else:
                it["outlier_reason"] = ""
                it["final_consistency_label"] = lab

    return items
