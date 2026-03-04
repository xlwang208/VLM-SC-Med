#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from vlm_sc_med.consistency import enforce_study_level_consistency


def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(p: Path, items: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def mock_reconcile(it: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic mock reconciler for the toy example:
    - If seg says 'lung' present -> chest
    - If seg says 'liver' present -> abdomen
    - If meta says pelvis -> pelvis
    Else other
    """
    seg = (it.get("seg_evidence") or {})
    meta = (it.get("meta_evidence") or {})
    organs = set((seg.get("organs_present") or []))
    if "lung" in organs or "heart" in organs:
        return {"label": "chest", "confidence": "high", "rationale": "seg shows thoracic organs", "overrides": []}
    if "liver" in organs or "kidney" in organs:
        return {"label": "abdomen", "confidence": "high", "rationale": "seg shows abdominal organs", "overrides": []}
    if str(meta.get("normalized_bodypart","")).lower() == "pelvis":
        return {"label": "pelvis", "confidence": "medium", "rationale": "metadata indicates pelvis", "overrides": ["used metadata (no physical evidence)"]}
    return {"label": "other", "confidence": "low", "rationale": "insufficient evidence", "overrides": []}


def main() -> None:
    ap = argparse.ArgumentParser("Run the pipeline with a mock reconciler (no LLM required).")
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--no-consistency", action="store_true")
    args = ap.parse_args()

    items = read_jsonl(Path(args.in_jsonl))
    out = []
    for it in items:
        obj = mock_reconcile(it)
        it2 = dict(it)
        it2["curated_label"] = obj["label"]
        it2["curated_confidence"] = obj["confidence"]
        it2["curated_rationale"] = obj["rationale"]
        it2["curated_overrides"] = obj["overrides"]
        out.append(it2)

    if not args.no_consistency:
        out = enforce_study_level_consistency(out)

    write_jsonl(Path(args.out_jsonl), out)


if __name__ == "__main__":
    main()
