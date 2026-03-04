#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from vlm_sc_med.llm_client import OpenAICompatClient
from vlm_sc_med.reconcile import Reconciler
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


def main() -> None:
    ap = argparse.ArgumentParser("Run the agentic semantic curation pipeline (LLM reconciliation + optional consistency).")
    ap.add_argument("--in-jsonl", required=True, help="Input JSONL. Each line is one series with evidence fields.")
    ap.add_argument("--out-jsonl", required=True, help="Output JSONL with curated labels + debug.")
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base_url, e.g. http://127.0.0.1:8001/v1")
    ap.add_argument("--api-key", default="EMPTY", help="API key (dummy is OK for many local servers).")
    ap.add_argument("--model", required=True, help="Model name exposed by the server.")
    ap.add_argument("--no-consistency", action="store_true", help="Disable study-level consistency post-processing.")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    items = read_jsonl(Path(args.in_jsonl))

    client = OpenAICompatClient(base_url=args.base_url, api_key=args.api_key, model=args.model)
    reconciler = Reconciler(client)

    out = []
    for it in items:
        visual = it.get("visual_evidence")
        seg = it.get("seg_evidence")
        meta = it.get("meta_evidence")
        report = it.get("report_evidence")

        obj, debug = reconciler.reconcile_one(
            visual=visual,
            seg=seg,
            meta=meta,
            report=report,
            temperature=args.temperature,
        )

        it_out = dict(it)
        it_out["curated_label"] = obj["label"]
        it_out["curated_confidence"] = obj["confidence"]
        it_out["curated_rationale"] = obj.get("rationale", "")
        it_out["curated_overrides"] = obj.get("overrides", [])
        it_out["llm_debug"] = debug
        out.append(it_out)

    if not args.no_consistency:
        out = enforce_study_level_consistency(out)

    write_jsonl(Path(args.out_jsonl), out)


if __name__ == "__main__":
    main()
