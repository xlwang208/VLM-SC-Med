from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .llm_client import OpenAICompatClient

ALLOWED_LABELS: Set[str] = {"head", "neck", "chest", "abdomen", "pelvis", "other"}
ALLOWED_CONF: Set[str] = {"low", "medium", "high"}


SYSTEM_PROMPT = """You are a medical imaging dataset curation assistant.
Your task: infer the anatomical coverage label for a CT series using MULTI-MODAL EVIDENCE.
You must resolve conflicts and output a single label from the allowed set.

Allowed labels:
- head
- neck
- chest
- abdomen
- pelvis
- other

Rules:
- Prefer physical/image-grounded evidence when available (e.g., detected organs / structures).
- Metadata and report titles can be wrong or incomplete.
- If the evidence is insufficient or conflicting beyond resolution, output "other" with low confidence.
- Output ONLY a JSON object (no markdown), with keys:
  label: one of allowed labels
  confidence: low|medium|high
  rationale: short string
  overrides: list of strings describing which sources you overrode and why
"""

USER_TEMPLATE = """Evidence for one CT series:

1) Visual prediction (from 2D projections):
{visual_evidence}

2) Segmentation-derived organ presence (binary/summary):
{seg_evidence}

3) DICOM metadata fields (normalized):
{meta_evidence}

4) Radiology report evidence (NLP/LLM-extracted):
{report_evidence}

Return JSON as specified.
"""


def _format_dict(d: Optional[Dict[str, Any]]) -> str:
    if not d:
        return "(missing)"
    lines = []
    for k, v in d.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


@dataclass
class Reconciler:
    client: OpenAICompatClient

    def reconcile_one(
        self,
        *,
        visual: Optional[Dict[str, Any]] = None,
        seg: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        report: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 768,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        user = USER_TEMPLATE.format(
            visual_evidence=_format_dict(visual),
            seg_evidence=_format_dict(seg),
            meta_evidence=_format_dict(meta),
            report_evidence=_format_dict(report),
        )
        obj, debug = self.client.chat_json(SYSTEM_PROMPT, user, temperature=temperature, max_tokens=max_tokens)

        # sanitize
        label = str(obj.get("label", "other")).strip().lower()
        conf = str(obj.get("confidence", "low")).strip().lower()
        if label not in ALLOWED_LABELS:
            label = "other"
        if conf not in ALLOWED_CONF:
            conf = "low"
        obj["label"] = label
        obj["confidence"] = conf
        if "overrides" in obj and not isinstance(obj["overrides"], list):
            obj["overrides"] = [str(obj["overrides"])]
        if "overrides" not in obj:
            obj["overrides"] = []
        return obj, debug
