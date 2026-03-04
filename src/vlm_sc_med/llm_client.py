from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI


def _extract_first_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Robustly parse the first JSON object from a model response.
    Returns (obj, err). If parsing fails, obj=None and err is a short code.
    """
    try:
        return json.loads(text), None
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, "no_json_found"
    try:
        return json.loads(m.group(0)), None
    except Exception:
        return None, "json_parse_failed"


@dataclass
class OpenAICompatClient:
    """
    Minimal OpenAI-compatible client wrapper.

    Works with:
    - OpenAI API
    - vLLM OpenAI-compatible server
    - any OpenAI-compatible endpoint that supports chat.completions

    You must provide:
      - base_url (e.g. http://127.0.0.1:8001/v1 for local vLLM)
      - api_key (can be a dummy string for local servers that ignore auth)
    """
    base_url: str
    api_key: str
    model: str
    timeout: int = 120

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    def chat_json(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns (parsed_json, debug) where debug contains raw_text and parse_error if any.
        """
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        obj, err = _extract_first_json(text)
        debug = {"raw_text": text, "parse_error": err}
        if obj is None:
            obj = {"label": "other", "confidence": "low", "error": err or "unknown", "raw_text": text}
        return obj, debug
