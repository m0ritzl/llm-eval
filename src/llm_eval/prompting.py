from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json

DEFAULT_SYSTEM = """You are an expert bilingual evaluator of machine translation quality. Be strict but fair."""

DEFAULT_USER_TEMPLATE = """Evaluate the quality of the following machine translation.
Return ONLY a JSON object with these fields:
score: holistic quality 0-100 (float)
adequacy: 0-5
fluency: 0-5
rationale: short explanation

Source: {source}
Hypothesis: {hypothesis}
{reference_block}
IMPORTANT: Output valid JSON only, no markdown fences.
"""

@dataclass
class PromptTemplate:
    name: str = "default"
    system: str = DEFAULT_SYSTEM
    user_template: str = DEFAULT_USER_TEMPLATE

    def build(self, source: str, hypothesis: str, reference: Optional[str]) -> Dict[str, str]:
        reference_block = f"Reference: {reference}" if reference else ""
        user = self.user_template.format(
            source=source.strip(), hypothesis=hypothesis.strip(), reference_block=reference_block
        )
        return {"system": self.system, "user": user}


def parse_score(raw: str) -> Dict[str, Any]:
    # attempt to extract json
    raw_stripped = raw.strip()
    first_brace = raw_stripped.find("{")
    last_brace = raw_stripped.rfind("}")
    if first_brace == -1 or last_brace == -1:
        return {"error": "no_json", "raw": raw}
    snippet = raw_stripped[first_brace : last_brace + 1]
    try:
        data = json.loads(snippet)
    except Exception as e:  # noqa: BLE001
        return {"error": "json_parse", "exception": str(e), "raw": raw}

    # normalize fields
    result = {
        "score": float(data.get("score")) if isinstance(data.get("score"), (int, float, str)) else None,
        "adequacy": float(data.get("adequacy")) if isinstance(data.get("adequacy"), (int, float, str)) else None,
        "fluency": float(data.get("fluency")) if isinstance(data.get("fluency"), (int, float, str)) else None,
        "rationale": data.get("rationale"),
        "raw_response": raw,
    }
    return result
