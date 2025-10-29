from __future__ import annotations
from typing import List, Optional, Dict, Any
from .data import Segment
from .prompting import PromptTemplate, parse_score
from .azure_client import chat_completion as default_chat_completion
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import statistics

class LLMScorer:
    def __init__(self, deployment: Optional[str] = None, template: Optional[PromptTemplate] = None):
        raw_dep = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt4o")
        # Strip accidental surrounding quotes
        raw_dep = raw_dep.strip().strip("'").strip('"')
        # If user mistakenly provided multiple models in single env var, take first and note
        if "," in raw_dep:
            parts = [p.strip() for p in raw_dep.split(",") if p.strip()]
            if parts:
                # Keep first; multi-model should be done via --models or AZURE_OPENAI_DEPLOYMENTS
                self.deployment = parts[0]
            else:
                self.deployment = raw_dep
        else:
            self.deployment = raw_dep
        self.template = template or PromptTemplate()

    def score_segment(self, seg: Segment, _chat_fn=default_chat_completion) -> Dict[str, Any]:
        prompt = self.template.build(seg.source, seg.hypothesis, seg.reference)
        raw = _chat_fn(deployment=self.deployment, system=prompt["system"], user=prompt["user"])
        parsed = parse_score(raw)
        parsed.update({"id": seg.id})
        return parsed

    def score(self, segments: List[Segment]) -> Dict[str, Any]:
        results = [self.score_segment(s) for s in segments]
        scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
        aggregate = {
            "mean_score": statistics.fmean(scores) if scores else None,
            "num_segments": len(segments),
            "num_scored": len(scores),
        }
        return {"segments": results, "aggregate": aggregate}


def score_multiple(
    segments: List[Segment],
    deployments: List[str],
    parallel: bool = True,
    max_workers: Optional[int] = None,
    chat_fn=default_chat_completion,
) -> Dict[str, Any]:
    """Score the same set of segments with multiple model deployments.

    Parameters
    ----------
    segments : list[Segment]
        Segments to evaluate.
    deployments : list[str]
        Deployment / model names.
    parallel : bool, default True
        If True, evaluate models concurrently (one thread per model).
    max_workers : int | None
        Cap workers; defaults to len(deployments) when parallel.

    Returns
    -------
    dict
        {
          "models": {
              deployment: {"segments": [...], "aggregate": {...}}
          },
          "summary": {"model_aggregates": [ {"model": deployment, ...aggregate...}, ... ]}
        }
    """
    results: Dict[str, Dict[str, Any]] = {}

    def _run(dep: str) -> tuple[str, Dict[str, Any]]:
        scorer = LLMScorer(deployment=dep)
        # Use injection by wrapping scorer.score_segment
        model_result: Dict[str, Any] = {"segments": [], "aggregate": {}}
        try:
            for s in segments:
                try:
                    seg_res = scorer.score_segment(s, _chat_fn=chat_fn)
                except Exception as seg_err:  # noqa: BLE001
                    seg_res = {"id": s.id, "error": str(seg_err)}
                seg_res["model"] = dep
                model_result["segments"].append(seg_res)
            scores = [r["score"] for r in model_result["segments"] if isinstance(r.get("score"), (int, float))]
            model_result["aggregate"] = {
                "mean_score": statistics.fmean(scores) if scores else None,
                "num_segments": len(segments),
                "num_scored": len(scores),
                "num_errors": sum(1 for r in model_result["segments"] if "error" in r),
            }
        except Exception as model_err:  # noqa: BLE001
            # Catastrophic model failure (e.g., auth / model not deployed)
            model_result = {
                "segments": [],
                "aggregate": {
                    "mean_score": None,
                    "num_segments": len(segments),
                    "num_scored": 0,
                    "num_errors": len(segments),
                    "error": str(model_err),
                },
            }
        return dep, model_result

    if parallel and len(deployments) > 1:
        with ThreadPoolExecutor(max_workers=max_workers or len(deployments)) as ex:
            future_map = {ex.submit(_run, d): d for d in deployments}
            for fut in as_completed(future_map):
                dep, res = fut.result()
                results[dep] = res
    else:
        for d in deployments:
            dep, res = _run(d)
            results[dep] = res

    summary = {
        "model_aggregates": [
            {"model": dep, **res["aggregate"]} for dep, res in results.items()
        ]
    }
    return {"models": results, "summary": summary}
