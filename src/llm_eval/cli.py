from __future__ import annotations
import typer
from typing import Optional
import os
from .data import read_tsv
from .evaluator import LLMScorer, score_multiple
from .metrics.basic import corpus_bleu
import json
from pathlib import Path
from dotenv import load_dotenv

app = typer.Typer(help="LLM-Eval: LLM-based MT evaluation (COMET-style) using Azure OpenAI")

@app.callback()
def _load_env():
    load_dotenv()

@app.command()
def score(
    data: str = typer.Option(..., help="Path to TSV file"),
    has_reference: bool = typer.Option(True, help="File includes reference column"),
    header: bool = typer.Option(False, help="First row is header"),
    output: Optional[str] = typer.Option(None, help="Write per-segment JSONL here"),
    jsonl: bool = typer.Option(False, help="Emit JSONL instead of summary only"),
    deployment: Optional[str] = typer.Option(None, help="Single deployment/model name override"),
    models: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of deployments to score with (overrides --deployment).",
    ),
    no_parallel: bool = typer.Option(False, help="Disable parallel scoring across models"),
):
    segments = read_tsv(data, has_reference=has_reference, header=header)
    # Resolve model list
    resolved_models: Optional[list[str]] = None
    if models:
        resolved_models = [m.strip() for m in models.split(",") if m.strip()]
    elif os.getenv("AZURE_OPENAI_DEPLOYMENTS"):
        resolved_models = [
            m.strip() for m in os.getenv("AZURE_OPENAI_DEPLOYMENTS", "").split(",") if m.strip()
        ]
    elif deployment:
        resolved_models = [deployment]
    else:
        # Fallback: if single deployment env contains commas, interpret as list
        dep_env = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip().strip('"').strip("'")
        if "," in dep_env:
            parts = [p.strip() for p in dep_env.split(",") if p.strip()]
            if parts:
                typer.echo(
                    "[warning] Detected comma-separated deployments in AZURE_OPENAI_DEPLOYMENT; using multi-model mode."
                )
                resolved_models = parts

    if resolved_models and len(resolved_models) > 1:
        multi = score_multiple(
            segments,
            deployments=resolved_models,
            parallel=not no_parallel,
        )
        # Optionally write JSONL (includes model per segment)
        if jsonl and output:
            with open(output, "w", encoding="utf-8") as f:
                for model_name, model_block in multi["models"].items():
                    for seg in model_block["segments"]:
                        f.write(json.dumps(seg, ensure_ascii=False) + "\n")
        typer.echo(json.dumps(multi, indent=2))
        return

    # Single model path (legacy behavior)
    single_deployment = resolved_models[0] if resolved_models else deployment
    scorer = LLMScorer(deployment=single_deployment)
    result = scorer.score(segments)
    # Add model field to each segment for consistency
    for seg in result["segments"]:
        seg["model"] = scorer.deployment
    if jsonl and output:
        with open(output, "w", encoding="utf-8") as f:
            for seg in result["segments"]:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")
    typer.echo(json.dumps({"model": scorer.deployment, **result["aggregate"]}, indent=2))

@app.command()
def metrics(
    data: str = typer.Option(..., help="Path to TSV file"),
    has_reference: bool = typer.Option(True, help="File includes reference column"),
    header: bool = typer.Option(False, help="First row is header"),
):
    if not has_reference:
        typer.echo("Need references for BLEU")
        raise typer.Exit(code=1)
    segments = read_tsv(data, has_reference=True, header=header)
    refs = [[*s.reference.split()] for s in segments if s.reference]  # type: ignore[arg-type]
    hyps = [[*s.hypothesis.split()] for s in segments]
    bleu = corpus_bleu(refs, hyps)
    typer.echo(json.dumps({"BLEU": bleu}, indent=2))

if __name__ == "__main__":  # pragma: no cover
    app()
