# LLM-Eval

A lightweight reimplementation of the core user-facing functionality of [Unbabel/COMET] but powered by Azure OpenAI LLM + (optionally) Azure embeddings instead of proprietary COMET neural scoring models.

## Goals
- Score Machine Translation (MT) hypotheses against source + reference using LLM judge prompts (Direct Assessment style)
- Provide *reference-free* (source+hyp only) and *reference-based* (source+ref+hyp) modes
- Offer aggregate metrics: mean score, z-normalized score, per-segment rationale
- Offer classic automatic metrics (BLEU, chrF, TER (planned)) for comparison
- Allow pluggable prompt templates and few-shot examples
- Reproducible, batchable, and streaming friendly

## High-Level Approach
1. Load triplets (source, reference, hypothesis) from TSV/CSV/JSONL
2. For each segment build a structured judging prompt:
   - Task framing
   - Quality dimensions (adequacy, fluency)
   - Numeric holistic score 0–100
   - Optional rationale
3. Send as chat completion to Azure OpenAI deployment (gpt-4o, gpt-4.1, o3-mini, etc)
4. Parse structured JSON block from model output
5. Aggregate + export

## Installation
```bash
pip install -e .
```
Copy `.env.example` to `.env` and set:
```
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt4o
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

## CLI Usage
Reference-based (has references):
```bash
llm-eval score --data data/sample.tsv --has-reference
```
Reference-free:
```bash
llm-eval score --data data/sample.tsv
```
Export JSONL with segment scores:
```bash
llm-eval score --data data/sample.tsv --output scores.jsonl --jsonl
```

Compute BLEU/chrF only:
```bash
llm-eval metrics --data data/sample.tsv --has-reference
```

## Input Format
Tab-separated by default:
```
source \t reference \t hypothesis
```
If `--no-header` is omitted and the first line contains these column names they will be auto-detected.
For reference-free mode, omit the middle column or pass `--columns source:hypothesis`.

## Prompt Customization
Provide a YAML file:
```yaml
name: adequacy_fluency_v1
system: |
  You are an expert bilingual evaluator of machine translation quality.
user_template: |
  Evaluate the translation quality.
  Source: {source}
  Hypothesis: {hypothesis}
  Reference: {reference_optional}
  Produce JSON with fields: score (0-100), adequacy (0-5), fluency (0-5), rationale.
```
Then:
```bash
llm-eval score --data sample.tsv --prompt prompt.yaml
```

## Output Schema (per segment)
```json
{
  "id": 17,
  "score": 82.5,
  "adequacy": 4.0,
  "fluency": 4.0,
  "rationale": "Meaning preserved; minor tense issue.",
  "raw_response": "... full model text ..."
}
```

## Roadmap
- [ ] Add TER
- [ ] Caching layer (sqlite) to avoid re-judging identical triplets
- [ ] Batch parallelism with asyncio
- [ ] Azure embeddings semantic similarity auxiliary metric
- [ ] Few-shot example injection utility
- [ ] Web dashboard (FastAPI + simple front-end)

## Disclaimer
LLM-based evaluation can drift; periodically re-benchmark against human DA scores. Always sanity check a sample manually.

---
MIT License.
