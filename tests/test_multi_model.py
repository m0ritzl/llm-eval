import os
from llm_eval.evaluator import score_multiple, LLMScorer
from llm_eval.data import Segment


def test_score_multiple_models(monkeypatch):
    segs = [Segment(id=0, source="a", hypothesis="b", reference="c")]
    deployments = ["modelA", "modelB"]

    def fake_chat_completion(deployment: str, system: str, user: str, **_: object) -> str:
        return '{"score": %d, "adequacy": 1, "fluency": 1, "rationale": "ok"}' % len(deployment)

    results = score_multiple(segs, deployments=deployments, parallel=False, chat_fn=fake_chat_completion)
    assert set(results["models"].keys()) == set(deployments)
    for d in deployments:
        seg_entry = results["models"][d]["segments"][0]
        assert seg_entry["model"] == d
        assert seg_entry["score"] == len(d)


def test_quoted_env_single(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", '"single-model"')
    scorer = LLMScorer()
    assert scorer.deployment == "single-model"
