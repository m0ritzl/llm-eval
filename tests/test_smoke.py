from llm_eval.data import read_tsv
from llm_eval.evaluator import LLMScorer

class DummyScorer(LLMScorer):
    def score_segment(self, seg):  # type: ignore[override]
        # Deterministic fake response for test stability
        return {
            "id": seg.id,
            "score": 80.0 + seg.id,
            "adequacy": 4.0,
            "fluency": 4.0,
            "rationale": "dummy",
            "raw_response": "{\"score\": 80, \"adequacy\":4, \"fluency\":4, \"rationale\":\"dummy\"}"
        }

def test_smoke_sample_file():
    segs = read_tsv("data/sample.tsv", has_reference=True, header=False)
    scorer = DummyScorer()
    result = scorer.score(segs)
    assert result["aggregate"]["num_segments"] == 3
    assert result["aggregate"]["num_scored"] == 3
    assert result["aggregate"]["mean_score"] == 81.0  # (80 + 81 + 82)/3
