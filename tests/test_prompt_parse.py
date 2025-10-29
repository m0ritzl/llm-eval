from llm_eval.prompting import parse_score

def test_parse_valid_json():
    raw = "Some preamble {\n  \"score\": 77.5, \"adequacy\": 4, \"fluency\": 5, \"rationale\": \"ok\"\n} extra"
    parsed = parse_score(raw)
    assert parsed["score"] == 77.5
    assert parsed["adequacy"] == 4.0
    assert parsed["fluency"] == 5.0
    assert parsed["rationale"] == "ok"


def test_parse_failure_returns_error():
    raw = "No braces here"
    parsed = parse_score(raw)
    assert parsed["error"] == "no_json"
