from concor_video.entity_extraction import extract_entities


def test_parenthesized_plural_target_has_nonempty_exact_span() -> None:
    result = extract_entities("seating(s) designed for rest.")
    target = result["target"]
    assert target["surface"] == "seating"
    assert (target["start"], target["end"]) == (0, 7)
