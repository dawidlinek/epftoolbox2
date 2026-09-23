import pandas as pd

from epftoolbox2.data.validators import ValidationResult, results_to_frame


def test_str_includes_info():
    result = ValidationResult(is_valid=False, errors=["gap"], info={"gap_count": 3, "expected_freq": "15min"})

    text = str(result)

    assert "gap_count: 3" in text
    assert "expected_freq: '15min'" in text


def test_str_truncates_long_info_lists():
    result = ValidationResult(info={"gaps": list(range(100))})

    line = next(line for line in str(result).splitlines() if "gaps" in line)

    assert "95 more" in line
    assert len(line) < 200


def test_str_names_the_validator_and_mentions_stats():
    result = ValidationResult(validator="NullCheckValidator", stats=pd.DataFrame({"column": ["a", "b"]}))

    text = str(result)

    assert text.startswith("NullCheckValidator: VALID")
    assert "Stats: 2 rows" in text


def test_add_error_marks_result_invalid():
    result = ValidationResult()
    result.add_error("broken")

    assert not result.is_valid
    assert result.errors == ["broken"]


def test_results_to_frame_one_row_per_message():
    ok = ValidationResult(validator="ContinuityValidator")
    bad = ValidationResult(is_valid=False, errors=["e1", "e2"], warnings=["w1"], validator="NullCheckValidator")

    table = results_to_frame([ok, bad])

    assert list(table.columns) == ["validator", "valid", "level", "message"]
    assert table.to_dict("records") == [
        {"validator": "ContinuityValidator", "valid": True, "level": "ok", "message": ""},
        {"validator": "NullCheckValidator", "valid": False, "level": "error", "message": "e1"},
        {"validator": "NullCheckValidator", "valid": False, "level": "error", "message": "e2"},
        {"validator": "NullCheckValidator", "valid": False, "level": "warning", "message": "w1"},
    ]


def test_results_to_frame_accepts_names():
    result = ValidationResult(warnings=["w"])

    by_dict = results_to_frame({"prices": result})
    by_pairs = results_to_frame([("prices", result)])

    assert list(by_dict["validator"]) == ["prices"]
    assert by_dict.equals(by_pairs)
    assert result.to_frame().loc[0, "level"] == "warning"


def test_results_to_frame_empty():
    assert results_to_frame([]).empty
