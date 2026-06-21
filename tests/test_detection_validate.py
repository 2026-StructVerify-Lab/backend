"""tests/test_detection_validate.py — detection/schema/validate unit tests (API 키 불필요)."""
import pytest

from structverify.core.schemas import ClaimSchema
from structverify.detection.schema.validate import (
    _extract_numbers_from_text,
    _safe_float,
    _source_phrase_in_claim,
    _validate_schema,
    _value_in_claim_text,
    _verify_and_correct_value,
)
from structverify.detection.schema_inductor import _source_phrase_in_claim as reexported


BIRTH_CLAIM = (
    "지난해 출생아 수는 23만 8000명으로 전년(24만 9000명)보다 1만 1000명 감소했다."
)


class TestSourcePhraseInClaim:
    def test_direct_substring(self):
        assert _source_phrase_in_claim("23만 8000명", BIRTH_CLAIM)

    def test_parenthetical_phrase(self):
        assert _source_phrase_in_claim("24만 9000명", BIRTH_CLAIM)

    def test_difference_phrase(self):
        assert _source_phrase_in_claim("1만 1000명", BIRTH_CLAIM)

    def test_rejects_hanja_variant_not_in_claim(self):
        assert not _source_phrase_in_claim("23만 8천명", BIRTH_CLAIM)

    def test_whitespace_insensitive(self):
        claim = "1만 7921 건이 접수됐다."
        assert _source_phrase_in_claim("1만 7921건", claim)

    def test_symbol_diff_via_digit_match(self):
        claim = "소비자물가는 6.8%↑로 상승했다."
        assert _source_phrase_in_claim("6.8%", claim)

    def test_empty_inputs(self):
        assert not _source_phrase_in_claim("", BIRTH_CLAIM)
        assert not _source_phrase_in_claim("23만", "")

    def test_reexported_from_schema_inductor(self):
        assert reexported is _source_phrase_in_claim


class TestExtractNumbersFromText:
    def test_man_pattern(self):
        nums = _extract_numbers_from_text("2만 171")
        assert 20171.0 in nums

    def test_man_four_digit_pattern(self):
        nums = _extract_numbers_from_text("2869만 3000명")
        assert 28693000.0 in nums

    def test_man_cheon_compound(self):
        nums = _extract_numbers_from_text("24만 2천")
        assert 242000.0 in nums

    def test_comma_integer(self):
        nums = _extract_numbers_from_text("238,317명")
        assert 238317.0 in nums

    def test_decimal(self):
        nums = _extract_numbers_from_text("6.7%")
        assert 6.7 in nums


class TestVerifyAndCorrectValue:
    def test_correct_value_unchanged(self):
        val, corrected = _verify_and_correct_value(20171.0, "2만 171")
        assert val == 20171.0
        assert corrected is False

    def test_fixes_wrong_conversion(self):
        val, corrected = _verify_and_correct_value(21710.0, "2만 171")
        assert val == 20171.0
        assert corrected is True

    def test_none_or_empty_phrase(self):
        assert _verify_and_correct_value(None, "2만") == (None, False)
        assert _verify_and_correct_value(100.0, "") == (100.0, False)


class TestValueInClaimText:
    def test_matches_man_notation(self):
        assert _value_in_claim_text(20171.0, "출생아는 2만 171명이다.")

    def test_no_match(self):
        assert not _value_in_claim_text(99999.0, "출생아는 2만 171명이다.")

    def test_none_value_passes(self):
        assert _value_in_claim_text(None, "")


class TestSafeFloat:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (64.2, 64.2),
            ("64.2%", 64.2),
            ("약 64", 64.0),
            ("  ", None),
            ("없음", None),
        ],
    )
    def test_parsing(self, raw, expected):
        assert _safe_float(raw) == expected


class TestValidateSchema:
    def test_valid_indicator(self):
        schema = ClaimSchema.model_construct(indicator="출생아 수")
        assert _validate_schema(schema)

    def test_missing_indicator(self):
        schema = ClaimSchema.model_construct(indicator=None)
        assert not _validate_schema(schema)

    def test_short_indicator(self):
        schema = ClaimSchema.model_construct(indicator="  x ")
        assert not _validate_schema(schema)
