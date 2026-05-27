import os
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from structverify.core.schemas import Evidence, VerdictType, VerificationResult
from structverify.eval.audit.kosis_grounding import (
    check_kosis_grounding,
    grounding_from_config,
    normalize_kosis_base_url,
    parse_stat_table_id,
    resolve_kosis_api_key,
    resolve_org_and_tbl,
)


def test_parse_stat_table_id():
    assert parse_stat_table_id("DT_101_DT_920012N_2058") == ("101", "DT_920012N_2058")


def test_resolve_org_and_tbl_with_hint():
    assert resolve_org_and_tbl("DT_200Y108", org_id_hint="301") == ("301", "DT_200Y108")
    assert resolve_org_and_tbl("TX_10506_A080", org_id_hint="127") == (
        "127",
        "TX_10506_A080",
    )


def test_normalize_kosis_base_url_strips_statistics_path():
    assert normalize_kosis_base_url(None) == "https://kosis.kr/openapi"
    assert (
        normalize_kosis_base_url("https://kosis.kr/openapi/statisticsData.do")
        == "https://kosis.kr/openapi"
    )


def test_resolve_kosis_api_key_from_env():
    with patch.dict(os.environ, {"KOSIS_API_KEY": "test-key"}, clear=False):
        assert resolve_kosis_api_key({}) == "test-key"
    assert resolve_kosis_api_key({"api_key": "inline"}) == "inline"


@pytest.mark.asyncio
async def test_grounding_ok():
    result = VerificationResult(
        claim_id=uuid4(),
        verdict=VerdictType.MATCH,
        evidence=Evidence(
            source_name="KOSIS",
            stat_table_id="DT_101_DT_920012N_2058",
            official_value=1.0,
        ),
    )
    with patch(
        "structverify.eval.audit.kosis_grounding.kosis_get_meta",
        new_callable=AsyncMock,
        return_value={"row": []},
    ):
        g = await check_kosis_grounding(
            result, api_key="key", base_url="https://kosis.kr/openapi"
        )
    assert g["kosis_grounding_ok"] is True


@pytest.mark.asyncio
async def test_grounding_dt_short_id_with_org_hint():
    result = VerificationResult(
        claim_id=uuid4(),
        verdict=VerdictType.MATCH,
        evidence=Evidence(
            source_name="KOSIS",
            stat_table_id="DT_200Y108",
            official_value=1.0,
        ),
    )
    with patch(
        "structverify.eval.audit.kosis_grounding.kosis_get_meta",
        new_callable=AsyncMock,
        return_value={"row": []},
    ) as mock_meta:
        g = await check_kosis_grounding(
            result,
            api_key="key",
            base_url="https://kosis.kr/openapi",
            org_id_hint="301",
        )
    assert g["kosis_grounding_ok"] is True
    mock_meta.assert_awaited_once()
    assert mock_meta.await_args[0][3:5] == ("301", "DT_200Y108")


@pytest.mark.asyncio
async def test_grounding_passes_openapi_base_not_doubled_path():
    result = VerificationResult(
        claim_id=uuid4(),
        verdict=VerdictType.MATCH,
        evidence=Evidence(
            source_name="KOSIS",
            stat_table_id="DT_200Y108",
            official_value=1.0,
        ),
    )
    with patch(
        "structverify.eval.audit.kosis_grounding.kosis_get_meta",
        new_callable=AsyncMock,
        return_value={"row": []},
    ) as mock_meta:
        await check_kosis_grounding(
            result,
            api_key="key",
            base_url="https://kosis.kr/openapi/statisticsData.do",
            org_id_hint="301",
        )
    assert mock_meta.await_args[0][1] == "https://kosis.kr/openapi"


@pytest.mark.asyncio
async def test_grounding_error_includes_detail():
    result = VerificationResult(
        claim_id=uuid4(),
        verdict=VerdictType.MATCH,
        evidence=Evidence(
            source_name="KOSIS",
            stat_table_id="DT_200Y108",
            official_value=1.0,
        ),
    )
    with patch(
        "structverify.eval.audit.kosis_grounding.kosis_get_meta",
        new_callable=AsyncMock,
        return_value={"kosis_error": "http", "detail": "404 Not Found"},
    ):
        g = await check_kosis_grounding(
            result, api_key="key", base_url="https://kosis.kr/openapi", org_id_hint="301"
        )
    assert g["kosis_grounding_ok"] is False
    assert "http" in (g.get("grounding_error") or "")
    assert "404" in (g.get("grounding_error") or "")


@pytest.mark.asyncio
async def test_grounding_from_config_uses_env_key():
    result = VerificationResult(
        claim_id=uuid4(),
        verdict=VerdictType.MATCH,
        evidence=Evidence(
            source_name="KOSIS",
            stat_table_id="DT_200Y108",
            official_value=1.0,
        ),
    )
    with patch.dict(os.environ, {"KOSIS_API_KEY": "env-key"}, clear=False):
        with patch(
            "structverify.eval.audit.kosis_grounding.kosis_get_meta",
            new_callable=AsyncMock,
            return_value={"row": []},
        ):
            g = await grounding_from_config(
                result, {"kosis": {}}, org_id_hint="301"
            )
    assert g["kosis_grounding_ok"] is True
    assert g.get("grounding_error") != "no_api_key"
