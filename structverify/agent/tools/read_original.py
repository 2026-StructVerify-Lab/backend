"""
structverify.agent.tools.read_original — 원문 기사 읽기 Tool.

Agent가 *claim 텍스트만으로 부족*할 때 원문 기사 전체/일부를 읽음.

예시 사용:
  - "이는 1991년 4월(8.7%) 이후 34년 만에" — claim 자체엔 1991년 비교만 있고
    원문에 *부가 맥락*이 있는지 확인할 때
  - "X→Y" 같은 차이 표현에서 *전년도 정확한 값*을 원문에서 추출하고 싶을 때
  - schema_inductor가 *value=null*로 보낸 schema인데 *원문에 수치가 있는지* 다시 확인

읽기 모드:
  - all: 전체 원문 (긴 기사는 토큰 비용)
  - chars[start:end]: 글자 인덱스로 부분 읽기
  - first/last N: 앞 N자 / 끝 N자

데이터 출처는 workspace.source.txt (job 시작 시 initialize됨).
"""
from __future__ import annotations

from structverify.utils.logger import get_logger
from typing import Any

from ..schemas import ActionType
from .base import ToolBase, ToolContext, ToolResult, register_tool

logger = get_logger(__name__)


# 한 번에 읽는 최대 길이 (LLM 토큰 절약). 너무 길면 잘라냄.
_DEFAULT_MAX_CHARS = 5000


@register_tool(ActionType.READ_ORIGINAL)
class ReadOriginalTool(ToolBase):
    """원문 기사 읽기.

    claim의 source_phrase만으론 부족할 때, 기사 *주변 문장* 또는 *전체*를 봄.
    """

    name = ActionType.READ_ORIGINAL
    description = (
        "원문 기사 읽기. claim의 source_phrase만으로 부족할 때 *주변 맥락* 또는 *전체* 확인. "
        "예: '지난해 같은 달'이 정확히 몇 년인지, 또는 비교 기준값이 원문에 있는지 등."
    )
    input_schema = {
        "mode": "읽기 모드: 'all' (전체) | 'first' (앞부분) | 'last' (끝부분) | 'chars' (인덱스 범위)",
        "max_chars": "(선택) 최대 글자 수. 기본 5000. 'all' 모드에서 길면 잘라냄.",
        "start": "(mode=chars일 때) 시작 글자 인덱스",
        "end": "(mode=chars일 때) 끝 글자 인덱스",
    }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        mode = (input_data.get("mode") or "all").lower().strip()
        try:
            max_chars = int(input_data.get("max_chars") or _DEFAULT_MAX_CHARS)
        except (TypeError, ValueError):
            max_chars = _DEFAULT_MAX_CHARS

        # workspace에서 원문 읽기
        try:
            full_text = context.workspace.read_source()
        except FileNotFoundError:
            return ToolResult(
                output={},
                summary="실패: workspace에 source.txt 없음",
                success=False,
                error="workspace.read_source() 실패 — source.txt가 없습니다. "
                      "Job 시작 시 workspace.initialize(source_text=...)가 호출됐는지 확인.",
            )
        except Exception as e:
            return ToolResult(
                output={},
                summary=f"실패: 원문 읽기 — {e}",
                success=False,
                error=str(e),
            )

        total_chars = len(full_text)

        # 모드별 처리
        if mode == "all":
            text = full_text
            truncated = False
            if len(text) > max_chars:
                text = text[:max_chars]
                truncated = True
            span_desc = f"전체 ({total_chars}자)" + (" — 잘림" if truncated else "")

        elif mode == "first":
            text = full_text[:max_chars]
            span_desc = f"앞 {len(text)}자 (전체 {total_chars}자)"

        elif mode == "last":
            text = full_text[-max_chars:]
            span_desc = f"끝 {len(text)}자 (전체 {total_chars}자)"

        elif mode == "chars":
            try:
                start = int(input_data.get("start", 0))
                end = int(input_data.get("end", total_chars))
            except (TypeError, ValueError) as e:
                return ToolResult(
                    output={},
                    summary=f"실패: start/end 변환 — {e}",
                    success=False,
                    error=f"start/end는 정수여야 합니다: {e}",
                )
            # 경계 검증
            start = max(0, min(start, total_chars))
            end = max(start, min(end, total_chars))
            text = full_text[start:end]
            # max_chars 제한도 적용
            if len(text) > max_chars:
                text = text[:max_chars]
            span_desc = f"chars[{start}:{end}] ({len(text)}자)"

        else:
            return ToolResult(
                output={},
                summary=f"실패: 알 수 없는 mode={mode!r}",
                success=False,
                error="mode는 'all' | 'first' | 'last' | 'chars' 중 하나여야 합니다.",
            )

        # 요약은 *처음 100자*만 미리보기로
        preview = text.strip()[:100].replace("\n", " ")
        if len(text) > 100:
            preview += "..."

        return ToolResult(
            output={
                "text": text,
                "total_chars": total_chars,
                "returned_chars": len(text),
                "mode": mode,
            },
            summary=f"원문 읽음 ({span_desc}): {preview}",
            success=True,
        )
