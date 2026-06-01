"""Mini SIR + oracle Claim alignment for outcome eval."""
from __future__ import annotations

import hashlib
from uuid import UUID, uuid4

from structverify.core.schemas import Claim, ClaimSchema, SourceOffset, SourceType, SIRDocument
from structverify.eval.schemas import OutcomeCase
from structverify.preprocessing.sir_builder import build_sir


def _doc_text(case: OutcomeCase) -> str:
    parts: list[str] = []
    if case.context_text:
        parts.append(case.context_text.strip())
    parts.append(case.claim_text.strip())
    return "\n\n".join(parts)


def build_sir_for_case(case: OutcomeCase) -> SIRDocument:
    text = _doc_text(case)
    sir_doc = build_sir(text, SourceType.TEXT)
    text_hash = hashlib.md5(text.encode()).hexdigest()
    sir_doc.doc_id = UUID(text_hash)
    # Agent uses raw_text when present (matches FastAPI path)
    sir_doc.raw_text = text
    return sir_doc


def _sentence_matches(claim_text: str, sent_text: str) -> bool:
    c = claim_text.strip()
    s = sent_text.strip()
    if c == s:
        return True
    if c in s or s in c:
        return True
    return False


def _oracle_schema_from_case(case: OutcomeCase) -> ClaimSchema:
    return ClaimSchema(
        indicator=case.indicator,
        time_period=case.time_period,
        unit=case.unit,
        value=case.stated_value,
        population="전체",
        parent_path=None,
        modifier=None,
        value_role="base",
    )


def claims_from_case(
    case: OutcomeCase,
    sir_doc: SIRDocument,
    *,
    schema_mode: str = "induce",
) -> list[Claim]:
    """Build oracle Claim; attach gold schema when schema_mode is oracle."""
    target_block = "b0000"
    target_sent = "b0000_s0000"
    found = False
    for block in sir_doc.blocks:
        for sent in block.sentences:
            if _sentence_matches(case.claim_text, sent.text):
                target_block = block.block_id
                target_sent = sent.sent_id
                found = True
                break
        if found:
            break

    schema = _oracle_schema_from_case(case) if schema_mode == "oracle" else None

    claim = Claim(
        claim_id=uuid4(),
        doc_id=sir_doc.doc_id,
        block_id=target_block,
        sent_id=target_sent,
        claim_text=case.claim_text,
        source_offset=SourceOffset(),
        check_worthy_score=1.0,
        context_text=case.context_text or case.claim_text,
        schema=schema,
    )
    return [claim]
