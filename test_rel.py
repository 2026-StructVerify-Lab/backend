# -*- coding: utf-8 -*-
from structverify.detection.schema_inductor import _source_phrase_in_claim

claim = "지난해 출생아 수는 23만 8000명으로 전년(24만 9000명)보다 1만 1000명 감소했다."

print(_source_phrase_in_claim("23만 8000명", claim))   # True여야 함
print(_source_phrase_in_claim("24만 9000명", claim))   # True여야 함
print(_source_phrase_in_claim("1만 1000명", claim))    # True여야 함
print(_source_phrase_in_claim("23만 8천명", claim))    # False → 폐기 원인