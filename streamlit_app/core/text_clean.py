from __future__ import annotations
import re
import unicodedata
from typing import List

_TOKEN_RE = re.compile(r"\S+")

def normalize_unicode(text: str) -> str:
    # 전각(｢ ｣ 등) / 이상한 공백 등을 정규화
    return unicodedata.normalize("NFKC", text)

def collapse_consecutive_duplicate_tokens(text: str, keep: int = 1) -> str:
    """
    '2024 2024 2024' / '년 년 년' 처럼 공백 기반 연속 토큰 중복 축약.
    """
    toks = _TOKEN_RE.findall(text)
    if not toks:
        return text
    out: List[str] = []
    prev = None
    run = 0
    for t in toks:
        if t == prev:
            run += 1
        else:
            prev = t
            run = 1
        if run <= keep:
            out.append(t)
    return " ".join(out)

def post_clean_text(text: str) -> str:
    """
    pp_v6 산출물/추출 텍스트에 대해 '앱 레벨'에서 안전하게 노이즈 완화.
    - NFKC 정규화(전각 제거)
    - 과도한 공백/빈줄 정리
    - 과도한 특수문자 반복 축약
    - 연속 중복 토큰 축약(가장 중요)
    """
    if not text:
        return ""

    t = text.replace("\x00", " ")
    t = normalize_unicode(t)

    # 공백/개행 정리
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{4,}", "\n\n", t)

    # 점선/기호 반복 축약
    t = re.sub(r"[·.…]{5,}", " ", t)
    t = re.sub(r"(.)\1{8,}", r"\1\1", t)  # 같은 문자 9회+ → 2개로

    # 핵심: 연속 토큰 중복 축약
    t = collapse_consecutive_duplicate_tokens(t, keep=1)

    return t.strip()