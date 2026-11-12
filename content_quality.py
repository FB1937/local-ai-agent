"""Deterministic quality signals for extracted web evidence.

These heuristics describe extraction quality only; they do not establish factual
accuracy, source authority, or whether a document answers the user's question.
"""
from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+#/-]*|[가-힣]{2,}")
REPLACEMENT_CHAR = "\ufffd"


def assess_extracted_content(text: str | None, document_type: str = "html") -> dict:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    char_count = len(normalized)
    tokens = TOKEN_RE.findall(normalized)
    token_count = len(tokens)
    unique_ratio = (
        len({token.casefold() for token in tokens}) / token_count
        if token_count else 0.0
    )
    alphanumeric_count = sum(character.isalnum() for character in normalized)
    alphanumeric_ratio = alphanumeric_count / char_count if char_count else 0.0

    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    repeated_ratio = (
        1 - len(set(lines)) / len(lines)
        if lines else 0.0
    )
    replacement_ratio = (
        (text or "").count(REPLACEMENT_CHAR) / max(1, len(text or ""))
    )

    if not normalized:
        status = "no_text"
        reason = (
            "PDF에 추출 가능한 텍스트가 없습니다. 스캔 이미지 기반 문서일 가능성이 있지만 "
            "빈 문서나 추출 실패와 구분되지는 않습니다."
            if document_type == "pdf"
            else "추출된 본문 텍스트가 없습니다."
        )
    elif char_count < 120 or token_count < 15:
        status = "thin"
        reason = "본문 길이 또는 토큰 수가 적어 근거로 사용하기에 부족할 수 있습니다."
    elif alphanumeric_ratio < 0.35 or unique_ratio < 0.12 or repeated_ratio > 0.55:
        status = "low_signal"
        reason = "문자 구성, 어휘 다양성 또는 반복 비율이 낮아 노이즈 가능성이 있습니다."
    elif replacement_ratio > 0.01:
        status = "encoding_warning"
        reason = "문자 인코딩 대체 문자가 포함되어 일부 내용이 손상되었을 수 있습니다."
    else:
        status = "usable"
        reason = "기본적인 길이와 텍스트 품질 기준을 충족했습니다."

    score = 0.0
    if char_count:
        length_score = min(1.0, char_count / 1200)
        diversity_score = min(1.0, unique_ratio / 0.5)
        signal_score = min(1.0, alphanumeric_ratio / 0.75)
        score = round(0.4 * length_score + 0.35 * diversity_score + 0.25 * signal_score, 3)
        if repeated_ratio > 0.55:
            score = round(score * 0.6, 3)
        if replacement_ratio > 0.01:
            score = round(score * 0.8, 3)

    return {
        "status": status,
        "score": score,
        "char_count": char_count,
        "token_count": token_count,
        "unique_token_ratio": round(unique_ratio, 3),
        "alphanumeric_ratio": round(alphanumeric_ratio, 3),
        "repeated_line_ratio": round(repeated_ratio, 3),
        "replacement_char_ratio": round(replacement_ratio, 4),
        "reason": reason,
        "is_factual_verification": False,
    }
