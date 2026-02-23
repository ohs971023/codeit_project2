# streamlit_app/core/llm.py
from __future__ import annotations

from typing import Any, List
from openai import OpenAI


def _supports_temperature(model: str) -> bool:
    """
    경험적으로 gpt-5 / o-series(Reasoning) 계열은 temperature/top_p 같은 샘플링 파라미터를
    아예 받지 않는 경우가 많음(400 Unsupported parameter). 그래서 해당 계열은 False 처리.
    """
    m = (model or "").strip().lower()
    if m.startswith("gpt-5"):
        return False
    if m.startswith("o"):
        return False
    return True


def _extract_text(resp: Any) -> str:
    """
    Responses API 응답에서 텍스트를 최대한 안전하게 추출.
    - resp.output_text가 있으면 우선 사용
    - 없으면 resp.output의 message -> output_text 파트에서 추출
    """
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    out = getattr(resp, "output", None) or []
    collected: List[str] = []

    for item in out:
        # message 타입만 대상으로
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None) or []
        for part in content:
            if getattr(part, "type", None) == "output_text":
                t = getattr(part, "text", None)
                if t:
                    collected.append(str(t))

    return "\n".join([c.strip() for c in collected if c.strip()]).strip()


def summarize_with_evidence(
    api_key: str,
    model: str,
    query: str,
    evidence: str,
    pages: list[int],
    *,
    temperature: float = 0.2,
    max_completion_tokens: int = 512,
    fallback_model: str = "gpt-4.1-mini",
    max_retries: int = 2,
) -> str:
    """
    근거 기반 요약/질의응답 + 근거 페이지 출력.

    빈 응답 방지:
    - GPT-5 계열은 reasoning 토큰이 많아 message가 비는 케이스가 있어
      reasoning.effort="low", summary="auto"를 넣고,
      빈 텍스트면 재시도 후 fallback_model로 1회 시도.
    """
    client = OpenAI(api_key=api_key)

    system = (
        "너는 근거 기반 요약/질의응답 도우미다. "
        "반드시 제공된 Evidence 안에서만 답하고, 근거가 없으면 'NOT_FOUND', 모르면 '응답 없음'이라고 말해라. "
        "반드시 마지막 줄에 '근거 페이지: p.xx, p.yy' 형식으로 페이지를 출력해라."
    )

    user = f"""
[질문/요청]
{query}

[Evidence]
{evidence}

[근거 페이지 후보]
{pages}
"""

    def _one_call(use_model: str, force_reasoning_low: bool) -> str:
        kwargs = dict(
            model=use_model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_output_tokens=max_completion_tokens,
        )

        m = (use_model or "").lower()

        # GPT-5 / reasoning 계열 안정화: reasoning 토큰 과다로 message가 비는 문제 완화
        if m.startswith("gpt-5"):
            kwargs["reasoning"] = {
                "effort": "low" if force_reasoning_low else "medium",
                "summary": "auto",
            }

        # temperature는 지원 모델에서만
        if _supports_temperature(use_model):
            kwargs["temperature"] = temperature

        resp = client.responses.create(**kwargs)
        return _extract_text(resp)

    # 1) 기본 호출 (gpt-5면 reasoning low로)
    text = _one_call(model, force_reasoning_low=True)

    # 2) 빈 응답이면 재시도
    tries = 0
    while (not text) and tries < max_retries:
        tries += 1
        text = _one_call(model, force_reasoning_low=True)

    # 3) 그래도 비면 fallback 모델 1회
    if not text and fallback_model:
        text = _one_call(fallback_model, force_reasoning_low=False)

    if not text:
        raise RuntimeError(
            "LLM이 텍스트 메시지를 생성하지 못했습니다. "
            "GPT-5 계열에서 reasoning만 생성되고 message가 없는 케이스일 수 있습니다. "
            "top_k/max_context_chars를 줄이거나 fallback 모델을 사용하세요."
        )

    return text