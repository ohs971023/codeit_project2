# streamlit_app/core/fixed_config.py
CONFIG = {
    "chunk_length": 1200,          # 전처리 chunk export에 사용
    "top_k": 20,
    "max_tokens": 2000,            # non gpt-5 (현재 미사용 가능)
    "max_completion_tokens": 2000, # gpt-5
    "temperature": 0.1,            # non gpt-5 (gpt-5는 미지원일 수 있음)
    "alpha": 0.7,                  # hybrid weight (추후 하이브리드 붙일 때)
    "max_context_chars": 4000,     # context hard cap (chars)
}