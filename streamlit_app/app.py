# streamlit_app/app.py
from __future__ import annotations

import time
import importlib
import inspect
import re
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Dict

import pandas as pd
import streamlit as st

from core.config import AppPaths, AppConfig
from core.fixed_config import CONFIG as FIXED
from core.runtime_imports import ensure_notebooks_on_syspath
from core.preprocess_runner import detect_available_pp_versions, run_preprocessing
from core.loaders import make_pdf_map, get_chunks
from core.retriever import build_or_load_hybrid, evidence_text
from core.llm import summarize_with_evidence
from core.render import (
    render_pdf_page_png,
    render_pdf_page_png_with_highlights,
    get_pdf_num_pages,
    find_pages_for_queries,
    find_pages_with_hit_counts,
)

# =========================================================
# Init
# =========================================================
paths = AppPaths()
cfg = AppConfig()
ensure_notebooks_on_syspath(paths)

st.set_page_config(page_title="RAG 플랫폼", layout="wide")
st.title("🚀 RAG 플랫폼 (서비스 / 실험+평가)")

if not paths.pdf_dir.exists():
    st.error(f"PDF 폴더가 없습니다: {paths.pdf_dir}")
    st.stop()

# =========================================================
# Cache
# =========================================================
@st.cache_data(show_spinner=False)
def cached_pdf_map(pdf_dir_str: str):
    return make_pdf_map(Path(pdf_dir_str))

@st.cache_data(show_spinner=False)
def cached_pp_versions():
    return detect_available_pp_versions(paths)

@st.cache_resource(show_spinner=False)
def cached_import(module_name: str):
    return importlib.import_module(module_name)

@st.cache_resource(show_spinner=False)
def cached_openai_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)

@st.cache_resource(show_spinner=False)
def cached_embed_model(model_name: str = "BAAI/bge-m3"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

pdf_map = cached_pdf_map(str(paths.pdf_dir))
pdf_names = ["(선택)"] + list(pdf_map.keys())
pp_versions = cached_pp_versions()

# =========================================================
# Helpers
# =========================================================
def get_api_key() -> str:
    k = (st.session_state.get("sb_api_key") or "").strip()
    if k:
        return k
    return (cfg.openai_api_key or "").strip()

def safe_call(func: Callable[..., Any], **kwargs) -> Any:
    sig = inspect.signature(func)
    call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**call_kwargs)

def set_state(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v

def get_state(k, default=None):
    return st.session_state.get(k, default)

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

# =========================================================
# 확정/후보 근거 로직 + 질문 맞춤 하이라이트
# =========================================================

WEAK_KEYWORDS = ["소요예산", "사업비", "예산", "부가가치세", "VAT", "부가세", "원"]
STRONG_PHRASES = ["협상에 의한 계약", "제한경쟁입찰", "총 소요예산", "소요예산:"]

def classify_question(q: str) -> str:
    q = _norm_ws(q)
    if any(k in q for k in ["예산", "사업비", "소요예산", "금액", "원"]):
        return "budget"
    if any(k in q for k in ["계약", "계약방식", "입찰", "협상", "제한경쟁", "일반경쟁", "수의계약"]):
        return "contract"
    if any(k in q for k in ["기간", "수행기간", "사업기간", "일로부터", "일", "개월"]):
        return "duration"
    if any(k in q for k in ["마감", "제출", "접수", "제안서", "마감일", "제출일시", "제출기한"]):
        return "deadline"
    if any(k in q for k in ["기관", "수요기관", "발주", "발주처", "주관", "담당"]):
        return "agency"
    if any(k in q for k in ["요구", "요구사항", "필수", "기능", "성능", "보안", "범위", "과업"]):
        return "requirements"
    return "general"

def _lines(text: str) -> list[str]:
    t = (text or "").replace("\r", "\n")
    t = re.sub(r"\n{2,}", "\n", t)
    ls = [re.sub(r"\s+", " ", x).strip() for x in t.split("\n")]
    return [x for x in ls if x]

def extract_big_numbers(text: str) -> List[str]:
    t = _norm_ws(text)
    return re.findall(r"\b\d[\d,]{4,}\b", t)

def number_variants(num: str) -> List[str]:
    n = num.strip()
    digits = re.sub(r"[^\d]", "", n)
    vars_ = []
    if n:
        vars_.append(n)
        vars_.append(n.replace(",", ", "))
        vars_.append(n.replace(",", " "))
        if digits:
            vars_.append(digits)
            vars_.append(digits + "원")
        if not n.endswith("원"):
            vars_.append(n + "원")
            vars_.append(n.replace(",", "") + "원")
    out = []
    seen = set()
    for v in vars_:
        vv = _norm_ws(v)
        if vv and vv not in seen:
            seen.add(vv)
            out.append(vv)
    return out

def make_weak_queries(answer: str, results: list, max_queries: int = 4) -> List[str]:
    t = _norm_ws(answer)
    if results:
        t += " " + _norm_ws(results[0].chunk.text)
    weak = []
    for kw in WEAK_KEYWORDS:
        if kw in t:
            weak.append(kw)
    out = []
    seen = set()
    for q in weak:
        qn = _norm_ws(q)
        if qn and qn not in seen:
            seen.add(qn)
            out.append(qn)
    return out[:max_queries]

def pick_best_highlight_snippet(question: str, answer: str, results: list, max_len: int = 80) -> list[str]:
    qtype = classify_question(question)
    q = _norm_ws(question)
    a = _norm_ws(answer)

    kw_groups = {
        "budget": ["소요예산", "사업비", "예산", "부가가치세", "VAT"],
        "contract": ["협상에 의한 계약", "제한경쟁", "일반경쟁", "수의계약", "입찰", "계약방식"],
        "duration": ["계약일로부터", "사업기간", "수행기간", "일", "개월"],
        "deadline": ["제출", "마감", "접수", "제출일시", "제출기한"],
        "agency": ["발주", "수요기관", "발주기관", "기관"],
        "requirements": ["필수", "요구사항", "기능", "성능", "보안", "범위", "과업"],
        "general": [],
    }
    kws = kw_groups.get(qtype, [])

    # 1) 근거 청크의 '해당 키워드 라인' 우선
    best_lines = []
    for r in (results or [])[:3]:
        txt = getattr(r.chunk, "text", "")
        for line in _lines(txt):
            if len(line) < 8:
                continue
            if any(kw in line for kw in kws):
                best_lines.append(line)
        if best_lines:
            break

    # 2) 없으면 질문 토큰 일부 포함 라인
    if not best_lines and results:
        q_tokens = [t for t in re.split(r"\s+", q) if len(t) >= 2]
        for r in results[:3]:
            for line in _lines(getattr(r.chunk, "text", "")):
                if any(t in line for t in q_tokens[:4]):
                    best_lines.append(line)
            if best_lines:
                break

    queries: List[str] = []
    if best_lines:
        best_lines = sorted(best_lines, key=lambda x: (-min(len(x), 120), x))
        for line in best_lines[:2]:
            queries.append(line[:max_len])
    else:
        # 3) 답변 구절
        if a:
            cand = a.split(":")[-1].strip() if ":" in a else a
            queries.append(cand[:max_len] if len(cand) > max_len else cand)

    # 4) 숫자/날짜는 예산/기간/마감에서만 보조
    if qtype in ("budget", "duration", "deadline"):
        nums = extract_big_numbers(a)
        if nums:
            queries.extend(number_variants(nums[0])[:2])
        dates = re.findall(r"\b20\d{2}[./-]\d{1,2}[./-]\d{1,2}\b", a)
        if dates:
            queries.append(dates[0])

    # 중복 제거
    out = []
    seen = set()
    for q in queries:
        qn = _norm_ws(q)
        if qn and qn not in seen:
            seen.add(qn)
            out.append(qn)
    return out[:3]

def corrected_pages_and_queries_strict(
    pdf_path: str,
    results: list,
    answer: str,
    question: str,
    *,
    max_pages_scan: int = 200,
    confirmed_max: int = 2,
    candidate_max: int = 12,
) -> Tuple[List[int], List[int], List[str]]:
    """
    ✅ hit-count 기반 확정 페이지 생성
    - strong_q로 페이지별 hit 수를 계산해서 상위 페이지를 확정(confirmed)
    - 약한 키워드(weak)로는 후보만(candidates)
    - 하이라이트는 확정 페이지에서만
    """
    if not results:
        return [], [], []

    strong_q = pick_best_highlight_snippet(question, answer, results, max_len=80)

    hit_map = find_pages_with_hit_counts(pdf_path, strong_q, max_pages_scan=max_pages_scan) if strong_q else {}
    confirmed_pages: List[int] = []
    candidate_pages: List[int] = []

    if hit_map:
        ranked = sorted(hit_map.items(), key=lambda kv: (-kv[1], kv[0]))
        confirmed_pages = [p for p, _ in ranked[:confirmed_max]]
        candidate_pages = [p for p, _ in ranked[confirmed_max:confirmed_max + candidate_max]]
    else:
        # strong_q로 hit가 하나도 없으면 확정은 비우고 후보만
        weak_q = make_weak_queries(answer, results)
        if weak_q:
            candidate_pages = find_pages_for_queries(pdf_path, weak_q, max_pages_scan=max_pages_scan)
        if len(candidate_pages) > candidate_max:
            candidate_pages = candidate_pages[:candidate_max]

    if not confirmed_pages and not candidate_pages:
        # 마지막 fallback: meta page는 후보로만
        fallback = []
        for r in results[:3]:
            try:
                p = int(getattr(r.chunk, "page", 0) or 0)
            except Exception:
                p = 0
            if p > 0:
                fallback.append(p)
        candidate_pages = sorted(set(fallback))[:candidate_max]

    highlight_queries = strong_q if confirmed_pages else []
    return confirmed_pages, candidate_pages, highlight_queries


# =========================================================
# 페이지 보기 UI (확정/후보 분리 + 확정만 하이라이트)
# =========================================================
def page_view_block_confirmed_candidate(
    pdf_path: str,
    confirmed_pages: List[int],
    candidate_pages: List[int],
    highlight_queries: List[str],
    prefix: str,
):
    if not pdf_path:
        st.caption("pdf_path 없음")
        return

    try:
        n_pages = get_pdf_num_pages(pdf_path)
    except Exception as e:
        st.error("PDF 페이지 수 읽기 실패")
        st.exception(e)
        return

    confirmed_pages = [p for p in confirmed_pages if isinstance(p, int) and 1 <= p <= n_pages]
    confirmed_pages = sorted(set(confirmed_pages))
    candidate_pages = [p for p in candidate_pages if isinstance(p, int) and 1 <= p <= n_pages]
    candidate_pages = sorted(set(candidate_pages))

    st.markdown("### 근거 페이지")
    st.markdown("**✅ 확정 근거 페이지(하이라이트 가능)**")
    st.write(confirmed_pages if confirmed_pages else [])
    st.markdown("**🟡 후보 페이지(하이라이트 없음)**")
    st.write(candidate_pages if candidate_pages else [])

    base_pages = confirmed_pages if confirmed_pages else candidate_pages
    if not base_pages:
        st.caption("표시할 페이지가 없습니다.")
        return

    default_p = base_pages[0]
    pv_key = f"{prefix}_pv_page"
    btn_key = f"{prefix}_pv_btn"
    state_key = f"{prefix}_page_to_render"
    hl_toggle_key = f"{prefix}_hl_toggle"

    can_highlight = bool(confirmed_pages) and bool(highlight_queries)
    st.checkbox("하이라이트 보기(확정 페이지에서만)", value=True, key=hl_toggle_key, disabled=(not can_highlight))

    pv_page = st.number_input(
        f"미리보기 페이지 (1..{n_pages})",
        min_value=1,
        max_value=n_pages,
        value=int(default_p),
        step=1,
        key=pv_key,
    )

    if st.button("페이지 보기", key=btn_key):
        st.session_state[state_key] = int(pv_page)

    page_to_render = st.session_state.get(state_key)
    if page_to_render is None:
        return

    use_hl = bool(st.session_state.get(hl_toggle_key, True))
    is_confirmed_page = page_to_render in confirmed_pages

    try:
        if can_highlight and use_hl and is_confirmed_page:
            img, hits = render_pdf_page_png_with_highlights(
                pdf_path,
                int(page_to_render),
                highlight_queries,
                zoom=2.0,
            )
            st.image(img, caption=f"p.{int(page_to_render)} (highlights={hits})", use_container_width=True)
        else:
            img = render_pdf_page_png(pdf_path, int(page_to_render), zoom=2.0)
            tag = "candidate/no-highlight" if (not is_confirmed_page) else "no-highlight"
            st.image(img, caption=f"p.{int(page_to_render)} ({tag})", use_container_width=True)
    except Exception as e:
        st.error("페이지 렌더링 실패")
        st.exception(e)


# =========================================================
# Monkeypatch: pp_v6 page=None 오류를 파일 수정 없이 해결
# =========================================================
def monkeypatch_pp_v6_table_title_guard():
    try:
        ppv6 = cached_import("preprocess.pp_v6")
    except Exception:
        return

    def _safe_int(x, default: int = 0) -> int:
        try:
            if x is None:
                return default
            return int(x)
        except Exception:
            return default

    if not hasattr(ppv6, "_find_table_title"):
        return

    def _find_table_title_safe(items, anchor_idx, page):
        page_i = _safe_int(page, default=0)

        def _is_table_like_caption(text: str) -> bool:
            t = (text or "").strip()
            return bool(t) and (("표" in t) or ("별표" in t) or ("총괄표" in t) or ("목록표" in t))

        for i in range(anchor_idx - 1, max(-1, anchor_idx - 40), -1):
            meta = items[i].get("metadata", {}) or {}
            mp = _safe_int(meta.get("page", -999), default=-999)
            if abs(mp - page_i) > 1:
                continue
            if meta.get("type") == "table_row":
                continue
            cand = (items[i].get("content", "") or "").strip()
            if _is_table_like_caption(cand):
                return cand

        for i in range(anchor_idx + 1, min(len(items), anchor_idx + 15)):
            meta = items[i].get("metadata", {}) or {}
            mp = _safe_int(meta.get("page", -999), default=-999)
            if abs(mp - page_i) > 1:
                continue
            if meta.get("type") == "table_row":
                continue
            cand = (items[i].get("content", "") or "").strip()
            if _is_table_like_caption(cand):
                return cand

        return ""

    ppv6._find_table_title = _find_table_title_safe


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("공통 설정")
    st.text_input(
        "OPENAI_API_KEY (.env 자동 로드 / 필요시 입력)",
        value=cfg.openai_api_key or "",
        type="password",
        key="sb_api_key",
    )
    st.divider()
    st.header("전처리 버전")
    if not pp_versions:
        st.warning("notebooks/preprocess에 pp_v*.py가 없어요.")
    st.selectbox(
        "pp version",
        pp_versions if pp_versions else ["pp_v6"],
        index=(len(pp_versions) - 1) if pp_versions else 0,
        key="sb_ppver",
    )


# =========================================================
# Tabs
# =========================================================
tab_service, tab_exp_eval = st.tabs(["🟢 서비스(고정)", "🧪 실험+📊 평가(합침)"])


# =========================================================
# 🟢 서비스 탭
# =========================================================
with tab_service:
    st.subheader("서비스 모드 (값 고정)")
    st.caption("hit-count 기반 확정 페이지 + 질문 맞춤 하이라이트 + 확정에서만 하이라이트")

    colL, colR = st.columns([1.2, 1.0])
    with colL:
        svc_doc = st.selectbox("문서 선택", pdf_names, key="svc_doc")
        svc_model = st.selectbox("모델", ["gpt-5-mini", "gpt-5-nano"], key="svc_model")
        svc_q = st.text_area("질문", "사업명은 무엇인가?", key="svc_q")
        auto_pp = st.checkbox("전처리 결과 없으면 자동 생성", value=True, key="svc_auto_pp")
        svc_run = st.button("▶️ 실행", type="primary", key="svc_run_btn")
    with colR:
        st.markdown("### 고정 CONFIG")
        st.json(FIXED)

    if svc_run:
        if svc_doc == "(선택)":
            st.warning("문서를 선택해줘.")
            st.stop()

        api_key = get_api_key()
        if not api_key:
            st.error("OPENAI_API_KEY가 비어있어요.")
            st.stop()

        doc_id = svc_doc
        pdf_path = str(pdf_map[svc_doc])

        top_k = int(FIXED["top_k"])
        max_context_chars = int(FIXED["max_context_chars"])
        max_completion_tokens = int(FIXED["max_completion_tokens"])
        temperature = float(FIXED["temperature"])
        alpha = float(FIXED["alpha"])
        chunk_length = int(FIXED["chunk_length"])

        version = st.session_state.get("sb_ppver") or (pp_versions[-1] if pp_versions else "pp_v6")
        precomputed = paths.chunks_dir / version / f"{doc_id}.jsonl"

        with st.status("서비스 실행 중…", expanded=True) as status:
            t0 = time.time()
            try:
                if auto_pp and not precomputed.exists():
                    ok, msg = run_preprocessing(version=version, doc_id=doc_id, size=chunk_length, paths=paths)
                    if ok:
                        st.success(msg)
                    else:
                        st.warning("전처리 자동 생성 실패 → pdf_fallback으로 진행합니다.")
                        st.write(msg)

                source = "precomputed_chunks" if precomputed.exists() else "pdf_fallback"

                chunks, artifact_path = get_chunks(
                    doc_id=doc_id,
                    pdf_path=pdf_path,
                    paths=paths,
                    source=source,
                    precomputed_jsonl=precomputed if source == "precomputed_chunks" else None,
                )

                retriever = build_or_load_hybrid(
                    chunks=chunks,
                    index_dir=paths.index_dir,
                    doc_id=doc_id,
                    source=f"{source}__{version}",
                    artifact_path=artifact_path,
                )

                results = retriever.search(svc_q, k=top_k, alpha=alpha)
                ev = evidence_text(results, max_chars=max_context_chars)

                answer = summarize_with_evidence(
                    api_key=api_key,
                    model=svc_model,
                    query=svc_q,
                    evidence=ev,
                    pages=[],
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                )

                confirmed_pages, candidate_pages, hl_queries = corrected_pages_and_queries_strict(
                    pdf_path,
                    results,
                    answer=answer,
                    question=svc_q,
                    max_pages_scan=200,
                    confirmed_max=2,
                    candidate_max=12,
                )

                set_state(
                    svc_last_answer=answer,
                    svc_last_confirmed_pages=confirmed_pages,
                    svc_last_candidate_pages=candidate_pages,
                    svc_last_results=results,
                    svc_last_pdf_path=pdf_path,
                    svc_last_hl_queries=hl_queries,
                )

                status.update(label=f"완료 ✅ (총 {time.time()-t0:.2f}s)", state="complete")
            except Exception as e:
                status.update(label="서비스 실패 ❌", state="error")
                st.error("서비스 실행 중 오류")
                st.exception(e)

    st.divider()
    st.markdown("## 결과")

    svc_answer = get_state("svc_last_answer", "")
    svc_confirmed = get_state("svc_last_confirmed_pages", []) or []
    svc_candidates = get_state("svc_last_candidate_pages", []) or []
    svc_pdf_path = get_state("svc_last_pdf_path", "")
    svc_hl_queries = get_state("svc_last_hl_queries", []) or []

    st.markdown("### 답변")
    st.write(svc_answer if svc_answer else "아직 답변이 없습니다.")

    page_view_block_confirmed_candidate(
        svc_pdf_path,
        svc_confirmed,
        svc_candidates,
        svc_hl_queries,
        prefix="svc",
    )


# =========================================================
# 🧪 실험+📊 평가 탭
# =========================================================
with tab_exp_eval:
    st.subheader("실험 + 평가 (한 화면)")

    # ------------------ 실험(단일) ------------------
    st.markdown("## 🧪 실험 (단일 문서)")
    exp_col1, exp_col2 = st.columns([1.2, 1.0])
    with exp_col1:
        exp_doc = st.selectbox("문서 선택", pdf_names, key="exp_doc")
        exp_q = st.text_area("질문", "계약방식은 무엇인가?", key="exp_q")
        exp_run = st.button("▶️ 실험 실행", type="primary", key="exp_run_btn")
    with exp_col2:
        exp_model = st.selectbox("모델", ["gpt-5-mini", "gpt-5-nano"], key="exp_model")
        exp_version = st.session_state.get("sb_ppver") or (pp_versions[-1] if pp_versions else "pp_v6")
        st.caption(f"전처리 버전: {exp_version}")
        exp_chunk = st.slider("chunk_length(전처리 size)", 300, 2000, int(FIXED["chunk_length"]), 100, key="exp_chunk")
        exp_pp = st.button("🚀 전처리 실행", key="exp_pp_btn")
        exp_top_k = st.slider("top_k", 2, 30, int(FIXED["top_k"]), 1, key="exp_topk")
        exp_ctx = st.slider("max_context_chars", 300, 8000, int(FIXED["max_context_chars"]), 100, key="exp_ctx")
        exp_alpha = st.slider("alpha", 0.0, 1.0, float(FIXED["alpha"]), 0.05, key="exp_alpha")
        exp_temp = st.slider("temperature", 0.0, 1.5, float(FIXED["temperature"]), 0.05, key="exp_temp")
        exp_mct = st.slider("max_completion_tokens", 128, 4096, int(FIXED["max_completion_tokens"]), 64, key="exp_mct")

    if exp_pp:
        if exp_doc == "(선택)":
            st.warning("문서를 선택해줘.")
        else:
            ok, msg = run_preprocessing(version=exp_version, doc_id=exp_doc, size=int(exp_chunk), paths=paths)
            st.success(msg) if ok else st.error(msg)

    if exp_run:
        if exp_doc == "(선택)":
            st.warning("문서를 선택해줘.")
            st.stop()

        api_key = get_api_key()
        if not api_key:
            st.error("OPENAI_API_KEY가 비어있어요.")
            st.stop()

        doc_id = exp_doc
        pdf_path = str(pdf_map[exp_doc])
        precomputed = paths.chunks_dir / exp_version / f"{doc_id}.jsonl"
        source = "precomputed_chunks" if precomputed.exists() else "pdf_fallback"

        with st.status("실험 실행 중…", expanded=True) as status:
            t0 = time.time()
            try:
                chunks, artifact_path = get_chunks(
                    doc_id=doc_id,
                    pdf_path=pdf_path,
                    paths=paths,
                    source=source,
                    precomputed_jsonl=precomputed if source == "precomputed_chunks" else None,
                )
                retriever = build_or_load_hybrid(
                    chunks=chunks,
                    index_dir=paths.index_dir,
                    doc_id=doc_id,
                    source=f"{source}__{exp_version}",
                    artifact_path=artifact_path,
                )
                results = retriever.search(exp_q, k=int(exp_top_k), alpha=float(exp_alpha))
                ev = evidence_text(results, max_chars=int(exp_ctx))

                answer = summarize_with_evidence(
                    api_key=api_key,
                    model=exp_model,
                    query=exp_q,
                    evidence=ev,
                    pages=[],
                    temperature=float(exp_temp),
                    max_completion_tokens=int(exp_mct),
                )

                confirmed_pages, candidate_pages, hl_queries = corrected_pages_and_queries_strict(
                    pdf_path,
                    results,
                    answer=answer,
                    question=exp_q,
                    max_pages_scan=200,
                    confirmed_max=2,
                    candidate_max=12,
                )

                set_state(
                    exp_last_answer=answer,
                    exp_last_confirmed_pages=confirmed_pages,
                    exp_last_candidate_pages=candidate_pages,
                    exp_last_pdf_path=pdf_path,
                    exp_last_hl_queries=hl_queries,
                )
                status.update(label=f"완료 ✅ (총 {time.time()-t0:.2f}s)", state="complete")
            except Exception as e:
                status.update(label="실험 실패 ❌", state="error")
                st.error("실험 실행 중 오류")
                st.exception(e)

    st.markdown("### 실험 결과")
    exp_answer = get_state("exp_last_answer", "")
    exp_confirmed = get_state("exp_last_confirmed_pages", []) or []
    exp_candidates = get_state("exp_last_candidate_pages", []) or []
    exp_pdf_path = get_state("exp_last_pdf_path", "")
    exp_hl_queries = get_state("exp_last_hl_queries", []) or []

    st.write(exp_answer if exp_answer else "아직 실험 결과가 없습니다.")
    page_view_block_confirmed_candidate(exp_pdf_path, exp_confirmed, exp_candidates, exp_hl_queries, prefix="exp")

    # ------------------ 평가(30문서) ------------------
    st.divider()
    st.markdown("## 📊 평가 (30문서 배치)")

    api_key = get_api_key()
    judge_model = st.selectbox("RAGAS Judge 모델", ["gpt-5-mini", "gpt-5-nano"], key="eval_judge_model")
    eval_top_k = st.slider("eval top_k", 2, 30, 20, 1, key="eval_topk")
    eval_sim_threshold = st.slider("eval sim_threshold", 50, 100, 80, 1, key="eval_sim")
    run_ragas = st.checkbox("RAGAS도 같이 실행", value=True, key="eval_run_ragas")

    st.markdown("### 평가 스펙 선택 (기존 요소)")
    eval_chunker = st.selectbox("chunker", ["C1", "C2", "C3", "C4"], index=2, key="eval_chunker")
    eval_retriever = st.selectbox("retriever", ["R1", "R2", "R3"], index=2, key="eval_retriever")
    eval_generator = st.selectbox("generator", ["G1", "G2"], index=0, key="eval_generator")

    eval_run = st.button("📊 평가 실행", type="primary", key="eval_run_btn")

    if eval_run:
        if not api_key:
            st.error("OPENAI_API_KEY가 비어있어요.")
            st.stop()

        with st.status("평가 실행 중…", expanded=True) as status:
            try:
                monkeypatch_pp_v6_table_title_guard()
                rag_experiment = cached_import("preprocess.rag_experiment")
                ragas_eval = cached_import("preprocess.ragas_eval")

                questions_df = rag_experiment.load_questions_df()
                gold_evidence_df = pd.read_csv(paths.eval_dir / "gold_evidence.csv")
                gold_fields_df = ragas_eval.load_gold_fields_jsonl(paths.eval_dir / "gold_fields.jsonl")

                doc_names = sorted(set(questions_df.loc[questions_df["doc_id"] != "*", "doc_id"].astype(str).tolist()))
                run_docs: List[Path] = [paths.pdf_dir / name for name in doc_names if (paths.pdf_dir / name).exists()]
                if not run_docs:
                    raise RuntimeError("평가 문서가 없습니다. questions.csv doc_id 와 data/raw/files 매칭 확인")

                spec = rag_experiment.ExperimentSpec(
                    exp_id=0,
                    chunker=str(eval_chunker),
                    retriever=str(eval_retriever),
                    generator=str(eval_generator),
                )

                client = cached_openai_client(api_key)
                embed_model = cached_embed_model("BAAI/bge-m3")

                chunker, retriever, gen = rag_experiment.make_components(spec, embed_model=embed_model, client=client)
                exp = rag_experiment.RAGExperiment(chunker, retriever, gen, questions_df)

                rows = []
                for dp in run_docs:
                    m = exp.run_single_doc_metrics(
                        doc_path=Path(dp),
                        gold_fields_df=gold_fields_df,
                        gold_evidence_df=gold_evidence_df,
                        top_k=int(eval_top_k),
                        sim_threshold=int(eval_sim_threshold),
                    )
                    rows.append(m)

                df = pd.DataFrame(rows)
                mean_df = df.mean(numeric_only=True).to_frame("mean").reset_index().rename(columns={"index": "metric"})
                set_state(eval_df=df, eval_mean_df=mean_df)

                status.update(label="평가 완료 ✅", state="complete")
            except Exception as e:
                status.update(label="평가 실패 ❌", state="error")
                st.error("평가 실행 중 오류")
                st.exception(e)

    st.divider()
    st.markdown("## 평가 결과")
    df = get_state("eval_df")
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.caption("아직 평가 결과가 없습니다.")

    st.markdown("### 평균표 (전체 평균)")
    mean_df = get_state("eval_mean_df")
    if mean_df is not None and len(mean_df) > 0:
        st.dataframe(mean_df, use_container_width=True)
    else:
        st.caption("평균표 없음(미실행 또는 지표 컬럼 없음).")