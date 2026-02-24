# streamlit_app/app.py
from __future__ import annotations

import sys
import json
import re
import time
import importlib
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from core.config import AppPaths, AppConfig
from core.config_io import write_fixed_config_py
from core.loaders import make_pdf_map, get_chunks
from core.preprocess_runner import detect_available_pp_versions, run_preprocessing
from core.retriever import build_or_load_hybrid, evidence_text
from core.llm import summarize_with_evidence
from core.render import (
    get_pdf_num_pages,
    render_pdf_page_png,
    render_pdf_page_png_with_highlights,
    find_pages_with_hit_counts,
    find_pages_for_queries,
)

# =========================================================
# Path bootstrap: notebooks on sys.path
# =========================================================
ROOT = Path(__file__).resolve().parents[1]     # .../streamlit_app -> repo root
NOTEBOOKS = ROOT / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))

# =========================================================
# Init
# =========================================================
paths = AppPaths()
cfg = AppConfig()

CACHE_DIR = paths.cache_dir
SERVICE_CFG_PATH = CACHE_DIR / "service_config.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FIXED_CONFIG_PY = paths.repo_root / "streamlit_app" / "core" / "fixed_config.py"

DEFAULT_CFG = {
    "pp_version": "pp_v5",
    "chunk_length": 1200,
    "top_k": 8,
    "alpha": 0.7,
    "max_context_chars": 2500,
    "max_completion_tokens": 800,
    "temperature": 0.1,
    "confirmed_max": 2,
    "candidate_max": 12,
    "max_pages_scan": 200,
}

WEAK_KEYWORDS = ["예산", "사업비", "소요예산", "부가가치세", "VAT", "계약", "입찰", "기간", "마감", "제출", "기관", "요구사항"]

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="RAG Chat Service", layout="wide")
st.title("🤖 RAG 서비스 / 🧪 실험·평가")

if not paths.pdf_dir.exists():
    st.error(f"PDF 폴더가 없습니다: {paths.pdf_dir}")
    st.stop()

pdf_map = make_pdf_map(paths.pdf_dir)
pdf_names = list(pdf_map.keys())
pp_versions = detect_available_pp_versions(paths) or ["pp_v5", "pp_v6", "pp_v4"]

tab_service, tab_exp_eval = st.tabs(["🟢 서비스(채팅)", "🧪 실험/평가(설정/평가)"])


# =========================================================
# module caches (평가용)
# =========================================================
@st.cache_resource(show_spinner=False)
def cached_import(module_name: str):
    return importlib.import_module(module_name)

@st.cache_resource(show_spinner=False)
def cached_embed_model(model_name: str = "BAAI/bge-m3"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def cached_openai_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)


# =========================================================
# Config IO
# =========================================================
def load_cfg() -> Dict[str, Any]:
    if SERVICE_CFG_PATH.exists():
        try:
            data = json.loads(SERVICE_CFG_PATH.read_text(encoding="utf-8"))
            return {**DEFAULT_CFG, **data}
        except Exception:
            return DEFAULT_CFG.copy()
    return DEFAULT_CFG.copy()

def save_cfg(cfg_obj: Dict[str, Any]) -> None:
    SERVICE_CFG_PATH.write_text(json.dumps(cfg_obj, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================================================
# 01_index_pages.json 로드 (목차 페이지 스킵용)
# =========================================================
@st.cache_data(show_spinner=False)
def load_index_pages(path_str: str) -> dict:
    p = Path(path_str)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)

_index_pages = load_index_pages(str(paths.index_pages_path))

def get_start_page(doc_id: str) -> int:
    info = _index_pages.get(doc_id, {})
    return max(1, int(info.get("start_page_label", 1)))

def get_index_page_labels(doc_id: str) -> List[int]:
    info = _index_pages.get(doc_id, {})
    return [int(p) for p in info.get("index_page_label", [])]

def get_front_pages(doc_id: str) -> List[tuple]:
    start = get_start_page(doc_id)
    if start <= 1:
        return []
    index_labels = set(get_index_page_labels(doc_id))
    result = []
    for p in range(1, start):
        if p == 1:
            label = "표지"
        elif p in index_labels:
            label = "목차"
        else:
            label = "앞부분"
        result.append((p, label))
    return result


# =========================================================
# Chat: 세션 내 문서별 유지(재시작 시 초기화)
# =========================================================
def ensure_chat_state():
    if "doc_messages" not in st.session_state:
        st.session_state["doc_messages"] = {}
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "active_doc" not in st.session_state:
        st.session_state["active_doc"] = None

def switch_doc(doc_id: str):
    ensure_chat_state()
    prev = st.session_state.get("active_doc")
    if prev != doc_id:
        if prev is not None:
            st.session_state["doc_messages"][prev] = st.session_state.get("messages", [])
        st.session_state["messages"] = st.session_state["doc_messages"].get(doc_id, [])
        st.session_state["active_doc"] = doc_id

        for k in [
            "svc_render_phys",
            "svc_confirmed_pages",
            "svc_candidate_pages",
            "svc_highlight_queries",
            "svc_pending_phys",
            "svc_pending_apply",
            "svc_pv_page_input",
            "svc_candidate_hl_once",
        ]:
            st.session_state.pop(k, None)


# =========================================================
# Retrieval helpers (서비스 탭)
# =========================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def clean_answer_remove_page_lines(answer: str) -> str:
    a = answer or ""
    a = re.sub(r"(?m)^\s*근거\s*페이지\s*:\s*.*$", "", a)
    a = re.sub(r"(?m)^\s*근거\s*페이지\s*.*$", "", a)
    a = re.sub(r"\n{3,}", "\n\n", a).strip()
    return a

def provisional_pages_from_results(results: list, max_pages: int = 3) -> List[int]:
    pages = []
    for r in (results or [])[:10]:
        try:
            p = int(getattr(r.chunk, "page", 0) or 0)
        except Exception:
            p = 0
        if p > 0 and p not in pages:
            pages.append(p)
        if len(pages) >= max_pages:
            break
    return pages

def extract_strong_queries(question: str, answer: str, top_chunk_text: str) -> List[str]:
    q = _norm(question)
    a = _norm(answer)
    lines = [x.strip() for x in (top_chunk_text or "").replace("\r", "\n").split("\n") if x.strip()]
    out: List[str] = []

    quoted = re.findall(r'"([^"]+)"', a)
    for s in quoted[:2]:
        out.append(_norm(s)[:80])

    if ":" in a:
        tail = _norm(a.split(":")[-1])[:80]
        if len(tail) >= 10:
            out.append(tail)

    for kw in WEAK_KEYWORDS:
        if kw in q:
            for ln in lines:
                if kw in ln and len(ln) >= 8:
                    out.append(_norm(ln)[:80])
                    break
            break

    if any(k in q for k in ["예산", "사업비", "소요예산", "금액"]):
        nums = re.findall(r"\b\d[\d,]{4,}\b", a)
        if nums:
            n = nums[0]
            out.append(n)
            out.append(n.replace(",", ""))

    if not out and a:
        out.append(a[:60])

    uniq = []
    seen = set()
    for x in out:
        x = _norm(x)
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:4]

def pick_pages_confirmed_candidate(
    pdf_path: str,
    strong_queries: List[str],
    weak_queries: List[str],
    cfg_obj: Dict[str, Any],
    fallback_pages: List[int],
    start_page: int = 1,
) -> Tuple[List[int], List[int]]:
    confirmed_max = int(cfg_obj["confirmed_max"])
    candidate_max = int(cfg_obj["candidate_max"])
    max_pages_scan = int(cfg_obj["max_pages_scan"])

    confirmed: List[int] = []
    candidate: List[int] = []

    hit_map = {}
    if strong_queries:
        try:
            hit_map = find_pages_with_hit_counts(
                pdf_path,
                strong_queries,
                start_page=start_page,
                max_pages_scan=max_pages_scan
            )
        except Exception:
            hit_map = {}

    if hit_map:
        ranked = sorted(hit_map.items(), key=lambda kv: (-kv[1], kv[0]))
        confirmed = [p for p, _ in ranked[:confirmed_max]]
        candidate = [p for p, _ in ranked[confirmed_max:confirmed_max + candidate_max]]
    else:
        confirmed = (fallback_pages or [])[:confirmed_max]
        if weak_queries:
            try:
                candidate = find_pages_for_queries(
                    pdf_path,
                    weak_queries,
                    start_page=start_page,
                    max_pages_scan=max_pages_scan
                )
            except Exception:
                candidate = []

    candidate = [p for p in candidate if p not in confirmed][:candidate_max]
    return confirmed, candidate

def sync_auto_navigate(target_phys: int, start_page: int, n_pages: int):
    target_phys = max(1, min(int(target_phys), int(n_pages)))
    st.session_state["svc_pending_phys"] = target_phys
    st.session_state["svc_pending_apply"] = True


# =========================================================
# Eval helpers: 표준화 + merge + 평균
# =========================================================
def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def standardize_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    colmap = {
        "ret_recall": ["ret_recall", "retrieval_recall@k", "retrieval_recall", "recall@k"],
        "ret_mrr": ["ret_mrr", "retrieval_mrr@k", "retrieval_mrr", "mrr@k"],
        "gen_fill": ["gen_fill", "gen_fill_rate", "generation_fill_rate"],
        "gen_match": ["gen_match", "gen_match_rate", "generation_match_rate"],
        "gen_sim": ["gen_sim", "gen_avg_similarity", "generation_avg_similarity"],
    }
    for std, cands in colmap.items():
        found = None
        for c in cands:
            if c in out.columns:
                found = c
                break
        if found and found != std:
            out.rename(columns={found: std}, inplace=True)
    for c in ["ret_recall", "ret_mrr", "gen_fill", "gen_match", "gen_sim"]:
        if c in out.columns:
            out[c] = _to_float_series(out[c])
    return out

def standardize_ragas_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    for c in ["faithfulness", "context_precision", "answer_correctness"]:
        if c in out.columns:
            out[c] = _to_float_series(out[c])
    return out

def merge_eval_and_ragas(doc_metrics_df: pd.DataFrame, ragas_doc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    left = standardize_eval_columns(doc_metrics_df.copy())
    if ragas_doc_df is None or len(ragas_doc_df) == 0:
        return left
    right = standardize_ragas_columns(ragas_doc_df.copy())
    if "doc_id" in left.columns and "doc_id" in right.columns:
        return left.merge(right, on="doc_id", how="left", suffixes=("", "_ragas"))
    return left

def make_mean_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    cols = [
        "ret_recall", "ret_mrr", "gen_fill", "gen_match", "gen_sim",
        "faithfulness", "context_precision", "answer_correctness",
    ]
    exist = [c for c in cols if c in df.columns]
    if not exist:
        return pd.DataFrame()
    means = df[exist].mean(numeric_only=True)
    return means.to_frame(name="mean").reset_index().rename(columns={"index": "metric"})


# =========================================================
# Monkeypatch: pp_v6 page=None 오류 런타임 해결 (파일 수정 X)
# =========================================================
def monkeypatch_pp_v6_table_title_guard():
    """
    preprocess.pp_v6._find_table_title 내부에서 page가 None일 때 int-NoneType 오류 방지.
    pp_v6가 존재할 때만 런타임 패치한다.
    """
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

        # backward scan
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

        # forward scan
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
# 🟢 서비스(채팅)
# =========================================================
with tab_service:
    cfg_service = load_cfg()

    col_chat, col_view = st.columns([1.15, 0.85], gap="large")

    with col_view:
        st.subheader("📄 PDF / 근거 보기")
        doc_id = st.selectbox("문서 선택", pdf_names, key="svc_doc_select")
        pdf_path = str(pdf_map[doc_id])

        switch_doc(doc_id)

        confirmed_pages = st.session_state.get("svc_confirmed_pages", []) or []
        candidate_pages = st.session_state.get("svc_candidate_pages", []) or []
        highlight_queries = st.session_state.get("svc_highlight_queries", []) or []

        start_page = get_start_page(doc_id)
        front_pages = get_front_pages(doc_id)
        offset = start_page - 1

        try:
            n_pages = get_pdf_num_pages(pdf_path)
        except Exception:
            n_pages = 1

        allow_candidate_hl = st.checkbox("후보 페이지 클릭 시 하이라이트 1회 켜기", value=False, key="svc_allow_candidate_hl")

        if st.session_state.get("svc_pending_apply"):
            target_phys = int(st.session_state.get("svc_pending_phys", start_page))
            target_phys = max(1, min(target_phys, int(n_pages)))
            st.session_state["svc_render_phys"] = target_phys

            content_max = max(1, n_pages - offset)
            content_p = target_phys - offset
            content_p = max(1, min(content_p, content_max))
            st.session_state["svc_pv_page_input"] = int(content_p)
            st.session_state["svc_pending_apply"] = False

        show_hl = st.checkbox("하이라이트 보기(확정 페이지에서만)", value=True, disabled=(not confirmed_pages))

        if front_pages:
            st.caption("앞부분 바로 보기")
            btn_cols = st.columns(len(front_pages))
            for col, (phys_p, lbl) in zip(btn_cols, front_pages):
                if col.button(lbl, key=f"svc_front_{phys_p}"):
                    sync_auto_navigate(phys_p, start_page, n_pages)
                    st.session_state["svc_candidate_hl_once"] = False
                    st.rerun()

        content_max = max(1, n_pages - offset)

        if "svc_pv_page_input" in st.session_state:
            default_content_p = int(st.session_state["svc_pv_page_input"])
        elif confirmed_pages:
            default_content_p = max(1, min(int(confirmed_pages[0]) - offset, content_max))
        elif candidate_pages:
            default_content_p = max(1, min(int(candidate_pages[0]) - offset, content_max))
        else:
            default_content_p = 1

        default_content_p = max(1, min(default_content_p, content_max))

        def _svc_page_changed():
            target_phys = int(st.session_state["svc_pv_page_input"]) + offset
            st.session_state["svc_render_phys"] = int(target_phys)
            st.session_state["svc_candidate_hl_once"] = False

        st.number_input(
            f"본문 페이지 (1..{content_max})",
            min_value=1,
            max_value=content_max,
            value=default_content_p,
            step=1,
            key="svc_pv_page_input",
            on_change=_svc_page_changed,
        )

        page_to_view = st.session_state.get("svc_render_phys")
        if page_to_view is None:
            page_to_view = int(st.session_state.get("svc_pv_page_input", default_content_p)) + offset
            st.session_state["svc_render_phys"] = int(page_to_view)
        page_to_view = int(page_to_view)

        is_confirmed = page_to_view in confirmed_pages
        is_candidate = page_to_view in candidate_pages
        is_front = page_to_view < start_page

        if is_front:
            front_label = next((lbl for p, lbl in front_pages if p == page_to_view), "앞부분")
            caption = f"{front_label} (물리 p.{page_to_view})"
        else:
            caption = f"p.{page_to_view - offset}"

        cand_once = bool(st.session_state.get("svc_candidate_hl_once", False))
        allow_cand_render = bool(allow_candidate_hl and is_candidate and cand_once)

        try:
            if (show_hl and is_confirmed and highlight_queries) or (allow_cand_render and highlight_queries):
                img, hits = render_pdf_page_png_with_highlights(pdf_path, page_to_view, highlight_queries, zoom=2.0)
                st.image(img, caption=f"{caption} (highlights={hits})", use_container_width=True)
                if allow_cand_render:
                    st.session_state["svc_candidate_hl_once"] = False
            else:
                img = render_pdf_page_png(pdf_path, page_to_view, zoom=2.0)
                st.image(img, caption=caption, use_container_width=True)
        except Exception as e:
            st.error("페이지 렌더링 실패")
            st.exception(e)

        if candidate_pages:
            cand_content = [max(1, p - offset) for p in candidate_pages if p >= start_page]
            st.markdown("**🟡 후보 페이지(클릭해서 이동)**")

            row = []
            max_show = int(cfg_service["candidate_max"])
            show_list = cand_content[:max_show]
            for idx, cp in enumerate(show_list):
                row.append(cp)
                if len(row) == 6 or idx == len(show_list) - 1:
                    cols = st.columns(len(row))
                    for c, page_num in zip(cols, row):
                        if c.button(f"{page_num}", key=f"cand_{doc_id}_{page_num}_{idx}"):
                            phys = int(page_num) + offset
                            sync_auto_navigate(phys, start_page, n_pages)
                            st.session_state["svc_candidate_hl_once"] = bool(allow_candidate_hl)
                            st.rerun()
                    row = []
        else:
            st.caption("후보 페이지 없음")

    with col_chat:
        st.subheader("💬 채팅")
        for m in st.session_state.get("messages", []):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_msg = st.chat_input("질문을 입력하세요 (예: 사업명/예산/계약방식/기간/마감/기관/요구사항)")

        if user_msg:
            api_key = cfg.openai_api_key
            if not api_key:
                st.error("OPENAI_API_KEY가 없습니다. .env 확인 필요")
                st.stop()

            st.session_state["messages"].append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            with st.chat_message("assistant"):
                with st.spinner("검색/답변 생성 중..."):
                    service_pp = str(cfg_service.get("pp_version", "pp_v5"))

                    jsonl = paths.chunks_dir / service_pp / f"{doc_id}.jsonl"
                    if not jsonl.exists():
                        ok, msg = run_preprocessing(service_pp, doc_id, int(cfg_service["chunk_length"]), paths)
                        if not ok:
                            st.error("전처리 생성 실패 → 서비스 실행을 중단합니다.")
                            st.write(msg)
                            st.stop()

                    chunks, artifact_path = get_chunks(
                        doc_id=doc_id,
                        pdf_path=pdf_path,
                        paths=paths,
                        source="precomputed_chunks",
                        precomputed_jsonl=jsonl,
                    )

                    retriever = build_or_load_hybrid(
                        chunks=chunks,
                        index_dir=paths.index_dir,
                        doc_id=doc_id,
                        source=f"precomputed__{service_pp}",
                        artifact_path=artifact_path,
                    )

                    results = retriever.search(user_msg, k=int(cfg_service["top_k"]), alpha=float(cfg_service["alpha"]))
                    ev = evidence_text(results, max_chars=int(cfg_service["max_context_chars"]))
                    prov_pages = provisional_pages_from_results(results, max_pages=3)

                    raw_answer = summarize_with_evidence(
                        api_key=api_key,
                        model="gpt-5-mini",
                        query=user_msg,
                        evidence=ev,
                        pages=prov_pages,
                        temperature=float(cfg_service["temperature"]),
                        max_completion_tokens=int(cfg_service["max_completion_tokens"]),
                    )

                    answer = clean_answer_remove_page_lines(raw_answer)
                    st.markdown(answer)

                    top_chunk_text = results[0].chunk.text if results else ""
                    strong_q = extract_strong_queries(user_msg, raw_answer, top_chunk_text)
                    weak_q = [kw for kw in WEAK_KEYWORDS if kw in (user_msg + " " + raw_answer)]

                    confirmed, candidates = pick_pages_confirmed_candidate(
                        pdf_path,
                        strong_q,
                        weak_q,
                        cfg_service,
                        fallback_pages=prov_pages,
                        start_page=get_start_page(doc_id),
                    )

                    st.session_state["svc_confirmed_pages"] = confirmed
                    st.session_state["svc_candidate_pages"] = candidates
                    st.session_state["svc_highlight_queries"] = strong_q if confirmed else []

                    target = None
                    if confirmed:
                        target = confirmed[0]
                    elif candidates:
                        target = candidates[0]
                    elif prov_pages:
                        target = prov_pages[0]

                    if target is not None:
                        sync_auto_navigate(int(target), get_start_page(doc_id), n_pages)
                        st.session_state["svc_candidate_hl_once"] = False

                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    st.rerun()


# =========================================================
# 🧪 실험/평가(설정 변경 + 평가 실행)
# =========================================================
with tab_exp_eval:
    st.subheader("🧪 실험/평가 설정")
    st.caption("여기서 저장한 값이 서비스에 자동 반영되며, fixed_config.py도 자동으로 덮어씁니다.")

    cfg_now = load_cfg()

    colA, colB = st.columns(2)
    with colA:
        pp_version = st.selectbox(
            "pp version(서비스에도 적용)",
            pp_versions,
            index=pp_versions.index(cfg_now["pp_version"]) if cfg_now["pp_version"] in pp_versions else 0
        )
        chunk_length = st.slider("chunk_length", 300, 2000, int(cfg_now["chunk_length"]), 100)
        top_k = st.slider("top_k", 2, 30, int(cfg_now["top_k"]), 1)
        alpha = st.slider("alpha(hybrid)", 0.0, 1.0, float(cfg_now["alpha"]), 0.05)
        max_context_chars = st.slider("max_context_chars", 500, 8000, int(cfg_now["max_context_chars"]), 100)

    with colB:
        max_completion_tokens = st.slider("max_completion_tokens", 128, 4096, int(cfg_now["max_completion_tokens"]), 64)
        temperature = st.slider("temperature", 0.0, 1.5, float(cfg_now["temperature"]), 0.05)
        confirmed_max = st.slider("confirmed_max", 1, 4, int(cfg_now["confirmed_max"]), 1)
        candidate_max = st.slider("candidate_max", 5, 30, int(cfg_now["candidate_max"]), 1)
        max_pages_scan = st.slider("max_pages_scan", 10, 400, int(cfg_now["max_pages_scan"]), 10)

    if st.button("✅ 저장 → 서비스 반영 + fixed_config.py 덮어쓰기"):
        cfg_new = {
            "pp_version": str(pp_version),
            "chunk_length": int(chunk_length),
            "top_k": int(top_k),
            "alpha": float(alpha),
            "max_context_chars": int(max_context_chars),
            "max_completion_tokens": int(max_completion_tokens),
            "temperature": float(temperature),
            "confirmed_max": int(confirmed_max),
            "candidate_max": int(candidate_max),
            "max_pages_scan": int(max_pages_scan),
        }
        save_cfg(cfg_new)

        fixed_out = {
            "pp_version": str(pp_version),
            "chunk_length": int(chunk_length),
            "top_k": int(top_k),
            "max_tokens": 2000,
            "max_completion_tokens": int(max_completion_tokens),
            "temperature": float(temperature),
            "alpha": float(alpha),
            "max_context_chars": int(max_context_chars),
        }
        write_fixed_config_py(FIXED_CONFIG_PY, fixed_out)
        st.success("저장 완료! 서비스 반영 + fixed_config.py 업데이트 완료")

    st.divider()
    st.subheader("📊 평가 실행 (rag_experiment.py / ragas_eval.py 기반)")
    st.caption("옵션 A: questions.csv에 포함된 doc_id만 평가합니다(파일 존재하는 것만).")

    eval_chunker = st.selectbox("chunker", ["C1", "C2", "C3", "C4"], index=0, key="eval_chunker")
    eval_retriever = st.selectbox("retriever", ["R1", "R2", "R3"], index=2, key="eval_retriever")
    eval_generator = st.selectbox("generator", ["G1", "G2"], index=0, key="eval_generator")
    eval_top_k = st.slider("eval top_k", 2, 30, int(top_k), 1, key="eval_topk")
    eval_sim_threshold = st.slider("eval sim_threshold", 50, 100, 80, 1, key="eval_sim")
    run_ragas = st.checkbox("RAGAS 실행", value=True, key="eval_ragas")
    judge_model = st.selectbox("RAGAS Judge", ["gpt-5-mini", "gpt-5-nano"], index=0, key="eval_judge")

    if st.button("📊 평가 실행", key="eval_run_btn"):
        api_key = cfg.openai_api_key
        if not api_key:
            st.error("OPENAI_API_KEY가 비어있어요.")
            st.stop()

        with st.status("평가 실행 중...", expanded=True) as status:
            t0 = time.time()
            try:
                # ✅ pp_v6 page=None 등 런타임 패치
                monkeypatch_pp_v6_table_title_guard()

                rag_experiment = cached_import("preprocess.rag_experiment")
                ragas_eval = cached_import("preprocess.ragas_eval")

                # (선택) rag_experiment CONFIG 반영 (기존 코드 유지)
                if hasattr(rag_experiment, "CONFIG") and isinstance(rag_experiment.CONFIG, dict):
                    rag_experiment.CONFIG["chunk_length"] = int(chunk_length)

                questions_df = rag_experiment.load_questions_df()
                gold_evidence_df = pd.read_csv(paths.eval_dir / "gold_evidence.csv")
                gold_fields_df = ragas_eval.load_gold_fields_jsonl(paths.eval_dir / "gold_fields.jsonl")

                # ✅ 옵션 A: questions.csv에 포함된 doc_id만 평가
                doc_names = sorted(set(
                    questions_df.loc[questions_df["doc_id"] != "*", "doc_id"].astype(str).tolist()
                ))

                missing = [name for name in doc_names if not (paths.pdf_dir / name).exists()]
                run_docs = [paths.pdf_dir / name for name in doc_names if (paths.pdf_dir / name).exists()]

                if not run_docs:
                    raise RuntimeError(
                        f"평가 문서가 없습니다. questions.csv doc_id 와 data/raw/files 매칭 확인\n"
                        f"missing sample={missing[:8]}"
                    )

                if missing:
                    st.warning(f"questions.csv에 있지만 PDF가 없는 문서가 {len(missing)}개 있습니다.")
                    st.caption("예: " + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))

                client = cached_openai_client(api_key)
                embed_model = cached_embed_model("BAAI/bge-m3")

                spec = rag_experiment.ExperimentSpec(
                    exp_id=0,
                    chunker=str(eval_chunker),
                    retriever=str(eval_retriever),
                    generator=str(eval_generator),
                )

                chunker_obj, retriever_obj, gen_obj = rag_experiment.make_components(
                    spec, embed_model=embed_model, client=client
                )
                exp = rag_experiment.RAGExperiment(chunker_obj, retriever_obj, gen_obj, questions_df)

                rows: List[Dict[str, Any]] = []
                failed: List[Dict[str, Any]] = []

                for dp in run_docs:
                    try:
                        m = exp.run_single_doc_metrics(
                            doc_path=Path(dp),
                            gold_fields_df=gold_fields_df,
                            gold_evidence_df=gold_evidence_df,
                            top_k=int(eval_top_k),
                            sim_threshold=int(eval_sim_threshold),
                        )
                        # 실험 스펙 태깅
                        m["chunker"] = spec.chunker
                        m["retriever"] = spec.retriever
                        m["generator"] = spec.generator
                        rows.append(m)
                    except Exception as e:
                        failed.append({"doc_id": dp.name, "error": repr(e)})

                if not rows:
                    raise RuntimeError(
                        "모든 문서 평가가 실패해서 결과가 비었습니다.\n"
                        f"실패 {len(failed)}개. failed[0]={failed[0] if failed else None}"
                    )

                doc_metrics_df = pd.DataFrame(rows)
                doc_metrics_df = standardize_eval_columns(doc_metrics_df)

                ragas_doc_df = None
                ragas_exp_df = None

                if run_ragas:
                    ragas_res = ragas_eval.run_experiment_with_ragas(
                        spec=spec,
                        run_docs=run_docs,
                        gold_fields_jsonl_path=paths.eval_dir / "gold_fields.jsonl",
                        embed_model=embed_model,
                        client=client,
                        evaluator_model=judge_model,
                        ragas_metrics=["faithfulness", "context_precision", "answer_correctness"],
                        compute_baseline_doc_metrics=False,
                        gold_evidence_df=gold_evidence_df,
                        sim_threshold=int(eval_sim_threshold),
                        # 아래 두 파라미터는 네 위 코드에 맞춰 옵션으로 둠(없으면 무시될 수 있음)
                        judge_max_context_chars_per_sample=int(max_context_chars),
                        judge_max_output_tokens=500,
                        judge_reasoning_effort="minimal",
                    )
                    ragas_doc_df = getattr(ragas_res, "ragas_doc_df", None)
                    ragas_exp_df = getattr(ragas_res, "ragas_exp_df", None)

                merged_df = merge_eval_and_ragas(doc_metrics_df, ragas_doc_df)
                mean_df = make_mean_table(merged_df)

                st.session_state["eval_doc_metrics_df"] = doc_metrics_df
                st.session_state["eval_ragas_doc_df"] = ragas_doc_df
                st.session_state["eval_ragas_exp_df"] = ragas_exp_df
                st.session_state["eval_merged_df"] = merged_df
                st.session_state["eval_mean_df"] = mean_df
                st.session_state["eval_failed_docs"] = pd.DataFrame(failed) if failed else pd.DataFrame()

                status.update(label=f"평가 완료 ✅ ({time.time()-t0:.2f}s)", state="complete")

            except Exception as e:
                status.update(label="평가 실패 ❌", state="error")
                st.error("평가 실행 중 오류")
                st.exception(e)

    st.divider()
    st.subheader("✅ 평가 결과 (문서별 + 평균)")

    failed_df = st.session_state.get("eval_failed_docs")
    if failed_df is not None and len(failed_df) > 0:
        st.warning(f"실패 문서 {len(failed_df)}개")
        st.dataframe(failed_df, use_container_width=True)

    merged_df = st.session_state.get("eval_merged_df")
    if merged_df is not None:
        preferred = [
            "doc_id",
            "ret_recall", "ret_mrr", "gen_fill", "gen_match", "gen_sim",
            "faithfulness", "context_precision", "answer_correctness",
            "chunker", "retriever", "generator",
        ]
        cols = [c for c in preferred if c in merged_df.columns] + [c for c in merged_df.columns if c not in preferred]
        st.dataframe(merged_df[cols], use_container_width=True)
    else:
        st.caption("아직 평가 결과가 없습니다.")

    st.subheader("📌 전체 평균표")
    mean_df = st.session_state.get("eval_mean_df")
    if mean_df is not None and len(mean_df) > 0:
        st.dataframe(mean_df, use_container_width=True)
    else:
        st.caption("평균표 없음(미실행 또는 지표 컬럼 없음).")

    ragas_exp_df = st.session_state.get("eval_ragas_exp_df")
    if ragas_exp_df is not None and isinstance(ragas_exp_df, pd.DataFrame) and len(ragas_exp_df) > 0:
        st.subheader("참고: RAGAS exp 평균")
        st.dataframe(ragas_exp_df, use_container_width=True)