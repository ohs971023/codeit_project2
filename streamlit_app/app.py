# streamlit_app/app.py
from __future__ import annotations

import json
import time
import re
import importlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd

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

paths = AppPaths()
cfg = AppConfig()

CACHE_DIR = paths.cache_dir
SERVICE_CFG_PATH = CACHE_DIR / "service_config.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FIXED_CONFIG_PY = paths.repo_root / "streamlit_app" / "core" / "fixed_config.py"

DEFAULT_CFG = {
    "pp_version": "pp_v5",          # ✅ 서비스 기본 pp_v5, 실험에서 바꾸면 서비스에도 적용
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
# Config IO (서비스 설정은 디스크에 저장: 서비스 재시작 후에도 유지)
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
# Chat: 세션 내 문서별 유지(재시작 시 초기화)
# - 디스크 저장 없음 (요구사항)
# =========================================================
def ensure_chat_state():
    if "doc_messages" not in st.session_state:
        st.session_state["doc_messages"] = {}  # {doc_id: [ {role, content}, ... ]}
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "active_doc" not in st.session_state:
        st.session_state["active_doc"] = None

def switch_doc(doc_id: str):
    """
    같은 세션에서 PDF를 바꿨다가 돌아오면 채팅 유지.
    Streamlit 재시작 시 session_state 초기화 → 채팅 초기화.
    """
    ensure_chat_state()
    prev = st.session_state.get("active_doc")

    if prev != doc_id:
        # 이전 문서의 채팅 저장
        if prev is not None:
            st.session_state["doc_messages"][prev] = st.session_state.get("messages", [])

        # 새 문서 채팅 복원 (없으면 빈 대화)
        st.session_state["messages"] = st.session_state["doc_messages"].get(doc_id, [])
        st.session_state["active_doc"] = doc_id

# =========================================================
# Retrieval helpers
# =========================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def provisional_pages_from_results(results: list, max_pages: int = 3) -> List[int]:
    pages = []
    for r in (results or [])[:5]:
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
    """
    예산 외 질문도 확정 hit가 나오도록:
    - 답변의 따옴표 안 구절
    - ':' 뒤 핵심
    - 근거 청크에서 키워드 포함 라인
    - 예산이면 큰 숫자도 보조
    """
    q = _norm(question)
    a = _norm(answer)
    lines = [x.strip() for x in (top_chunk_text or "").replace("\r","\n").split("\n") if x.strip()]
    out: List[str] = []

    # 1) 따옴표 안 구절
    quoted = re.findall(r'"([^"]+)"', a)
    for s in quoted[:2]:
        out.append(_norm(s)[:80])

    # 2) 콜론 뒤
    if ":" in a:
        out.append(_norm(a.split(":")[-1])[:80])

    # 3) 키워드 라인
    for kw in WEAK_KEYWORDS:
        if kw in q:
            for ln in lines:
                if kw in ln and len(ln) >= 8:
                    out.append(_norm(ln)[:80])
                    break
            break

    # 4) 예산이면 숫자 보조
    if any(k in q for k in ["예산", "사업비", "소요예산", "금액"]):
        nums = re.findall(r"\b\d[\d,]{4,}\b", a)
        if nums:
            n = nums[0]
            out.append(n)
            out.append(n.replace(",", ""))

    # fallback
    if not out and a:
        out.append(a[:60])

    # 중복 제거
    uniq = []
    seen = set()
    for x in out:
        x = _norm(x)
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:3]

def pick_pages_confirmed_candidate(
    pdf_path: str,
    strong_queries: List[str],
    weak_queries: List[str],
    cfg_obj: Dict[str, Any],
    fallback_pages: List[int],
) -> Tuple[List[int], List[int]]:
    """
    - confirmed: hit-count 상위 페이지
    - candidate: weak 키워드로 넓게 검색한 페이지
    - hit가 0이면 fallback_pages를 confirmed로 승격
    """
    confirmed_max = int(cfg_obj["confirmed_max"])
    candidate_max = int(cfg_obj["candidate_max"])
    max_pages_scan = int(cfg_obj["max_pages_scan"])

    confirmed: List[int] = []
    candidate: List[int] = []

    hit_map = {}
    if strong_queries:
        try:
            hit_map = find_pages_with_hit_counts(pdf_path, strong_queries, max_pages_scan=max_pages_scan)
        except Exception:
            hit_map = {}

    if hit_map:
        ranked = sorted(hit_map.items(), key=lambda kv: (-kv[1], kv[0]))
        confirmed = [p for p, _ in ranked[:confirmed_max]]
        candidate = [p for p, _ in ranked[confirmed_max:confirmed_max + candidate_max]]
    else:
        # fallback: meta page를 confirmed로
        confirmed = (fallback_pages or [])[:confirmed_max]
        if weak_queries:
            try:
                candidate = find_pages_for_queries(pdf_path, weak_queries, max_pages_scan=max_pages_scan)
            except Exception:
                candidate = []

    candidate = [p for p in candidate if p not in confirmed][:candidate_max]
    return confirmed, candidate

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="RAG Chat Service", layout="wide")
st.title("🤖 RAG 서비스 / 🧪 실험·평가")

pdf_map = make_pdf_map(paths.pdf_dir)
pdf_names = list(pdf_map.keys())
pp_versions = detect_available_pp_versions(paths) or ["pp_v5", "pp_v6", "pp_v4"]

tab_service, tab_exp_eval = st.tabs(["🟢 서비스(채팅)", "🧪 실험/평가(설정/평가)"])

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

        # ✅ 문서 바뀔 때 세션 내에서만 채팅 저장/복원 (재시작 시 초기화)
        switch_doc(doc_id)

        confirmed_pages = st.session_state.get("svc_confirmed_pages", [])
        candidate_pages = st.session_state.get("svc_candidate_pages", [])
        highlight_queries = st.session_state.get("svc_highlight_queries", [])

        st.markdown("**✅ 확정 근거 페이지(하이라이트 가능)**")
        st.write(confirmed_pages if confirmed_pages else [])

        st.markdown("**🟡 후보 페이지(하이라이트 없음)**")
        st.write(candidate_pages if candidate_pages else [])

        try:
            n_pages = get_pdf_num_pages(pdf_path)
        except Exception:
            n_pages = 1

        base_pages = confirmed_pages if confirmed_pages else candidate_pages
        default_page = int(base_pages[0]) if base_pages else 1
        page_to_view = st.number_input("미리보기 페이지", min_value=1, max_value=int(n_pages), value=int(default_page), step=1)

        show_hl = st.checkbox("하이라이트 보기(확정 페이지에서만)", value=True, disabled=(not confirmed_pages))

        if st.button("페이지 보기"):
            is_confirmed = page_to_view in (confirmed_pages or [])
            if show_hl and is_confirmed and highlight_queries:
                img, hits = render_pdf_page_png_with_highlights(pdf_path, int(page_to_view), highlight_queries, zoom=2.0)
                st.image(img, caption=f"p.{page_to_view} (highlights={hits})", use_container_width=True)
            else:
                img = render_pdf_page_png(pdf_path, int(page_to_view), zoom=2.0)
                st.image(img, caption=f"p.{page_to_view}", use_container_width=True)

    with col_chat:
        st.subheader("💬 채팅")

        # ✅ 현재 문서(messages) 표시
        for m in st.session_state.get("messages", []):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_msg = st.chat_input("질문을 입력하세요 (예: 사업명/예산/계약방식/기간/마감/기관/요구사항)")

        if user_msg:
            api_key = cfg.openai_api_key
            if not api_key:
                st.error("OPENAI_API_KEY가 없습니다. .env 확인 필요")
                st.stop()

            # user append (세션에만)
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
                            st.warning("전처리 생성 실패 → pdf_fallback으로 진행")
                            jsonl = None

                    source = "precomputed_chunks" if jsonl and jsonl.exists() else "pdf_fallback"
                    chunks, artifact_path = get_chunks(
                        doc_id=doc_id,
                        pdf_path=pdf_path,
                        paths=paths,
                        source=source,
                        precomputed_jsonl=(jsonl if source == "precomputed_chunks" else None),
                    )

                    retriever = build_or_load_hybrid(
                        chunks=chunks,
                        index_dir=paths.index_dir,
                        doc_id=doc_id,
                        source=f"{source}__{service_pp}",
                        artifact_path=artifact_path,
                    )
                    results = retriever.search(user_msg, k=int(cfg_service["top_k"]), alpha=float(cfg_service["alpha"]))

                    ev = evidence_text(results, max_chars=int(cfg_service["max_context_chars"]))
                    prov_pages = provisional_pages_from_results(results, max_pages=3)

                    answer = summarize_with_evidence(
                        api_key=api_key,
                        model="gpt-5-mini",
                        query=user_msg,
                        evidence=ev,
                        pages=prov_pages,  # ✅ 페이지 환각 방지
                        temperature=float(cfg_service["temperature"]),
                        max_completion_tokens=int(cfg_service["max_completion_tokens"]),
                    )
                    st.markdown(answer)

                    top_chunk_text = results[0].chunk.text if results else ""
                    strong_q = extract_strong_queries(user_msg, answer, top_chunk_text)
                    weak_q = [kw for kw in WEAK_KEYWORDS if kw in (user_msg + " " + answer)]

                    confirmed, candidates = pick_pages_confirmed_candidate(
                        pdf_path,
                        strong_q,
                        weak_q,
                        cfg_service,
                        fallback_pages=prov_pages,
                    )

                    st.session_state["svc_confirmed_pages"] = confirmed
                    st.session_state["svc_candidate_pages"] = candidates
                    st.session_state["svc_highlight_queries"] = strong_q if confirmed else []

                    # assistant append (세션에만)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})


# =========================================================
# 🧪 실험/평가(설정 변경 → fixed_config.py 덮어쓰기 + 서비스 반영)
# =========================================================
with tab_exp_eval:
    st.subheader("🧪 실험/평가 설정")
    st.caption("여기서 저장한 값이 서비스에 자동 반영되며, fixed_config.py도 자동으로 덮어씁니다.")

    cfg_now = load_cfg()

    colA, colB = st.columns(2)
    with colA:
        pp_version = st.selectbox("pp version(서비스에도 적용)", pp_versions, index=pp_versions.index(cfg_now["pp_version"]) if cfg_now["pp_version"] in pp_versions else 0)
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
            "chunk_length": int(chunk_length),
            "top_k": int(top_k),
            "max_tokens": 2000,
            "max_completion_tokens": int(max_completion_tokens),
            "temperature": float(temperature),
            "alpha": float(alpha),
            "max_context_chars": int(max_context_chars),
        }
        write_fixed_config_py(FIXED_CONFIG_PY, fixed_out)

        st.success("저장 완료! 서비스에 즉시 반영되고, fixed_config.py도 업데이트되었습니다.")

    st.divider()
    st.subheader("📊 평가 실행 (rag_experiment.py / ragas_eval.py 기반)")
    st.caption("평가에서 C1 chunk_length 등 CONFIG를 바꾸고 싶으면, 런타임에 rag_experiment.CONFIG를 덮어씁니다(파일 수정 없이).")

    eval_chunker = st.selectbox("chunker", ["C1", "C2", "C3", "C4"], index=0)
    eval_retriever = st.selectbox("retriever", ["R1", "R2", "R3"], index=2)
    eval_generator = st.selectbox("generator", ["G1", "G2"], index=0)
    eval_top_k = st.slider("eval top_k", 2, 30, int(top_k), 1)
    eval_sim_threshold = st.slider("eval sim_threshold", 50, 100, 80, 1)
    run_ragas = st.checkbox("RAGAS 실행", value=True)
    judge_model = st.selectbox("RAGAS Judge", ["gpt-5-mini", "gpt-5-nano"], index=0)

    if st.button("📊 평가 실행"):
        api_key = get_api_key()
        if not api_key:
            st.error("OPENAI_API_KEY가 비어있어요.")
            st.stop()

        with st.spinner("평가 실행 중..."):
            rag_experiment = cached_import("preprocess.rag_experiment")
            ragas_eval = cached_import("preprocess.ragas_eval")

            if hasattr(rag_experiment, "CONFIG") and isinstance(rag_experiment.CONFIG, dict):
                rag_experiment.CONFIG["chunk_length"] = int(chunk_length)

            questions_df = rag_experiment.load_questions_df()
            gold_evidence_df = pd.read_csv(paths.eval_dir / "gold_evidence.csv")
            gold_fields_df = ragas_eval.load_gold_fields_jsonl(paths.eval_dir / "gold_fields.jsonl")

            doc_names = sorted(set(questions_df.loc[questions_df["doc_id"] != "*", "doc_id"].astype(str).tolist()))
            run_docs = [paths.pdf_dir / name for name in doc_names if (paths.pdf_dir / name).exists()]

            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            embed_model = cached_embed_model("BAAI/bge-m3")

            spec = rag_experiment.ExperimentSpec(exp_id=0, chunker=eval_chunker, retriever=eval_retriever, generator=eval_generator)
            chunker_obj, retriever_obj, gen_obj = rag_experiment.make_components(spec, embed_model=embed_model, client=client)
            exp = rag_experiment.RAGExperiment(chunker_obj, retriever_obj, gen_obj, questions_df)

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
            st.dataframe(df, use_container_width=True)

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
                )
                st.dataframe(ragas_res.ragas_doc_df, use_container_width=True)