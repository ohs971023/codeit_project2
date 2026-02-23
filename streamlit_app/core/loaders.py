from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re

import pdfplumber

from .config import AppPaths
from .text_clean import post_clean_text


@dataclass
class Chunk:
    doc_id: str
    page: int
    chunk_id: str
    text: str
    section_path: Optional[str] = None


def _norm_doc_id(s: str) -> str:
    s = str(s).strip().strip('"').strip()
    s = re.sub(r"\s+", " ", s)
    return s


def make_pdf_map(pdf_dir: Path) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    for p in sorted(pdf_dir.glob("*.pdf")):
        m[_norm_doc_id(p.name)] = p
    return m


def load_chunks_from_jsonl(jsonl_path: Path, doc_id: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            page = obj.get("page", obj.get("page_label", 0))
            try:
                page = int(float(page))
            except Exception:
                page = 0

            text = obj.get("text", obj.get("content", "")) or ""
            text = post_clean_text(str(text))

            if page <= 0 or not text:
                continue

            chunk_id = str(obj.get("chunk_id", f"{Path(doc_id).stem}__p{page:04d}_c{i:04d}"))
            section_path = obj.get("section_path") or obj.get("section") or None

            chunks.append(Chunk(
                doc_id=doc_id,
                page=page,
                chunk_id=chunk_id,
                text=text,
                section_path=section_path,
            ))
    return chunks


def extract_page_texts_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            txt = post_clean_text(txt)
            if txt:
                out.append((pidx, txt))
    return out


def get_chunks(
    doc_id: str,
    pdf_path: Optional[str],
    paths: AppPaths,
    source: str,
    *,
    precomputed_jsonl: Optional[Path] = None,
) -> Tuple[List[Chunk], Path]:
    """
    source:
      - precomputed_chunks : precomputed_jsonl를 반드시 넘기는 걸 권장
      - pdf_fallback
    Returns: (chunks, artifact_path)
    """
    doc_id_n = _norm_doc_id(doc_id)

    if source == "precomputed_chunks":
        if precomputed_jsonl is None:
            raise ValueError("precomputed_chunks source에는 precomputed_jsonl 경로를 넘겨야 합니다.")
        if not precomputed_jsonl.exists():
            raise FileNotFoundError(f"precomputed jsonl not found: {precomputed_jsonl}")
        chunks = load_chunks_from_jsonl(precomputed_jsonl, doc_id_n)
        return chunks, precomputed_jsonl

    if source == "pdf_fallback":
        if not pdf_path:
            raise FileNotFoundError("pdf_path가 없어서 pdf_fallback으로 로드할 수 없습니다.")
        page_texts = extract_page_texts_from_pdf(pdf_path)
        chunks = [
            Chunk(doc_id=doc_id_n, page=p, chunk_id=f"{Path(doc_id_n).stem}__p{p:04d}_pagechunk", text=t)
            for p, t in page_texts
        ]
        return chunks, Path(pdf_path)

    raise ValueError(f"Unknown source: {source}")