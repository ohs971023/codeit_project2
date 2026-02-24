from __future__ import annotations

import re
from typing import List, Tuple, Dict

import fitz  # pymupdf


def get_pdf_num_pages(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    return int(n)


def render_pdf_page_png(pdf_path: str, page_1indexed: int, zoom: float = 2.0) -> bytes:
    doc = fitz.open(pdf_path)
    n = doc.page_count
    if page_1indexed < 1 or page_1indexed > n:
        doc.close()
        raise ValueError(f"page out of range: {page_1indexed} (valid: 1..{n})")
    page = doc.load_page(page_1indexed - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    b = pix.tobytes("png")
    doc.close()
    return b


def _normalize_for_search(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def render_pdf_page_png_with_highlights(
    pdf_path: str,
    page_1indexed: int,
    highlight_queries: List[str],
    zoom: float = 2.0,
    max_hits_per_query: int = 25,
) -> Tuple[bytes, int]:
    """
    특정 페이지에서 highlight_queries를 찾아 하이라이트 주석을 올리고 PNG로 렌더.
    반환: (png_bytes, total_hits)
    """
    doc = fitz.open(pdf_path)
    n = doc.page_count
    if page_1indexed < 1 or page_1indexed > n:
        doc.close()
        raise ValueError(f"page out of range: {page_1indexed} (valid: 1..{n})")

    page = doc.load_page(page_1indexed - 1)

    total_hits = 0
    for q in highlight_queries:
        qn = _normalize_for_search(q)
        if not qn:
            continue

        rects = page.search_for(qn)
        if not rects and len(qn) > 40:
            rects = page.search_for(qn[:40])

        if rects:
            rects = rects[:max_hits_per_query]
            total_hits += len(rects)
            for r in rects:
                annot = page.add_highlight_annot(r)
                annot.update()

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes, total_hits


def find_pages_for_queries(
    pdf_path: str,
    queries: List[str],
    *,
    start_page: int = 1,
    max_pages_scan: int = 200,
) -> List[int]:
    """
    PDF를 스캔해서 queries 중 하나라도 hit가 있는 페이지 목록(1-indexed)을 반환.
    start_page: 스캔 시작 물리적 페이지(1-indexed). 목차 등 앞 페이지를 제외할 때 사용.
    """
    if not queries:
        return []

    doc = fitz.open(pdf_path)
    n = doc.page_count
    p0_start = max(0, start_page - 1)
    limit = min(n, p0_start + max_pages_scan)

    pages: List[int] = []
    for p0 in range(p0_start, limit):
        page = doc.load_page(p0)
        hit_any = False
        for q in queries:
            qn = _normalize_for_search(q)
            if not qn:
                continue
            rects = page.search_for(qn)
            if not rects and len(qn) > 40:
                rects = page.search_for(qn[:40])
            if rects:
                hit_any = True
                break
        if hit_any:
            pages.append(p0 + 1)

    doc.close()
    return pages


def find_pages_with_hit_counts(
    pdf_path: str,
    queries: List[str],
    *,
    start_page: int = 1,
    max_pages_scan: int = 200,
    max_hits_per_query: int = 100,
) -> Dict[int, int]:
    """
    PDF를 스캔해서 queries hit 개수를 페이지별로 합산해 반환.
    반환: {page_1indexed: total_hits}
    - start_page: 스캔 시작 물리적 페이지(1-indexed). 목차 등 앞 페이지를 제외할 때 사용.
    - 여러 query의 hit를 합산
    - hit가 0인 페이지는 포함하지 않음
    """
    if not queries:
        return {}

    doc = fitz.open(pdf_path)
    n = doc.page_count
    p0_start = max(0, start_page - 1)
    limit = min(n, p0_start + max_pages_scan)

    hit_map: Dict[int, int] = {}
    for p0 in range(p0_start, limit):
        page = doc.load_page(p0)
        total = 0
        for q in queries:
            qn = _normalize_for_search(q)
            if not qn:
                continue
            rects = page.search_for(qn)
            if not rects and len(qn) > 40:
                rects = page.search_for(qn[:40])
            if rects:
                total += min(len(rects), max_hits_per_query)
        if total > 0:
            hit_map[p0 + 1] = total

    doc.close()
    return hit_map