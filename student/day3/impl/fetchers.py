# -*- coding: utf-8 -*-
"""
정부/공공 포털 및 일반 웹에서 '사업 공고'를 찾기 위한 검색 래퍼 (디버깅 출력 추가)
"""

from typing import List, Dict, Any
import os, sys, time
from datetime import datetime
from student.day1.impl.tavily_client import search_tavily

DEFAULT_TOPK = 7
DEFAULT_TIMEOUT = 20

# 기본 TopK(권장): NIPA 3, Bizinfo 2, Web 2
NIPA_TOPK = 3
BIZINFO_TOPK = 2
WEB_TOPK = 2


# ===============================================================
# 🔹 간단한 로그 출력 함수 (디버그/추적용)
# ===============================================================
def log(msg: str, level: str = "INFO"):
    """시간 + 레벨 + 메시지 출력"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", file=sys.stderr, flush=True)


def trace_start(func_name: str, query: str):
    log(f"🚀 {func_name}() 시작 — query='{query}'", "DEBUG")


def trace_end(func_name: str, count: int, elapsed_ms: float):
    log(f"✅ {func_name}() 완료 — 결과 {count}건 | {elapsed_ms:.0f}ms", "DEBUG")


def trace_error(func_name: str, e: Exception):
    log(f"❌ {func_name}() 오류 — {type(e).__name__}: {e}", "ERROR")


# ===============================================================
# 🔸 검색 함수들
# ===============================================================
def fetch_nipa(query: str, topk: int = NIPA_TOPK) -> List[Dict[str, Any]]:
    func_name = "fetch_nipa"
    trace_start(func_name, query)
    start = time.time()

    try:
        key = os.getenv("TAVILY_API_KEY", "")
        q = f"{query} 공고 모집 지원 site:nipa.kr"

        results = search_tavily(
            q, key,
            top_k=topk,
            timeout=DEFAULT_TIMEOUT,
            include_domains=["nipa.kr"],
        )

        trace_end(func_name, len(results), (time.time() - start) * 1000)
        return results

    except Exception as e:
        trace_error(func_name, e)
        return []


def fetch_bizinfo(query: str, topk: int = BIZINFO_TOPK) -> List[Dict[str, Any]]:
    func_name = "fetch_bizinfo"
    trace_start(func_name, query)
    start = time.time()

    try:
        key = os.getenv("TAVILY_API_KEY", "")
        q = f"{query} 공고 모집 지원 site:bizinfo.go.kr"

        results = search_tavily(
            q, key,
            top_k=topk,
            timeout=DEFAULT_TIMEOUT,
            include_domains=["bizinfo.go.kr"],
        )

        trace_end(func_name, len(results), (time.time() - start) * 1000)
        return results

    except Exception as e:
        trace_error(func_name, e)
        return []


def fetch_web(query: str, topk: int = WEB_TOPK) -> List[Dict[str, Any]]:
    func_name = "fetch_web"
    trace_start(func_name, query)
    start = time.time()

    try:
        api_key = os.getenv("TAVILY_API_KEY", "")
        search_query = f"{query} 모집 공고 지원 사업"

        results = search_tavily(
            search_query,
            api_key,
            top_k=topk,
            timeout=DEFAULT_TIMEOUT
        )

        trace_end(func_name, len(results), (time.time() - start) * 1000)
        return results

    except Exception as e:
        trace_error(func_name, e)
        return []


# ===============================================================
# 🔸 통합 호출 (fetch_all)
# ===============================================================
def fetch_all(query: str) -> List[Dict[str, Any]]:
    func_name = "fetch_all"
    trace_start(func_name, query)
    start = time.time()

    all_results: List[Dict[str, Any]] = []

    try:
        all_results.extend(fetch_nipa(query))
    except Exception as e:
        trace_error("fetch_nipa (in fetch_all)", e)

    try:
        all_results.extend(fetch_bizinfo(query))
    except Exception as e:
        trace_error("fetch_bizinfo (in fetch_all)", e)

    try:
        all_results.extend(fetch_web(query))
    except Exception as e:
        trace_error("fetch_web (in fetch_all)", e)

    trace_end(func_name, len(all_results), (time.time() - start) * 1000)
    return all_results
