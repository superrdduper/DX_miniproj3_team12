# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
from typing import Dict, Any, List
import numpy as np

from student.common.schemas import Day5Plan
from .embeddings import Embeddings
from .store import FaissStore

def _idx_paths(index_dir: str):
    return (
        os.path.join(index_dir, "faiss.index"),
        os.path.join(index_dir, "docs.jsonl"),
    )

def _load_store(plan: Day5Plan, emb: Embeddings) -> FaissStore:
    index_path, docs_path = _idx_paths(plan.index_dir)
    if not (os.path.exists(index_path) and os.path.exists(docs_path)):
        raise FileNotFoundError(f"FAISS 인덱스가 없습니다. 먼저 ingest를 실행하세요: {plan.index_dir}")
    store = FaissStore.load(index_path, docs_path)
    # 차원 체크
    test_dim = emb.encode(["__dim_check__"]).shape[1]
    if store.dim != test_dim:
        raise ValueError(f"임베딩 차원이 인덱스와 다릅니다. (index={store.dim}, embedder={test_dim})")
    return store

def _gate(contexts: List[Dict[str, Any]], plan: Day5Plan) -> Dict[str, Any]:
    if not contexts:
        return {"status":"insufficient","top_score":0.0,"mean_topk":0.0}
    top_score = float(contexts[0]["score"])
    mean_topk = float(np.mean([c["score"] for c in contexts[:plan.top_k]]))
    if top_score >= plan.min_score and mean_topk >= plan.min_mean_topk:
        return {"status":"enough","top_score":top_score,"mean_topk":mean_topk}
    return {"status":"insufficient","top_score":top_score,"mean_topk":mean_topk}

def _draft_answer(query: str, contexts: List[Dict[str, Any]], plan: Day5Plan) -> str:
    """
    모든 검색결과의 공모전 메타데이터를 포함한 초안 생성
    """
    if not contexts:
        return "검색 결과가 없습니다."

    lines = []
    lines.append(f"🔎 '{query}' 검색 결과 {len(contexts)}개의 공모전을 찾았습니다.\n")
    lines.append("📋 전체 공모전 목록:\n")

    for i, c in enumerate(contexts, 1):
        # 메타 필드 가져오기
        f = (c.get("meta", {}).get("fields")) or {}
        if not f and isinstance(c.get("text"), str):
            try:
                f = json.loads(c["text"])
            except Exception:
                f = {}

        title = f.get("공모전명", f"공모전 #{i}")
        host = f.get("주최", "주최 미상")
        field = f.get("분야", "-")
        prize = f.get("상금(단위: 만 원)", "미정")
        deadline = f.get("마감일", "-")
        eligibility = f.get("참가 자격", "-")
        team_size = f.get("팀 규모", "-")
        preferred_major = f.get("전공 우대", "-")
        desc = (f.get("상세 내용", "") or "").strip()
        score = c.get("score", 0)

        # 상세 내용 일부만 (너무 길면 200자 제한)
        if len(desc) > 200:
            desc = desc[:200] + "…"

        lines.append(
            f"{i}. **{title}** ({field}) — {host}\n"
            f"   🏆 **상금:** {prize}만 원 | 🗓 **마감:** {deadline}\n"
            f"   👥 **참가 자격:** {eligibility} | 👤 **팀 규모:** {team_size}\n"
            f"   🎓 **전공 우대:** {preferred_major}\n"
            f"   💬 **상세 내용:** {desc}\n"
            f"   🔹 **매칭도:** {score*100:.1f}%\n"
        )

    # 전체 평균/TopK 정보
    top_score = float(contexts[0].get("score", 0.0))
    mean_score = float(np.mean([c.get("score", 0.0) for c in contexts[:plan.top_k]]))
    lines.append("\n📊 **검색 통계 요약**")
    lines.append(f"- 상위 1개 매칭도: {top_score*100:.1f}%")
    lines.append(f"- 상위 {plan.top_k} 평균 매칭도: {mean_score*100:.1f}%")

    return "\n".join(lines)


class Day5Agent:  # Day2Agent → Day5Agent
    def __init__(self, plan_defaults: Day5Plan = Day5Plan()):
        self.plan_defaults = plan_defaults

    def handle(self, query: str, plan: Day5Plan = None) -> Dict[str, Any]:
        plan = plan or self.plan_defaults
        emb = Embeddings(model=plan.embedding_model)

        store = _load_store(plan, emb)
        qv = emb.encode([query])[0]
        contexts = store.search(qv, top_k=plan.top_k)

        gate = _gate(contexts, plan)
        
        payload: Dict[str, Any] = {
            "type": "contest_recommendation",  # rag_answer → contest_recommendation
            "query": query,
            "plan": plan.__dict__,
            "contexts": contexts,
            "gating": gate,
            "answer": "",
            "stats": {  # 통계 정보 추가
                "total_results": len(contexts),
                "avg_score": float(np.mean([c["score"] for c in contexts])) if contexts else 0.0,
                "search_method": "rag_only" if plan.force_rag_only else "hybrid"
            }
        }
        
        if plan.force_rag_only or (gate["status"] == "enough" and plan.return_draft_when_enough):
            payload["answer"] = _draft_answer(query, contexts, plan)
        
        return payload