# -*- coding: utf-8 -*-
"""
Day2 인덱싱 엔트리포인트
- 목표: 코퍼스 생성 → 임베딩 → FAISS 저장 + docs.jsonl 저장
"""

import os, argparse, numpy as np
from typing import List
import pandas as pd
import time

from ingest import build_corpus, save_docs_jsonl
from embeddings import Embeddings
from store import FaissStore


def build_index(paths: List[str], index_dir: str, model: str | None = None, batch_size: int = 128):
    print("🚀 [START] 인덱싱 파이프라인 시작")

    corpus = build_corpus(paths)
    if len(corpus) == 0:
        raise ValueError("❌ 인덱싱할 문서가 없습니다.")

    texts = [item["text"] for item in corpus]
    print(f"📄 총 문서 수: {len(texts)}개")

    emb = Embeddings(model=model, batch_size=batch_size)
    print(f"🧠 임베딩 모델: {model or '기본값'}")

    # ⚙️ 임베딩 + 내용 확인
    vecs_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # ✅ 디버그: 각 문서 내용 일부 출력
        print(f"\n=== 🔹 Batch {i // batch_size + 1} / {len(texts) // batch_size + 1} ===")
        for j, t in enumerate(batch):
            # 너무 길면 앞부분만 보기 (100자 제한)
            snippet = (t[:120] + " ...") if len(t) > 120 else t
            print(f"📝 [Doc {i + j}] {snippet}")

        vecs_batch = emb.encode(batch)
        vecs_list.append(vecs_batch)
        print(f"✅ Batch {i + len(batch)}/{len(texts)} 임베딩 완료")

    vecs = np.vstack(vecs_list)
    print(f"✅ 전체 임베딩 완료! (shape={vecs.shape})")

    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "faiss.index")
    docs_path = os.path.join(index_dir, "docs.jsonl")

    store = FaissStore(dim=vecs.shape[1], index_path=index_path, docs_path=docs_path)
    store.add(vecs, corpus)
    store.save()

    save_docs_jsonl(corpus, docs_path)
    print(f"\n💾 인덱스 및 문서 저장 완료: {index_dir}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--index_dir", default="indices/day5")
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)

    build_index(
        paths=args.paths,
        index_dir=args.index_dir,
        model=args.model,
        batch_size=args.batch_size,
    )
