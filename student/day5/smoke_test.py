# -*- coding: utf-8 -*-
"""
Day5 build_index 단독 테스트
"""

import os
import sys
from pathlib import Path

# 루트 경로 설정
def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return start

ROOT = _find_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# .env 로드
ENV_PATH = ROOT / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH, override=False)
except:
    pass

# ───────── 테스트 시작 ─────────
print("=" * 60)
print("Day5 build_index 단독 테스트")
print("=" * 60)

# 1. 더미 데이터 생성
print("\n[1/4] 더미 공모전 데이터 생성...")
from student.day5.impl.build_index import build_corpus

corpus = build_corpus()
print(f"✅ 생성 완료: {len(corpus)}개 공모전")
print(f"   예시: {corpus[0]['text'][:100]}...")

# 2. 임베딩 생성
print("\n[2/4] 임베딩 생성...")
from student.day2.impl.embeddings import Embeddings

model = "text-embedding-3-small"
emb = Embeddings(model=model, batch_size=4)
texts = [item["text"] for item in corpus]
vectors = emb.encode(texts)
print(f"✅ 임베딩 완료: shape={vectors.shape}")

# 3. FAISS 인덱스 생성 및 저장
print("\n[3/4] FAISS 인덱스 생성...")
from student.day2.impl.store import FaissStore

index_dir = "indices/day5"
os.makedirs(index_dir, exist_ok=True)

index_path = os.path.join(index_dir, "faiss.index")
docs_path = os.path.join(index_dir, "docs.jsonl")

store = FaissStore(dim=vectors.shape[1], index_path=index_path, docs_path=docs_path)
store.add(vectors, corpus)
store.save()
print(f"✅ 인덱스 저장: {index_path}")

# 4. docs.jsonl 저장
print("\n[4/4] docs.jsonl 저장...")
from student.day5.impl.ingest import save_docs_jsonl

save_docs_jsonl(corpus, docs_path)
print(f"✅ 문서 저장: {docs_path}")

# 5. 검증
print("\n[검증] 저장된 파일 확인...")
idx_file = Path(index_path)
docs_file = Path(docs_path)

if idx_file.exists():
    print(f"✅ faiss.index: {idx_file.stat().st_size:,} bytes")
else:
    print(f"❌ faiss.index 없음")

if docs_file.exists():
    lines = docs_file.read_text().splitlines()
    print(f"✅ docs.jsonl: {len(lines)} 라인, {docs_file.stat().st_size:,} bytes")
else:
    print(f"❌ docs.jsonl 없음")

# 6. 로드 테스트
print("\n[로드 테스트] 저장된 인덱스 다시 로드...")
try:
    store_loaded = FaissStore.load(index_path, docs_path)
    print(f"✅ 로드 성공")
    
    # 간단한 검색 테스트
    query_vec = vectors[0]  # 첫 번째 공모전 벡터로 검색
    hits = store_loaded.search(query_vec, top_k=3)
    print(f"✅ 검색 테스트: {len(hits)}개 결과")
    for i, hit in enumerate(hits, 1):
        name = "알 수 없음"
        text = hit.get("text", "")
        if "[공모전명]:" in text:
            name = text.split("[공모전명]:")[1].split("\n")[0].strip()
        print(f"   {i}. [{hit.get('score', 0):.3f}] {name}")
    
except Exception as e:
    print(f"❌ 로드 실패: {e}")

print("\n" + "=" * 60)
print("[DONE] build_index 테스트 완료 ✅")
print("=" * 60)