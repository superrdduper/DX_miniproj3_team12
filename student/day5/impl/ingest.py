# -*- coding: utf-8 -*-
"""
인덱싱 입력 데이터 로딩/정제/청크
"""

import re, json
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd 

def read_text_file(path: str) -> str:
    """
    안전한 텍스트 로드(utf-8, errors='ignore')
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-G-01] 구현 지침
    #  - with open(path, "r", encoding="utf-8", errors="ignore") as f: return f.read()
    # ----------------------------------------------------------------------------
    # 정답 구현:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(path: str) -> str:
    """
    pypdf 로 PDF 모든 페이지 텍스트 추출
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-G-02] 구현 지침
    #  - from pypdf import PdfReader
    #  - reader = PdfReader(path); texts=[]
    #  - for page in reader.pages: texts.append(page.extract_text() or "")
    #  - return "\n".join(texts)
    # ----------------------------------------------------------------------------
    # 정답 구현:
    from pypdf import PdfReader  # type: ignore
    reader = PdfReader(path)
    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)


def clean_text(s: str) -> str:
    """
    과도한 공백/개행/컨트롤 문자 정제
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-G-03] 구현 지침
    #  - s = s or ""
    #  - s = re.sub(r"\r", "\n", s)
    #  - s = re.sub(r"[ \t]+", " ", s)
    #  - s = re.sub(r"\n{3,}", "\n\n", s)
    #  - return s.strip()
    # ----------------------------------------------------------------------------
    # 정답 구현:
    s = s or ""
    s = re.sub(r"\r", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    """
    슬라이딩 윈도우로 청크 분할.
    - 길이가 chunk_size 이하이면 그대로 1청크
    - 그 외에는 overlap 적용하여 분할
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-G-04] 구현 지침
    #  - if len(text) <= chunk_size: return [text]
    #  - chunks=[]; start=0
    #  - while start < len(text):
    #       end = min(len(text), start+chunk_size)
    #       chunks.append(text[start:end])
    #       start += (chunk_size - chunk_overlap)
    #  - return chunks
    # ----------------------------------------------------------------------------
    # 정답 구현:
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks


def load_documents(paths_or_dir: List[str]) -> List[Dict[str, Any]]:
    """
    입력 경로(디렉토리/파일)에서 txt/md/pdf 수집 → [{"path":..., "text":...}, ...]
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-G-05] 구현 지침
    #  - files=[]
    #  - for p in paths_or_dir:
    #       pp=Path(p)
    #       if pp.is_dir():
    #          for ext in ("*.txt","*.md","*.pdf"): files.extend([str(x) for x in pp.rglob(ext)])
    #       else: files.append(str(pp))
    #  - docs=[]
    #  - for fp in files:
    #       ext = fp.lower().split(".")[-1]
    #       if ext in ("txt","md"): raw = read_text_file(fp)
    #       elif ext=="pdf": raw = read_pdf_file(fp)
    #       else: continue
    #       txt = clean_text(raw); docs.append({"path":fp,"text":txt})
    #  - return docs
    # ----------------------------------------------------------------------------
    # 정답 구현:
    files: List[str] = []
    for p in paths_or_dir:
        pp = Path(p)
        if pp.is_dir():
            for ext in ("*.txt", "*.md", "*.pdf", "*.csv"):  # ✅ CSV 확장자 추가
                files.extend([str(x) for x in pp.rglob(ext)])
        else:
            files.append(str(pp))

    docs: List[Dict[str, Any]] = []
    for fp in files:
        ext = fp.lower().split(".")[-1]
        if ext in ("txt", "md"):
            raw = read_text_file(fp)
        elif ext == "pdf":
            raw = read_pdf_file(fp)
        elif ext == "csv":
            # ✅ CSV 파일은 한 줄(한 행)씩 분리해서 저장
            try:
                df = pd.read_csv(fp, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(fp, encoding="cp949")

            # 각 행(row)을 하나의 문서로 취급
            for idx, row in df.iterrows():
                record_text = clean_text(json.dumps(row.to_dict(), ensure_ascii=False))
                docs.append({"path": f"{fp}::row_{idx}", "text": record_text})
        else:
            continue
    return docs


def build_corpus(paths_or_dir: List[str]) -> List[Dict[str, Any]]:
    """
    CSV/JSON 문서에서 자연어 코퍼스 생성
    반환: [{"id":..., "text":..., "meta":{"path":..., "chunk":..., "fields":...}}, ...]
    - text 필드에는 '공모전명' + '상세 내용' + '전공 우대'를 포함하여 임베딩 품질 향상
    """
    docs = load_documents(paths_or_dir)
    corpus: List[Dict[str, Any]] = []

    for d in docs:
        try:
            record = json.loads(d["text"]) if isinstance(d["text"], str) else d["text"]
        except Exception:
            record = {"공모전명": d["text"], "상세 내용": "", "전공 우대": ""}

        # ✅ 임베딩 텍스트 구성: 공모전명 + 상세 내용 + 전공 우대
        title = str(record.get("공모전명", "")).strip()
        desc = str(record.get("상세 내용", "")).strip()
        major = str(record.get("전공 우대", "")).strip()

        # 결합 순서: 공모전명 → 상세내용 → 전공우대
        text_parts = [p for p in [title, desc, f"(전공 우대: {major})" if major else ""] if p]
        text_for_embedding = ". ".join(text_parts) if text_parts else f"(제목 없음) from {d['path']}"

        # ✅ 청크 분할 (길 경우 여러 청크로)
        chunks = chunk_text(text_for_embedding)
        for i, ch in enumerate(chunks):
            cid = f"{d['path']}::chunk_{i:04d}"
            corpus.append({
                "id": cid,
                "text": ch,  # ✅ 공모전명 + 상세내용 + 전공우대 포함
                "meta": {
                    "path": d["path"],
                    "chunk": i,
                    "fields": record  # 원본 필드 전체 저장
                }
            })

    return corpus

def save_docs_jsonl(items: List[Dict[str, Any]], out_path: str):
    """
    문서 메타를 JSONL로 저장(ensure_ascii=False)
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-G-07] 구현 지침
    #  - with open(out_path,"w",encoding="utf-8") as f:
    #       for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # ----------------------------------------------------------------------------
    # 정답 구현:
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")