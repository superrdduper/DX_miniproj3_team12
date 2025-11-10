# -*- coding: utf-8 -*-
"""
OpenAI 임베딩 래퍼
- 요구사항: 배치 인코딩, 재시도(backoff), L2 정규화
"""

import os, time
from typing import List
import numpy as np
# from httpx import ReadTimeout  # 선택: 재시도 구분용
# from openai import OpenAI


class Embeddings:
    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 128,
        max_retries: int = 4,
        *,
        timeout: int = 30,                 # 선택: 소프트 타임아웃(encode에서 사용)
        normalize: bool = True,            # 선택: L2 정규화 on/off
        api_key: str | None = None,        # 선택: 환경변수보다 코드 우선 주입 가능
        base_url: str | None = None,       # 선택: 사내 프록시/엔터프라이즈 게이트웨이
        client: object | None = None,      # 선택: 외부에서 SDK 클라이언트 주입(모킹/테스트)
        seed: int | None = None,           # 선택: 더미 모드 재현성
    ):
        """
        요구사항:
          - 모델/배치/재시도/클라이언트 구성
          - 키 없거나 SDK 미설치인 환경에서도 즉시 죽지 않도록 설계(self.client=None 허용)
        """
        # 1) 기본 필드 및 검증 ------------------------------------------------------
        self.model = model or "text-embedding-3-small"
        self.batch_size = int(batch_size) if batch_size and batch_size > 0 else 1
        self.max_retries = int(max_retries) if max_retries and max_retries > 0 else 1
        self.timeout = int(timeout)
        self.normalize = bool(normalize)

        # 더미/테스트용 난수기(encode에서 필요할 수 있음)
        import numpy as _np  # 지역 임포트로 네임스페이스 충돌 방지
        self._rng = _np.random.RandomState(seed if seed is not None else 42)

        # 2) 외부에서 클라이언트 객체를 직접 주입한 경우 우선 사용 ------------------
        if client is not None:
            self.client = client
            self._use_dummy = False
            return

        # 3) 환경 변수/인자로 OpenAI 클라이언트 구성 --------------------------------
        #    - 키 우선순위: 인자 api_key → 환경변수 OPENAI_API_KEY
        #    - 베이스URL: 인자 base_url → 환경변수 OPENAI_BASE_URL (또는 OPENAI_API_BASE)
        key = api_key or os.getenv("OPENAI_API_KEY")
        base = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

        self.client = None
        self._use_dummy = True  # 기본은 더미, 성공하면 False로 전환

        if key:
            try:
                # 지연 임포트: SDK가 없어도 모듈 로드시 에러 안 나게
                from openai import OpenAI
                self.client = OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)
                self._use_dummy = False
            except Exception:
                # SDK 미설치/버전 문제/기타 예외 → 더미 모드 유지
                self.client = None
                self._use_dummy = True
        # key가 없으면 self.client=None 유지(더미 모드)

        # ----------------------------------------------------------------------------
        # TODO[DAY2-E-01] 구현 지침
        #  - self.model = model or "text-embedding-3-small"
        #  - self.batch_size = batch_size; self.max_retries = max_retries
        #  - key = os.getenv("OPENAI_API_KEY"); self.client = OpenAI(api_key=key)
        # ----------------------------------------------------------------------------
        # raise NotImplementedError("TODO[DAY2-E-01]: Embeddings.__init__ 구성")

    def _embed_once(self, text: str) -> np.ndarray:
        """
        단일 텍스트 임베딩 호출 → np.ndarray(float32) + L2 정규화
        - 예외 발생 시 상위 encode에서 재시도하도록 예외를 그대로 올려보냄
        """
        # ----------------------------------------------------------------------------
        # TODO[DAY2-E-02] 구현 지침
        #  - resp = self.client.embeddings.create(model=self.model, input=text)
        #  - vec = np.array(resp.data[0].embedding, dtype="float32")
        #  - norm = np.linalg.norm(vec) + 1e-12; vec = vec / norm
        #  - return vec
        # ----------------------------------------------------------------------------
        # raise NotImplementedError("TODO[DAY2-E-02]: 단일 임베딩 호출")
        def _embed_once(self, text: str) -> np.ndarray:
            # 임베딩 API 호출: 텍스트 한 줄 단위로 model을 적용
            resp = self.client.embeddings.create(model=self.model, input=text)
            # 한 줄씩 임베딩하므로 data[0], float32 타입으로 숫자 벡터 추출
            vec = np.array(resp.data[0].embedding, dtype="float32")
            # L2 정규화: 벡터 길이를 1로 만들어 거리 계산 시 영향 최소화
            # 1e-12 추가: 0으로 나누는 것 방지
            norm = np.linalg.norm(vec) + 1e-12
            vec = vec / norm
            # 정규화된 벡터 반환
            return vec

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        배치 인코딩 + 재시도(backoff). 최종 shape = (N, D)
        - 비어 있으면 (0, D) 반환. D는 1536 등 모델 차원 (미정이면 1536 가정 가능)
        """
        # ----------------------------------------------------------------------------
        # TODO[DAY2-E-03] 구현 지침
        #  - if not texts: return np.zeros((0, 1536), dtype="float32")
        #  - out = []
        #  - for start in range(0, len(texts), self.batch_size):
        #       batch = texts[start:start+self.batch_size]
        #       for each in batch:
        #          for attempt in range(self.max_retries):
        #             try:
        #                out.append(self._embed_once(each)); break
        #             except Exception as e:
        #                time.sleep(0.5 * (2 ** attempt))
        #                if attempt == self.max_retries - 1: raise
        #  - return np.vstack(out)
        # ----------------------------------------------------------------------------
        # raise NotImplementedError("TODO[DAY2-E-03]: 배치 임베딩 인코딩")
        if not texts:
            return np.zeros((0, 1536), dtype="float32") #비어 있으면 (0,D) 반환. 1536 == 임베딩 벡터 길이의 기본값
 
        out = [] #임베딩 결과 저장용 리스트
        for start in range(0, len(texts), self.batch_size): #배치 사이즈만큼 나누어서 처리
            batch = texts[start:start + self.batch_size]
            for each in batch:
                for attempt in range(self.max_retries):
                    try: #각 문장을 _embed_once 함수로 임베딩 시도
                        out.append(self._embed_once(each))
                        break
                    except Exception as e: #실패하면 점점 기다리는 시간을 늘려가면서 다시 시도
                        time.sleep(0.5 * (2 ** attempt))
                        if attempt == self.max_retries - 1:
                            raise
        return np.vstack(out) #out리스트에 저장된 벡터들을 하나의 큰 배열로 합쳐서 반환
        #5개 문장 → (5, 1536) 크기의 배열
