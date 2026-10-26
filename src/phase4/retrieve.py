"""기본 검색 경로(hybrid). dense와 sparse 결과를 RRF로 합친다."""

import sys
from pathlib import Path
from types import SimpleNamespace

# phase2의 search.py·collection.py를 가져다 쓰기 위해 import 검색 경로에 추가한다.
sys.path.insert(0, str(Path(__file__).parent.parent / "phase2"))

from collection import COLLECTION_NAME
from search import search_dense, search_sparse

# RRF 상수. 원 논문의 관례값 60을 명시해서 Qdrant 기본값에 기대지 않는다.
RRF_K = 60


def rrf_fuse(hit_lists, k=RRF_K):
    # 검색 결과 목록 여러 개(각각 점수 높은 순)를 RRF로 합쳐 한 목록으로 돌려준다.
    # 청크의 점수 = 목록마다 1 / (k + 순위) 의 합. 순위는 Qdrant 서버와 같게 0부터 센다.
    # 점수가 같으면 chunk_id 순으로 놓는다: 서버는 동점 순서가 실행마다 바뀌어 재현이 안 된다.
    scores = {}    # chunk_id -> 합산 점수
    payloads = {}  # chunk_id -> 그 청크의 payload

    for hits in hit_lists:
        for rank in range(len(hits)):
            payload = hits[rank].payload
            chunk_id = payload["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
                payloads[chunk_id] = payload
            scores[chunk_id] += 1 / (k + rank)

    # (점수 내림차순, chunk_id 오름차순)으로 정렬한다. 점수에 -를 붙여 내림차순을 만든다.
    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))

    fused = []
    for chunk_id in ordered_ids:
        fused.append(SimpleNamespace(payload=payloads[chunk_id], score=scores[chunk_id]))
    return fused


def retrieve_hybrid(client, dense, sparse, limit=5, candidates=50, collection=COLLECTION_NAME):
    # 기본 검색: dense와 sparse를 각각 candidates개씩 가져와 rrf_fuse로 합치고 상위 limit개를 돌려준다.
    # limit(돌려줄 개수)과 candidates(각 방식에서 가져올 후보 수)는 별개다.
    # 검색 평가는 둘 다 50으로 쟀고, 답변 생성에서는 후보 50개 중 앞 5개만 쓴다.
    dense_hits = search_dense(client, dense, limit=candidates, collection=collection)
    sparse_hits = search_sparse(client, sparse, limit=candidates, collection=collection)
    return rrf_fuse([dense_hits, sparse_hits])[:limit]
