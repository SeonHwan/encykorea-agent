"""평가 질문 30개로 검색 품질을 잰다 (recall@k, MRR)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

# phase2 폴더의 search.py·collection.py를 가져다 쓰기 위해 import 검색 경로에 추가한다.
# (search.py가 `from collection import ...`처럼 같은 폴더 기준으로 import하기 때문에 폴더째 추가해야 한다.)
sys.path.insert(0, str(Path(__file__).parent.parent / "phase2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase4"))

from qdrant_client import QdrantClient, models

from collection import COLLECTION_NAME, QDRANT_URL
from encode_test import load_model
from search import SHOW_FIELDS, encode_query, search_dense, search_sparse
from retrieve import RRF_K, retrieve_hybrid, rrf_fuse

# 이 파일(eval_retrieval.py)이 있는 폴더 옆의 eval_queries.json
QUERIES_PATH = Path(__file__).parent / "eval_queries.json"
# 청크 전체(embedding_text 포함). 컬렉션 payload에는 embedding_text가 없어서 여기서 찾아온다.
CHUNKS_PATH = Path(__file__).parent.parent.parent / "data" / "chunks.jsonl"


def load_queries():
    # eval_queries.json을 읽어 질문 목록(딕셔너리의 리스트)으로 돌려준다.
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)
    return queries


def load_embedding_texts():
    texts = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            texts[chunk["chunk_id"]] = chunk["embedding_text"]
    return texts


def is_answer_chunk(payload, query, level):
    if level == "item":
        for eid in query["answer_eids"]:
            if payload["eid"] == eid:
                return True
        return False

    if level == "section":
        for label in query["answer_sections"]:
            same_eid = payload["eid"] == label["eid"]
            same_kind = payload["kind"] == label["kind"]
            same_path = payload["path"] == label["path"]
            if same_eid and same_kind and same_path:
                return True
        return False

    raise ValueError(f"level은 'item' 또는 'section'이어야 합니다: {level!r}")


def find_first_rank(hits, query, level):
    for index in range(len(hits)):
        payload = hits[index].payload
        if is_answer_chunk(payload, query, level):
            return index + 1
    return None


# recall을 재는 k 값들. "상위 k개 안에 정답이 있는가"
RECALL_KS = [1, 5, 10, 50]


def summarize_ranks(ranks):
    total = len(ranks)

    summary = {}
    for k in RECALL_KS:
        hit_count = 0
        for rank in ranks:
            if rank is not None and rank <= k:
                hit_count += 1
        summary[f"recall@{k}"] = hit_count / total

    reciprocal_sum = 0.0
    for rank in ranks:
        if rank is not None:
            reciprocal_sum += 1 / rank
    summary["mrr"] = reciprocal_sum / total

    return summary


def search_hybrid(client, dense, sparse, limit=5, candidates=50):
    # dense와 sparse를 각각 candidates개씩 가져와(prefetch) 서버에서 RRF로 합치고 상위 limit개를 돌려준다.
    # Prefetch의 limit 기본값이 10이라 지정하지 않으면 후보가 10개로 줄어든다.
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense, using="dense", limit=candidates),
            models.Prefetch(query=sparse, using="sparse", limit=candidates),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=RRF_K)),
        limit=limit,
        with_payload=SHOW_FIELDS,
    )
    return response.points


def encode_colbert(model, texts, batch_size=16):
    # 글 여러 개를 colbert 벡터로 인코딩한다. 글마다 (토큰 수, 1024) 크기의 배열이 하나씩 나온다.
    # max_length=3072는 청크를 인코딩할 때(index.py)와 같게 맞춘 값이다.
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=3072,
        return_dense=False,
        return_sparse=False,
        return_colbert_vecs=True,
    )
    return output["colbert_vecs"]


def rerank_colbert(model, question, hits, texts):
    question_vecs = encode_colbert(model, [question])[0]

    passages = []
    for hit in hits:
        passages.append(texts[hit.payload["chunk_id"]])
    passage_vecs = encode_colbert(model, passages)

    scores = {}
    for index in range(len(hits)):
        chunk_id = hits[index].payload["chunk_id"]
        scores[chunk_id] = float(model.colbert_score(question_vecs, passage_vecs[index]))

    # 점수 내림차순, 동점이면 chunk_id 순 (rrf_fuse와 같은 규칙)
    ordered = sorted(hits, key=lambda hit: (-scores[hit.payload["chunk_id"]], hit.payload["chunk_id"]))

    reranked = []
    for hit in ordered:
        reranked.append(SimpleNamespace(payload=hit.payload, score=scores[hit.payload["chunk_id"]]))
    return reranked


def evaluate(client, model, queries, mode, texts=None, collection=COLLECTION_NAME):
    max_k = max(RECALL_KS)

    item_ranks = []
    section_ranks = []
    for query in queries:
        dense, sparse = encode_query(model, query["query"])

        if mode == "dense":
            hits = search_dense(client, dense, limit=max_k, collection=collection)
        elif mode == "sparse":
            hits = search_sparse(client, sparse, limit=max_k, collection=collection)
        elif mode == "hybrid":
            # 서버 RRF(search_hybrid)는 동점 순서가 실행마다 바뀌어서, 직접 합친다 (retrieve_hybrid).
            hits = retrieve_hybrid(client, dense, sparse, limit=max_k, candidates=max_k, collection=collection)
        elif mode == "rerank":
            candidates = retrieve_hybrid(client, dense, sparse, limit=max_k, candidates=max_k, collection=collection)
            hits = rerank_colbert(model, query["query"], candidates, texts)
        else:
            raise ValueError(f"mode는 'dense'·'sparse'·'hybrid'·'rerank' 중 하나여야 합니다: {mode!r}")

        item_ranks.append(find_first_rank(hits, query, "item"))
        section_ranks.append(find_first_rank(hits, query, "section"))

    return {"item": item_ranks, "section": section_ranks}


def rank_text(rank):
    # None(상위 50개에 정답 없음)은 '-'로 보여 준다
    if rank is None:
        return "-"
    return str(rank)


def main():
    queries = load_queries()
    client = QdrantClient(url=QDRANT_URL)
    model = load_model()

    texts = load_embedding_texts()
    modes = ["dense", "sparse", "hybrid", "rerank"]
    results = {}
    for mode in modes:
        results[mode] = evaluate(client, model, queries, mode, texts)

    # 요약표: 모드별로 항목 기준 / 섹션 기준
    print()
    print(f"===== 모드별 비교 (질문 {len(queries)}개)")
    header = f"{'지표':<11}"
    for mode in modes:
        header += f"{mode + ' 항목':>14}{mode + ' 섹션':>14}"
    print(header)

    summaries = {}
    for mode in modes:
        summaries[mode] = {
            "item": summarize_ranks(results[mode]["item"]),
            "section": summarize_ranks(results[mode]["section"]),
        }
    for name in summaries["dense"]["item"]:
        line = f"{name:<11}"
        for mode in modes:
            line += f"{summaries[mode]['item'][name]:>14.3f}{summaries[mode]['section'][name]:>14.3f}"
        print(line)

    # 질문별 순위: 모드마다 (항목/섹션)
    print()
    print("===== 질문별 첫 정답 순위 (항목/섹션)")
    print(f"{'id':<5}{'유형':<6}{'dense':>10}{'sparse':>10}{'hybrid':>10}{'rerank':>10}  질문")
    for number in range(len(queries)):
        query = queries[number]
        line = f"{query['id']:<5}{query['type']:<6}"
        for mode in modes:
            item_rank = rank_text(results[mode]["item"][number])
            section_rank = rank_text(results[mode]["section"][number])
            line += f"{item_rank + '/' + section_rank:>10}"
        print(line + f"  {query['query'][:24]}")


if __name__ == "__main__":
    main()
