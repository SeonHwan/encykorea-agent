"""문맥 주입 비교용 컬렉션 두 개(ctx_off·ctx_on)를 만든다. 기존 encykorea_chunks는 건드리지 않는다."""

import sys
from pathlib import Path

# phase2의 collection.py·index.py·encode_test.py를 가져다 쓰기 위해 경로에 추가한다.
sys.path.insert(0, str(Path(__file__).parent.parent / "phase2"))

from qdrant_client import QdrantClient, models

from collection import QDRANT_URL

# 이 파일이 만들고 지울 수 있는 컬렉션 이름. 기존 컬렉션(encykorea_chunks)이 실수로 지워지지 않게 한다.
EXPERIMENT_NAMES = ["ctx_off", "ctx_on"]


def create_named_collection(client, name):
    # phase2/collection.py의 create_collection·create_payload_indexes와 같은 구조에 이름만 인자로 받는다.
    if name not in EXPERIMENT_NAMES:
        raise ValueError(f"실험용 컬렉션 이름이 아닙니다: {name!r} (허용: {EXPERIMENT_NAMES})")

    # 같은 이름이 있으면 지우고 다시 만든다. 위에서 이름을 확인했으니 기존 컬렉션은 지워지지 않는다.
    if client.collection_exists(name):
        client.delete_collection(name)
        print("기존 컬렉션을 지웠습니다:", name)

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )
    for field_name in ["eid", "era", "era_top", "field", "field_top", "kind"]:
        client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    print("컬렉션을 만들었습니다:", name)


import json

from encode_test import load_model
from index import encode_batch, make_point

CHUNKS_PATH = Path(__file__).parent.parent.parent / "data" / "chunks.jsonl"
CONTEXTS_PATH = Path(__file__).parent.parent.parent / "data" / "contexts.jsonl"


def load_experiment_chunks():
    # contexts.jsonl에 문맥이 있는 청크만 chunks.jsonl에서 골라, 문맥 없음/있음 두 벌의 청크 목록을 만든다.
    contexts = {}
    for line in open(CONTEXTS_PATH, encoding="utf-8"):
        record = json.loads(line)
        # 빈 문맥은 조용히 섞이면 측정이 흐려지므로 바로 멈춘다
        if not record["context"].strip():
            raise ValueError(f"빈 문맥이 있습니다: {record['chunk_id']}")
        contexts[record["chunk_id"]] = record["context"]

    off_chunks = []
    on_chunks = []
    for line in open(CHUNKS_PATH, encoding="utf-8"):
        chunk = json.loads(line)
        if chunk["chunk_id"] not in contexts:
            continue
        off_chunks.append(chunk)

        # 문맥 있음: 같은 청크 딕셔너리를 복사해 embedding_text 맨 앞에 문맥 한 문장을 붙인다
        on_chunk = dict(chunk)
        on_chunk["embedding_text"] = contexts[chunk["chunk_id"]] + "\n" + chunk["embedding_text"]
        on_chunks.append(on_chunk)

    if len(off_chunks) != len(contexts):
        raise ValueError(f"문맥 {len(contexts)}개 중 chunks.jsonl에서 찾은 청크 {len(off_chunks)}개: 불일치")
    return off_chunks, on_chunks


def index_chunks(client, model, name, chunks, batch_size=500, send_size=200):
    if name not in EXPERIMENT_NAMES:
        raise ValueError(f"실험용 컬렉션 이름이 아닙니다: {name!r}")

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        results = encode_batch(model, batch)

        points = []
        for i in range(len(batch)):
            points.append(make_point(results[i], batch[i]))
        for pos in range(0, len(points), send_size):
            client.upsert(collection_name=name, points=points[pos:pos + send_size], wait=True)

        total += len(points)
        print(f"  [{name}] {total} / {len(chunks)} 적재")
    print(f"[{name}] 포인트 수:", client.count(name).count)


def main():
    client = QdrantClient(url=QDRANT_URL)
    model = load_model()
    off_chunks, on_chunks = load_experiment_chunks()
    print(f"청크 {len(off_chunks):,}개 (문맥 없음 = 문맥 있음 개수)")

    create_named_collection(client, "ctx_off")
    index_chunks(client, model, "ctx_off", off_chunks)
    create_named_collection(client, "ctx_on")
    index_chunks(client, model, "ctx_on", on_chunks)


if __name__ == "__main__":
    main()
