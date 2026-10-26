"""청크를 bge-m3로 인코딩해 Qdrant에 적재한다."""

import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from collection import COLLECTION_NAME, QDRANT_URL
from encode_test import load_model

CHUNKS_PATH = Path("data/chunks.jsonl")

# chunk_id → UUID 변환의 기준값. 이 값을 바꾸면 모든 포인트 id가 달라지므로 고정한다.
ID_NAMESPACE = uuid.UUID("6f1c2b8e-3d4a-4e5f-9a7b-1c2d3e4f5a6b")


def load_chunks(path=CHUNKS_PATH):
    # jsonl: 한 줄에 JSON 하나. 줄마다 읽어서 딕셔너리로 바꿔 리스트에 모은다.
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append(chunk)
    return chunks


def split_values(text):
    values = []
    for part in text.split("|"):
        part = part.strip()
        if part != "":
            values.append(part)
    return values


def top_levels(values):
    # 각 값에서 "/" 앞부분만 모으되, 같은 값은 한 번만 넣는다. ["조선/조선 전기", "조선/조선 후기"] → ["조선"]
    tops = []
    for value in values:
        top = value.split("/")[0]
        if top not in tops:
            tops.append(top)
    return tops


def make_payload(chunk):
    era_list = split_values(chunk["era"])
    field_list = split_values(chunk["field"])

    payload = {
        "chunk_id": chunk["chunk_id"],
        "eid": chunk["eid"],
        "headword": chunk["headword"],
        "url": chunk["url"],
        "field": field_list,
        "field_top": top_levels(field_list),
        "era": era_list,
        "era_top": top_levels(era_list),
        "primary_type": chunk["primary_type"],
        "kind": chunk["kind"],
        "path": chunk["path"],
        "text": chunk["text"],
    }
    return payload


def encode_batch(model, batch):
    # batch: 청크 딕셔너리 여러 개. 인코딩 입력은 embedding_text다 (text가 아님).
    texts = []
    for chunk in batch:
        texts.append(chunk["embedding_text"])

    # max_length=3072: 가장 긴 청크(2,853토큰)까지 자르지 않고 넣는다. batch_size=32: 측정에서 빨랐던 값.
    output = model.encode(
        texts,
        batch_size=32,
        max_length=3072,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    # 청크마다 결과를 묶는다. sparse는 {토큰ID: 점수}를 두 리스트(번호들, 점수들)로 풀어 둔다.
    results = []
    for i in range(len(batch)):
        weights = output["lexical_weights"][i]
        indices = []
        values = []
        for token_id, weight in weights.items():
            indices.append(int(token_id))
            values.append(float(weight))
        results.append({
            "chunk_id": batch[i]["chunk_id"],
            "dense": output["dense_vecs"][i],
            "sparse_indices": indices,
            "sparse_values": values,
        })
    return results


def make_point(result, chunk):
    # result: encode_batch의 청크 한 개 결과, chunk: 같은 청크의 원본 딕셔너리.
    # Qdrant 포인트 id는 정수나 UUID만 된다. chunk_id("E0000025-000")에서 UUID를 만들면
    # 같은 chunk_id는 항상 같은 UUID가 나와서, 다시 적재해도 같은 포인트를 덮어쓴다.
    point_id = str(uuid.uuid5(ID_NAMESPACE, result["chunk_id"]))

    point = models.PointStruct(
        id=point_id,
        vector={
            "dense": result["dense"].tolist(),
            "sparse": models.SparseVector(
                indices=result["sparse_indices"],
                values=result["sparse_values"],
            ),
        },
        payload=make_payload(chunk),
    )
    return point


def main():
    chunks = load_chunks()
    model = load_model()
    client = QdrantClient(url=QDRANT_URL)

    batch_size = 2000  # 한 번에 인코딩하는 청크 수
    send_size = 200    # 한 번에 Qdrant로 보내는 포인트 수 (요청 하나가 너무 커지지 않게)
    total = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        results = encode_batch(model, batch)

        points = []
        for i in range(len(batch)):
            points.append(make_point(results[i], batch[i]))

        for pos in range(0, len(points), send_size):
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[pos:pos + send_size],
                wait=True,  # 서버가 저장을 끝낸 뒤에 돌아온다
            )

        total += len(points)
        print(f"{total} / {len(chunks)} 적재 완료")

    print("포인트 수:", client.count(COLLECTION_NAME).count)


if __name__ == "__main__":
    main()


def update_payloads(client, chunks, send_size=500):
    total = 0
    for start in range(0, len(chunks), send_size):
        batch = chunks[start:start + send_size]

        operations = []
        for chunk in batch:
            point_id = str(uuid.uuid5(ID_NAMESPACE, chunk["chunk_id"]))
            operations.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload=make_payload(chunk),
                        points=[point_id],
                    )
                )
            )
        client.batch_update_points(
            collection_name=COLLECTION_NAME,
            update_operations=operations,
            wait=True,
        )
        total += len(batch)
        print(f"{total} / {len(chunks)} payload 갱신")
