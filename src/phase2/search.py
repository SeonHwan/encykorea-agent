"""질문을 인코딩해 Qdrant에서 dense / sparse / 필터 검색을 해 본다."""

from qdrant_client import QdrantClient, models

from collection import COLLECTION_NAME, QDRANT_URL
from encode_test import load_model


def encode_query(model, question):
    # 질문 한 개를 인코딩한다. 청크와 같은 모델·같은 max_length로 만들어야 같은 공간에서 비교된다.
    output = model.encode(
        [question],
        max_length=3072,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense = output["dense_vecs"][0].tolist()

    # sparse: {토큰ID: 점수} → 번호 리스트와 점수 리스트 (index.py의 encode_batch와 같은 방식)
    indices = []
    values = []
    for token_id, weight in output["lexical_weights"][0].items():
        indices.append(int(token_id))
        values.append(float(weight))
    sparse = models.SparseVector(indices=indices, values=values)

    return dense, sparse


# 결과에서 꺼내 볼 payload 필드. 벡터와 긴 text 전체는 가져오지 않는다.
SHOW_FIELDS = ["eid", "headword", "path", "kind", "chunk_id", "text"]


def search_dense(client, dense, limit=5, query_filter=None, collection=COLLECTION_NAME):
    response = client.query_points(
        collection_name=collection,
        query=dense,
        using="dense",
        query_filter=query_filter,
        limit=limit,
        with_payload=SHOW_FIELDS,
    )
    return response.points


def search_sparse(client, sparse, limit=5, query_filter=None, collection=COLLECTION_NAME):
    # dense와 같은 query_points다. query에 SparseVector를 넣고 using만 "sparse"로 바꾼다.
    response = client.query_points(
        collection_name=collection,
        query=sparse,
        using="sparse",
        query_filter=query_filter,
        limit=limit,
        with_payload=SHOW_FIELDS,
    )
    return response.points


def get_allowed_values(client, key):
    # 필터에 쓸 수 있는 값 목록을 Qdrant에서 직접 읽는다 (facet: 필드 값별 청크 수).
    # 목록을 코드에 적어 두면 데이터가 바뀔 때 어긋나서 읽어 온다.
    response = client.facet(COLLECTION_NAME, key=key, limit=1000)
    values = []
    for hit in response.hits:
        values.append(hit.value)
    return values


def make_condition(client, key, value):
    # eid는 15,836종이라 목록을 읽지 않는다. 틀린 값이면 필터가 걸린 채 결과가 0개일 뿐,
    # 필터 없는 전체 검색으로 바뀌지는 않는다.
    if key == "eid":
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    # 그 밖의 필드는 허용 목록에 없는 값이면 오류를 낸다. 오타나 "조선 말기" 같은 말이
    # 조용히 무시되어 필터 없는 전체 검색이 되는 것을 막기 위해서다.
    allowed = get_allowed_values(client, key)
    if value not in allowed:
        raise ValueError(f"{key}에 없는 값: {value!r} (가능한 값 {len(allowed)}종, 예: {allowed[:5]})")
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def make_filter(client, era_top=None, era=None, field_top=None, field=None, eid=None, kind=None):
    wanted = {"era_top": era_top, "era": era, "field_top": field_top, "field": field, "eid": eid, "kind": kind}

    conditions = []
    for key, value in wanted.items():
        if value is not None:
            conditions.append(make_condition(client, key, value))

    if len(conditions) == 0:
        return None  # 조건이 하나도 없으면 필터 없음
    return models.Filter(must=conditions)


def print_hits(title, hits):
    print(f"--- {title}")
    for rank in range(len(hits)):
        hit = hits[rank]
        payload = hit.payload
        path = " > ".join(payload["path"])
        text = payload["text"].replace("\n", " ")[:50]
        print(f"{rank + 1}. {hit.score:.4f}  「{payload['headword']}」 [{payload['kind']}] {path}")
        print(f"      {text}…")


QUESTIONS = [
    "왕건은 어떻게 후삼국을 통일했나?",
    "훈민정음은 누가 왜 만들었나?",
    "임진왜란 때 이순신이 한산도에서 거둔 승리",
    "936년에 일어난 일",
    "고려 후기의 불교 사상",
]


def main():
    client = QdrantClient(url=QDRANT_URL)
    model = load_model()

    for number in range(len(QUESTIONS)):
        question = QUESTIONS[number]
        dense, sparse = encode_query(model, question)

        print()
        print(f"===== 질문 {number + 1}: {question}")
        print_hits("dense", search_dense(client, dense))
        print_hits("sparse", search_sparse(client, sparse))

        # 질문 5번(마지막)은 시대 필터를 건 dense도 본다
        if number == len(QUESTIONS) - 1:
            era_filter = make_filter(client, era_top="고려")
            print_hits("dense + era_top=고려", search_dense(client, dense, query_filter=era_filter))


if __name__ == "__main__":
    main()
