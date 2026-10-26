"""Qdrant 컬렉션을 만든다 (named vector dense·sparse, payload 인덱스)."""

from qdrant_client import QdrantClient, models

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "encykorea_chunks"


def create_collection(client):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        # dense: bge-m3 출력 1024개, 코사인 유사도로 비교한다.
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
        },
        # sparse: 토큰별 가중치. 크기를 미리 정하지 않는다(토큰이 몇 개든 됨).
        # IDF 보정(Modifier.IDF)은 켜지 않는다: bge-m3가 이미 학습한 가중치를 쓰기 때문.
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )


def create_payload_indexes(client):
    field_names = ["eid", "era", "era_top", "field", "field_top", "kind"]

    for field_name in field_names:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def main():
    client = QdrantClient(url=QDRANT_URL)

    # 같은 이름의 컬렉션이 있으면 지우고 처음부터 다시 만든다 (스키마를 바꿔 가며 시험할 수 있게).
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print("기존 컬렉션을 지웠습니다:", COLLECTION_NAME)

    create_collection(client)
    create_payload_indexes(client)
    print("컬렉션을 만들었습니다:", COLLECTION_NAME)


if __name__ == "__main__":
    main()
