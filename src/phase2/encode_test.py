"""bge-m3로 문장 몇 개를 인코딩해 dense·sparse 출력을 확인한다."""

from FlagEmbedding import BGEM3FlagModel


def load_model():
    # use_fp16=True: 숫자를 32비트 대신 16비트로 계산해 VRAM과 시간을 줄인다 (정확도 손실은 아주 작음).
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return model


def encode_sentences(model, sentences):
    output = model.encode(
        sentences,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return output


def main():
    sentences = [
        "왕건은 후삼국을 통일하고 고려를 세웠다.",
        "고려의 태조는 936년에 후삼국을 통일하였다.",
        "세종은 훈민정음을 창제하였다.",
    ]

    model = load_model()
    output = encode_sentences(model, sentences)

    dense_list = output["dense_vecs"]
    sparse_list = output["lexical_weights"]
    # 토큰 ID(숫자)를 사람이 읽을 수 있는 토큰 글자로 바꾼다
    token_list = model.convert_id_to_token(sparse_list)

    for i in range(len(sentences)):
        print("문장:", sentences[i])
        print("  dense 길이:", len(dense_list[i]))
        print("  dense 앞 5개:", dense_list[i][:5])
        print("  sparse 토큰 수:", len(token_list[i]))
        print("  sparse 토큰 → 가중치:", token_list[i])


if __name__ == "__main__":
    main()
