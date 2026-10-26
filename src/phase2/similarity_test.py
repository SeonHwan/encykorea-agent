"""두 문장의 dense 벡터로 코사인 유사도를 손으로 계산해 본다."""

from encode_test import encode_sentences, load_model


def dot_product(vec_a, vec_b):
    total = 0.0
    for i in range(len(vec_a)):
        total = total + float(vec_a[i]) * float(vec_b[i])
    return total


def main():
    texts = ["왕건", "고려 태조", "세종"]

    model = load_model()
    output = encode_sentences(model, texts)
    dense_list = output["dense_vecs"]

    # 세 표현을 둘씩 짝지어 점수를 낸다: (0,1) (0,2) (1,2)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = dot_product(dense_list[i], dense_list[j])
            print(texts[i], "↔", texts[j], ":", round(score, 4))


if __name__ == "__main__":
    main()
