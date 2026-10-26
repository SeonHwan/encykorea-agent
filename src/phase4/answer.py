"""질문과 검색 결과로 Claude에게 답을 받는다. 이 파일은 프로젝트 루트에서 실행해야 한다
(chunk.py가 토크나이저를 상대 경로 data/tokenizer/...로 읽기 때문)."""

import sys
from pathlib import Path

# phase1의 chunk.py에서 clean_text(본문 정리)를 가져다 쓴다.
# 검색에 쓴 embedding_text와 같은 규칙으로 정리해서, 모델이 읽는 글과 검색한 글이 어긋나지 않게 한다.
sys.path.insert(0, str(Path(__file__).parent.parent / "phase1"))

from chunk import clean_text


def format_context(hits):
    blocks = []
    for index in range(len(hits)):
        payload = hits[index].payload
        number = index + 1

        title = f"「{payload['headword']}」"
        if len(payload["path"]) > 0:
            title += " > " + " > ".join(payload["path"])
        elif payload["kind"] == "summary":
            title += " (요약)"

        body = clean_text(payload["text"])
        blocks.append(f"[{number}] {title}\n{body}")

    return "\n\n".join(blocks)


import anthropic
import openai
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

CLAUDE_MODEL = "claude-sonnet-5-5"
GPT_MODEL = "gpt-5.6-terra"

# 두 모델에 똑같이 주는 답변 규칙. 규칙을 바꾸면 비교가 어긋나므로 모델별로 따로 만들지 않는다.
SYSTEM_PROMPT = (
    "당신은 한국민족문화대백과사전을 근거로 한국사 질문에 답하는 도우미입니다.\n"
    "1. 사용자 메시지의 <근거> 안에 있는 내용만 사용해 답합니다. 근거에 없는 내용은 지어내지 않습니다. "
    "근거로 답할 수 없으면 '제공된 근거로는 답할 수 없습니다'라고 말합니다.\n"
    "2. 답변의 각 문장 끝에 그 문장의 근거가 된 번호를 [1]처럼 붙입니다. 여러 근거면 [1][3]처럼 붙입니다.\n"
    "3. 항목 코드(E로 시작하는 숫자)나 링크는 쓰지 않습니다.\n"
    "4. 존댓말로 간결하게 답합니다."
)


def build_user_message(question, context):
    # 두 모델에 같은 사용자 메시지를 준다: 근거 글 묶음 뒤에 질문.
    return f"<근거>\n{context}\n</근거>\n\n질문: {question}"


def ask_claude(client, user_message, max_tokens=2000):
    # Claude는 thinking이 기본으로 켜져 있어 max_tokens가 작으면 텍스트가 빈 응답이 나올 수 있다.
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text
    return {"text": text.strip(), "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens, "stop": response.stop_reason}


def ask_gpt(client, user_message, max_tokens=2000):
    # GPT는 Chat Completions로 부른다 (Responses와 Chat Completions가 둘 다 통함을 확인).
    response = client.chat.completions.create(
        model=GPT_MODEL,
        max_completion_tokens=max_tokens,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_message}],
    )
    choice = response.choices[0]
    return {"text": (choice.message.content or "").strip(), "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens, "stop": choice.finish_reason}


def make_clients():
    # 각 SDK가 ANTHROPIC_API_KEY, OPENAI_API_KEY를 환경변수에서 읽는다.
    return anthropic.Anthropic(), openai.OpenAI()


import re


def cited_numbers(answer):
    # 답변에서 모델이 쓴 근거 번호 [n]을 처음 나온 순서대로, 중복 없이 뽑는다.
    # 모델마다 번호를 문장 끝 "[2]", 마침표 뒤 ".[1][2]", 줄 끝 등 여러 위치에 둬서, 위치와 상관없이 [숫자]를 모두 찾는다.
    found = re.findall(r"\[(\d+)\]", answer)

    numbers = []
    for text in found:
        number = int(text)
        if number not in numbers:
            numbers.append(number)
    return numbers


def build_source_line(sources, numbers):
    headword_by_number = {}
    for source in sources:
        headword_by_number[source["n"]] = source["headword"]

    headwords = []
    for number in numbers:
        if number not in headword_by_number:
            continue  # 범위 밖 번호(모델이 없는 번호를 쓴 경우)는 출처에 넣지 않는다
        headword = headword_by_number[number]
        if headword not in headwords:
            headwords.append(headword)

    if len(headwords) == 0:
        return ""  # 인용이 없으면 출처 줄도 없다 ("근거로 답할 수 없습니다" 같은 답)

    names = ", ".join(f"「{headword}」" for headword in headwords)
    return f"출처: {names}, ⓒ한국민족문화대백과사전"


def strip_numbers(answer):
    text = re.sub(r"\s*\[\d+\]", "", answer)           # 번호와 그 앞 공백을 지운다
    text = re.sub(r"[ \t]+\n", "\n", text)               # 줄 끝에 남은 공백
    text = re.sub(r"[ \t]{2,}", " ", text)               # 이중 공백
    text = re.sub(r"\n{3,}", "\n\n", text)               # 빈 줄이 셋 이상이면 하나로
    return text.strip()


def attach_sources(answer, sources):
    # 모델의 답변(본문 + [n])을 최종 형태로 바꾼다: 번호를 지운 본문 + 출처 줄 + 검색 안내.
    # 순서가 중요하다: 번호를 지우기 전에 먼저 읽어서 출처를 만든다.
    body = strip_numbers(answer)

    # 답할 수 없다고 결론낸 답에는 출처 줄을 붙이지 않는다. 모델이 "없다"는 답에도 번호를 붙이는 경우가 있어서,
    # 번호가 있어도 첫 문장이 답할 수 없다는 결론이면 출처를 뺀다.
    if is_unanswerable(answer):
        return body

    numbers = cited_numbers(answer)
    source_line = build_source_line(sources, numbers)

    if source_line == "":
        return body  # 인용이 없는 답에는 출처 줄도 없다
    return f"{body}\n\n{source_line}"


def make_sources(hits):
    # 검색 결과를 번호와 항목명을 짝지은 목록으로 바꾼다. 번호는 format_context가 붙인 번호와 같다(1부터).
    sources = []
    for index in range(len(hits)):
        payload = hits[index].payload
        sources.append({"n": index + 1, "headword": payload["headword"], "kind": payload["kind"],
                        "path": payload["path"], "chunk_id": payload["chunk_id"]})
    return sources


def prepare_input(search_client, embed_model, question, limit=5, candidates=50):
    # 모델을 부르기 전까지의 준비: 검색 → 근거 글 → 모델 입력. 호출은 하지 않는다.
    # 검색 모듈은 여기서 가져온다 (answer.py를 import만 할 때 모델을 읽지 않게).
    sys.path.insert(0, str(Path(__file__).parent.parent / "phase2"))
    from search import encode_query
    from retrieve import retrieve_hybrid

    dense, sparse = encode_query(embed_model, question)
    hits = retrieve_hybrid(search_client, dense, sparse, limit=limit, candidates=candidates)
    context = format_context(hits)
    return {"user_message": build_user_message(question, context),
            "sources": make_sources(hits), "context": context}


def answer_question(search_client, embed_model, llm_client, model_name, question):
    # 질문 하나 → 최종 답변. model_name은 "claude" 또는 "gpt"이고 llm_client는 그 모델의 SDK 클라이언트다.
    prepared = prepare_input(search_client, embed_model, question)

    if model_name == "claude":
        result = ask_claude(llm_client, prepared["user_message"])
    elif model_name == "gpt":
        result = ask_gpt(llm_client, prepared["user_message"])
    else:
        raise ValueError(f"model_name은 'claude' 또는 'gpt'여야 합니다: {model_name!r}")

    final = attach_sources(result["text"], prepared["sources"])
    return {"answer": final, "raw": result["text"], "sources": prepared["sources"],
            "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"], "stop": result["stop"]}


def is_unanswerable(answer):
    # 모델이 "근거로는 답할 수 없다"고 결론낸 답인지 본다. 첫 문장만 보고, 번호 [n]은 무시한다.
    # 부분 답변("일부만 설명할 수 있습니다 ... 세부는 답할 수 없습니다")은 근거를 실제로 썼으므로 해당하지 않는다.
    text = re.sub(r"\s*\[\d+\]", "", answer).strip()
    first_sentence = re.split(r"(?<=[.?!])\s+|\n", text, maxsplit=1)[0]

    # 첫 문장이 "답할 수 없습니다 / 알 수 없습니다"로 결론내는지
    return bool(re.search(r"(답할|알) 수 없(습니다|다)\.?$", first_sentence))
