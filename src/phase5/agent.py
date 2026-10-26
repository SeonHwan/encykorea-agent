"""질문을 보고 스스로 검색 도구를 골라 여러 번 호출하고 답하는 에이전트.
프로젝트 루트에서 실행한다 (answer.py가 chunk.py를 import하기 때문)."""

import sys
from pathlib import Path

for folder in ["phase4", "phase2"]:
    sys.path.insert(0, str(Path(__file__).parent.parent / folder))

import time

from answer import CLAUDE_MODEL, attach_sources, clean_text
from retrieve import retrieve_hybrid
from search import encode_query


# 모델에게 보여 줄 도구 설명서. 모델은 이 글만 보고 도구를 언제, 어떻게 부를지 정한다.
SEARCH_TOOL_DEFINITION = {
    "name": "search_encyclopedia",
    "description": (
        "한국민족문화대백과사전에서 질문과 관련된 글 조각을 검색한다. "
        "한 번 부르면 관련도가 높은 조각 5개가 [번호] 「항목명」 > 소제목 형식으로 돌아온다. "
        "인물·사건·유물·제도 등 한국사 사실을 답하기 전에 먼저 부른다. "
        "질문이 여러 대상이나 여러 측면을 묻는다면(예: 두 인물 비교, 출생과 업적) 대상별로 나누어 여러 번 부른다. "
        "결과가 질문과 맞지 않으면 항목명이나 핵심어를 바꾸어 다시 부를 수 있다. "
        "검색 결과에 없는 내용은 지어내지 말고, 근거를 찾지 못했으면 찾지 못했다고 답한다."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "검색할 문장 또는 핵심어. 항목명이 짐작되면 항목명을 넣는다. 예: '광종 노비안검법', '신석기시대 빗살무늬토기'",
            },
        },
        "required": ["query"],
    },
}


class SearchTool:
    # 검색 도구 하나. 호출할 때마다 근거 번호를 이어 붙이기 위해 상태(다음 번호, 번호→항목)를 가진다.
    def __init__(self, search_client, embed_model, limit=5):
        self.search_client = search_client
        self.embed_model = embed_model
        self.limit = limit
        self.next_number = 1   # 다음에 붙일 근거 번호 (한 번 검색에 limit개씩 늘어난다)
        self.sources = []      # [{"n": 번호, "headword": 항목명, ...}] 에이전트 전체에서 쌓인다

    def run(self, query):
        # query로 hybrid 검색을 하고, 번호를 이어 붙인 근거 글을 돌려준다.
        dense, sparse = encode_query(self.embed_model, query)
        hits = retrieve_hybrid(self.search_client, dense, sparse, limit=self.limit)

        blocks = []
        for hit in hits:
            payload = hit.payload
            number = self.next_number
            self.next_number += 1

            title = f"「{payload['headword']}」"
            if len(payload["path"]) > 0:
                title += " > " + " > ".join(payload["path"])
            elif payload["kind"] == "summary":
                title += " (요약)"

            self.sources.append({"n": number, "headword": payload["headword"], "kind": payload["kind"],
                                 "path": payload["path"], "chunk_id": payload["chunk_id"]})
            blocks.append(f"[{number}] {title}\n{clean_text(payload['text'])}")

        if len(blocks) == 0:
            return "검색 결과가 없습니다."
        return "\n\n".join(blocks)


def execute_tool_call(tool, tool_use_block):
    # 모델이 요청한 도구 호출 하나를 실행하고, 모델에게 돌려줄 tool_result 블록(dict)을 만든다.
    # 실패해도 예외를 밖으로 던지지 않고 is_error=True 결과로 돌려, 모델이 다시 판단하게 한다.
    result = {"type": "tool_result", "tool_use_id": tool_use_block.id}

    if tool_use_block.name != "search_encyclopedia":
        result["content"] = f"알 수 없는 도구입니다: {tool_use_block.name}"
        result["is_error"] = True
        return result

    query = tool_use_block.input.get("query")
    if not isinstance(query, str) or query.strip() == "":
        result["content"] = "query는 비어 있지 않은 문자열이어야 합니다."
        result["is_error"] = True
        return result

    try:
        result["content"] = tool.run(query)
    except Exception as error:
        result["content"] = f"검색 중 오류가 났습니다: {error}"
        result["is_error"] = True
    return result


# 에이전트용 답변 규칙. 단일 호출용 SYSTEM_PROMPT와 달리 근거가 사용자 메시지가 아니라 도구 결과로 들어온다.
AGENT_SYSTEM_PROMPT = (
    "당신은 한국민족문화대백과사전을 근거로 한국사 질문에 답하는 도우미입니다.\n"
    "1. 사실을 답하기 전에 search_encyclopedia 도구로 근거를 찾습니다. 검색 결과에 있는 내용만 사용해 답하고, "
    "없는 내용은 지어내지 않습니다. 근거를 찾지 못했으면 '제공된 근거로는 답할 수 없습니다'라고 말합니다.\n"
    "2. 답변의 각 문장 끝에 그 문장의 근거가 된 번호를 [1]처럼 붙입니다. 여러 근거면 [1][3]처럼 붙입니다. "
    "번호는 검색 결과에 붙은 번호를 그대로 씁니다.\n"
    "3. 문장 끝의 번호는 그 문장의 내용이 실제로 들어 있는 검색 결과의 번호만 붙입니다. "
    "검색 결과에 없는 평가나 판단(예: '가장 유명하다')은 쓰지 않습니다.\n"
    "4. 검색 결과마다 이름이나 내용이 다르게 적혀 있으면, 같은 대상이라고 검색 결과가 밝히지 않는 한 하나로 합치지 말고 "
    "'항목마다 다르게 적혀 있다'고 밝힙니다.\n"
    "5. 근거를 찾지 못했을 때 다른 항목이나 자료를 확인하라고 안내하지 않습니다. "
    "찾지 못했다는 사실과 검색에서 확인된 내용만 말합니다.\n"
    "6. 항목 코드(E로 시작하는 숫자)나 링크는 쓰지 않습니다.\n"
    "7. 존댓말로 간결하게 답합니다."
)


# 규칙 3개(번호 정확도·이름 합치기·다른 자료 안내 금지)를 더하기 전의 프롬프트. 규칙 유무를 비교할 때 쓴다.
AGENT_SYSTEM_PROMPT_BASE = (
    "당신은 한국민족문화대백과사전을 근거로 한국사 질문에 답하는 도우미입니다.\n"
    "1. 사실을 답하기 전에 search_encyclopedia 도구로 근거를 찾습니다. 검색 결과에 있는 내용만 사용해 답하고, "
    "없는 내용은 지어내지 않습니다. 근거를 찾지 못했으면 '제공된 근거로는 답할 수 없습니다'라고 말합니다.\n"
    "2. 답변의 각 문장 끝에 그 문장의 근거가 된 번호를 [1]처럼 붙입니다. 여러 근거면 [1][3]처럼 붙입니다. "
    "번호는 검색 결과에 붙은 번호를 그대로 씁니다.\n"
    "3. 항목 코드(E로 시작하는 숫자)나 링크는 쓰지 않습니다.\n"
    "4. 존댓말로 간결하게 답합니다."
)


def run_agent(llm_client, tool, question, max_turns=5, max_tokens=2000, system_prompt=AGENT_SYSTEM_PROMPT):
    messages = [{"role": "user", "content": question}]
    log = []  # 모델 호출 한 번마다 한 줄: 도구 입력, 토큰, 걸린 시간

    for turn in range(max_turns):
        started = time.time()
        response = llm_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=[SEARCH_TOOL_DEFINITION],
            messages=messages,
        )
        entry = {"turn": turn + 1, "stop": response.stop_reason,
                 "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
                 "seconds": round(time.time() - started, 2), "queries": []}
        log.append(entry)

        if response.stop_reason != "tool_use":
            # 도구 호출 없이 끝났으면 이것이 최종 답이다.
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text
            answer = attach_sources(text.strip(), tool.sources)
            return {"answer": answer, "raw": text.strip(), "stop": response.stop_reason,
                    "log": log, "sources": tool.sources}

        # 모델의 이번 응답(thinking·도구 요청 포함)을 그대로 대화에 이어 붙이고, 도구 결과를 user 메시지로 돌려준다.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                entry["queries"].append(block.input.get("query"))
                results.append(execute_tool_call(tool, block))
        messages.append({"role": "user", "content": results})

    # 상한까지 도구만 부르다 끝난 경우
    return {"answer": "", "raw": "", "stop": "max_turns", "log": log, "sources": tool.sources}
