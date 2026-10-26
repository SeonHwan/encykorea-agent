"""답변 평가셋(eval_answers.json)을 세 조건으로 돌려 data/ 아래에 한 줄씩 저장한다.
조건: S=단일 호출, A=에이전트(규칙 있음), B=에이전트(규칙 없음)
프로젝트 루트에서 실행한다:  uv run python src/phase5/run_eval.py S|A|B [질문id,질문id,...]
이미 저장된 질문은 건너뛰므로 끊겨도 같은 명령으로 이어 받는다."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
for folder in ["phase5", "phase4", "phase2"]:
    sys.path.insert(0, str(ROOT / "src" / folder))

from qdrant_client import QdrantClient

from agent import AGENT_SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT_BASE, SearchTool, run_agent
from answer import answer_question, make_clients
from collection import QDRANT_URL
from encode_test import load_model

QUESTIONS_PATH = ROOT / "src" / "phase5" / "eval_answers.json"
# B(규칙 없음)는 어려운 유형만 돌린다. 쉬운 사실형과 대조 질문은 규칙 유무로 달라질 가능성이 낮아 뺀다.
SKIP_KINDS_FOR_B = ["쉬운사실", "대조"]


def load_done(path):
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["id"])
    return done


def eids_of(sources):
    # sources의 chunk_id("E0000025-000")에서 항목 eid만 뽑는다.
    return [source["chunk_id"].split("-")[0] for source in sources]


def main():
    condition = sys.argv[1]
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    if condition not in ["S", "A", "B"]:
        raise ValueError("조건은 S, A, B 중 하나여야 합니다.")

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    out_path = ROOT / "data" / f"answers_eval_{condition}.jsonl"
    done = load_done(out_path)

    claude_client, _ = make_clients()
    search_client = QdrantClient(url=QDRANT_URL)
    embed_model = load_model()

    for question in questions:
        if question["id"] in done:
            continue
        if only is not None and question["id"] not in only:
            continue
        if condition == "B" and question["kind"] in SKIP_KINDS_FOR_B:
            continue

        if condition == "S":
            result = answer_question(search_client, embed_model, claude_client, "claude", question["query"])
            record = {"answer": result["answer"], "raw": result["raw"], "stop": result["stop"],
                      "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
                      "calls": 1, "sources": result["sources"]}
        else:
            tool = SearchTool(search_client, embed_model)
            prompt = AGENT_SYSTEM_PROMPT if condition == "A" else AGENT_SYSTEM_PROMPT_BASE
            result = run_agent(claude_client, tool, question["query"], system_prompt=prompt)
            record = {"answer": result["answer"], "raw": result["raw"], "stop": result["stop"],
                      "input_tokens": sum(e["input_tokens"] for e in result["log"]),
                      "output_tokens": sum(e["output_tokens"] for e in result["log"]),
                      "calls": len(result["log"]), "log": result["log"], "sources": result["sources"]}

        record["id"] = question["id"]
        record["kind"] = question["kind"]
        record["query"] = question["query"]
        record["retrieved_eids"] = eids_of(record["sources"])
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("done", condition, question["id"], record["stop"], flush=True)


if __name__ == "__main__":
    main()
