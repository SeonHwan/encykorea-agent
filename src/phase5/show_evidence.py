"""채점용: 저장된 답변의 근거 글을 chunk_id로 data/chunks.jsonl에서 꺼내 번호와 함께 보여 준다.
사용: uv 없이 python으로 가능.  python src/phase5/show_evidence.py 조건(S|A|B) 질문id [번호 번호 ...]
번호는 답변의 [n]. 생략하면 전부 보여 준다. 에이전트(A/B)의 번호는 검색 호출을 이어 붙인 번호다."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def load_chunks(chunk_ids):
    wanted = set(chunk_ids)
    found = {}
    with open(ROOT / "data" / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk["chunk_id"] in wanted:
                found[chunk["chunk_id"]] = chunk
    return found


def main():
    condition, qid = sys.argv[1], sys.argv[2]
    numbers = [int(n) for n in sys.argv[3:]]
    record = None
    with open(ROOT / "data" / f"answers_eval_{condition}.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] == qid:
                record = row
    if record is None:
        print("저장된 답변이 없습니다.")
        return

    sources = {source["n"]: source for source in record["sources"]}
    chosen = numbers if len(numbers) > 0 else sorted(sources)
    chunks = load_chunks([sources[n]["chunk_id"] for n in chosen if n in sources])
    for n in chosen:
        if n not in sources:
            print(f"[{n}] (이 답변에 없는 번호)")
            continue
        chunk = chunks[sources[n]["chunk_id"]]
        path = " > ".join(chunk["path"]) if chunk["path"] else chunk["kind"]
        print(f"[{n}] 「{chunk['headword']}」 > {path}  ({chunk['chunk_id']})")
        print(chunk["text"].strip())
        print()


if __name__ == "__main__":
    main()
