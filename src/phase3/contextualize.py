"""청크마다 LLM이 항목 안에서의 위치·맥락을 한 줄로 쓴다 (Contextual Retrieval)."""

import json
import random
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

MODEL = "claude-sonnet-5-5"

# 모든 청크에 같은 지시문. 바뀌지 않는 부분이라 항목 글 뒤가 아니라 앞(시스템)에 둔다.
SYSTEM_PROMPT = (
    "당신은 한국민족문화대백과사전 항목을 검색하기 좋게 돕는 편집자입니다. "
    "항목 전체 글과 그중 한 조각(청크)이 주어지면, 그 조각이 항목 안에서 어떤 위치에 있고 "
    "무엇을 다루는지를 한국어 한 문장으로 씁니다. 길이는 80자 이내로 합니다. "
    "조각 내용을 요약하지 말고, 항목 이름과 조각의 주제(대상·시기·사건)만 짚습니다. "
    "조각에 없는 사실을 지어내지 않습니다. "
    "문장 하나만 출력하고 따옴표나 머리말을 붙이지 않습니다."
)


def make_context(client, document_text, chunk_text, max_tokens=120):
    # 항목 전체 글(document_text)은 같은 항목의 모든 청크에서 같아서 cache_control을 붙인다.
    # 청크마다 바뀌는 chunk_text는 그 뒤에 둔다 (캐시는 앞부분이 같아야 적용된다).
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<document>\n{document_text}\n</document>",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": f"위 항목에서 다음 조각의 위치와 맥락을 설명하세요.\n<chunk>\n{chunk_text}\n</chunk>",
                },
            ],
        }],
    )
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text
    return text.strip(), response.usage, response.stop_reason


CHUNKS_PATH = Path(__file__).parent.parent.parent / "data" / "chunks.jsonl"
QUERIES_PATH = Path(__file__).parent / "eval_queries.json"


def select_entries(random_count, seed=42):
    queries = json.load(open(QUERIES_PATH, encoding="utf-8"))
    answer_eids = set()
    for query in queries:
        for eid in query["answer_eids"]:
            answer_eids.add(eid)

    # 항목마다 청크 수를 센다 (chunks.jsonl은 한 줄이 청크 하나)
    chunk_counts = {}
    for line in open(CHUNKS_PATH, encoding="utf-8"):
        chunk = json.loads(line)
        eid = chunk["eid"]
        chunk_counts[eid] = chunk_counts.get(eid, 0) + 1

    # 정답이 아닌 항목 중에서 무작위로 뽑는다. sorted로 순서를 고정해야 seed가 같은 결과를 낸다.
    others = sorted(eid for eid in chunk_counts if eid not in answer_eids)
    rng = random.Random(seed)
    picked = rng.sample(others, random_count)

    selected = sorted(answer_eids) + sorted(picked)
    return selected, chunk_counts


CONTEXTS_PATH = Path(__file__).parent.parent.parent / "data" / "contexts.jsonl"

# claude-sonnet-5-5 가격 (100만 토큰당 달러). 캐시 쓰기는 입력의 1.25배(5분 캐시).
PRICE_INPUT = 2.0
PRICE_CACHE_WRITE = 2.5
PRICE_CACHE_READ = 0.2
PRICE_OUTPUT = 10.0


def build_document(chunks, headword=None):
    # 한 항목의 청크들을 순서대로 이어 항목 전체 글을 만든다. 소제목 경로는 "## 경로"로 붙인다.
    parts = []
    for chunk in chunks:
        title = " > ".join(chunk["path"])
        if title:
            parts.append(f"## {title}\n{chunk['text']}")
        else:
            parts.append(chunk["text"])
    document = "\n\n".join(parts)
    if headword:
        document = f"# 항목명: {headword}\n\n{document}"
    return document


def usage_cost(usage):
    # 응답의 토큰 사용량을 달러로 바꾼다. 캐시 항목은 없을 때 None이 올 수 있어 0으로 바꾼다.
    cache_write = usage.cache_creation_input_tokens or 0
    cache_read = usage.cache_read_input_tokens or 0
    total = (
        usage.input_tokens * PRICE_INPUT
        + cache_write * PRICE_CACHE_WRITE
        + cache_read * PRICE_CACHE_READ
        + usage.output_tokens * PRICE_OUTPUT
    )
    return total / 1_000_000


def run_contexts(client, eids, output_path=CONTEXTS_PATH, budget=20.0, dry_run=False):
    done = set()
    if output_path.exists():
        for line in open(output_path, encoding="utf-8"):
            done.add(json.loads(line)["chunk_id"])

    # 대상 항목의 청크를 항목별로 모은다 (chunks.jsonl 파일 순서 = 항목 안 청크 순서)
    wanted = set(eids)
    by_entry = {}
    for line in open(CHUNKS_PATH, encoding="utf-8"):
        chunk = json.loads(line)
        if chunk["eid"] in wanted:
            if chunk["eid"] not in by_entry:
                by_entry[chunk["eid"]] = []
            by_entry[chunk["eid"]].append(chunk)

    total = 0
    for eid in eids:
        total += len(by_entry[eid])
    remaining = total - len(done & {c["chunk_id"] for e in by_entry for c in by_entry[e]})
    print(f"대상 항목 {len(eids)}개, 청크 {total:,}개, 이미 만든 것 {total - remaining:,}개, 남은 호출 {remaining:,}번")
    if dry_run:
        return

    spent = 0.0
    made = 0
    truncated = 0
    started = time.time()
    with open(output_path, "a", encoding="utf-8") as out:
        for eid in eids:
            chunks = by_entry[eid]
            pending = []
            for chunk in chunks:
                if chunk["chunk_id"] not in done:
                    pending.append(chunk)
            if len(pending) == 0:
                continue

            document = build_document(chunks)
            for chunk in pending:
                if spent >= budget:
                    print(f"비용 상한 ${budget:.2f}에 닿아 멈춥니다. (만든 청크 {made:,}개, 누적 ${spent:.2f})")
                    return
                try:
                    context, usage, stop_reason = make_context(client, document, chunk["text"])
                except anthropic.APIError as error:
                    print(f"API 오류로 멈춥니다: {chunk['chunk_id']} | {type(error).__name__} | {str(error)[:200]}")
                    return

                cost = usage_cost(usage)
                spent += cost
                made += 1
                if stop_reason == "max_tokens":
                    truncated += 1
                record = {"chunk_id": chunk["chunk_id"], "eid": eid, "context": context,
                          "stop_reason": stop_reason, "cost": round(cost, 6)}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()   # 한 줄마다 디스크에 써서, 중간에 끊겨도 여기까지는 남는다

                if made % 100 == 0:
                    elapsed = time.time() - started
                    print(f"  {made:,}/{remaining:,}개 | 누적 ${spent:.2f} | {elapsed/60:.1f}분 | 잘림 {truncated}개")

    elapsed = time.time() - started
    print(f"완료: {made:,}개 | 누적 ${spent:.2f} | {elapsed/60:.1f}분 | 잘림 {truncated}개")


def fix_contexts(client, path=CONTEXTS_PATH, max_tokens=600, dry_run=False):
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    bad_ids = set()
    for record in records:
        if record["stop_reason"] == "max_tokens" or not record["context"]:
            bad_ids.add(record["chunk_id"])
    print(f"다시 만들 청크 {len(bad_ids)}개: {sorted(bad_ids)}")
    if dry_run or len(bad_ids) == 0:
        return

    # 해당 청크가 속한 항목의 청크를 모은다 (문서 전체를 다시 조립해야 해서)
    bad_eids = {chunk_id.split("-")[0] for chunk_id in bad_ids}
    by_entry = {}
    for line in open(CHUNKS_PATH, encoding="utf-8"):
        chunk = json.loads(line)
        if chunk["eid"] in bad_eids:
            by_entry.setdefault(chunk["eid"], []).append(chunk)

    fixed = {}
    spent = 0.0
    for eid in sorted(by_entry):
        chunks = by_entry[eid]
        document = build_document(chunks, headword=chunks[0]["headword"])
        for chunk in chunks:
            if chunk["chunk_id"] not in bad_ids:
                continue
            context, usage, stop_reason = make_context(client, document, chunk["text"], max_tokens=max_tokens)
            cost = usage_cost(usage)
            spent += cost
            ok = stop_reason != "max_tokens" and len(context) > 0
            print(f"  {chunk['chunk_id']} 「{chunk['headword']}」 {'성공' if ok else '실패(기존 유지)'} | {stop_reason} | 출력 {usage.output_tokens}토큰 | {context[:60]}")
            if ok:
                fixed[chunk["chunk_id"]] = {"chunk_id": chunk["chunk_id"], "eid": eid, "context": context,
                                            "stop_reason": stop_reason, "cost": round(cost, 6)}

    # 성공한 것만 기존 레코드를 바꿔 쓴다. 임시 파일에 먼저 쓰고 바꿔서 도중에 끊겨도 원본이 남는다.
    temp_path = path.with_suffix(".jsonl.tmp")
    with open(temp_path, "w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(fixed.get(record["chunk_id"], record), ensure_ascii=False) + "\n")
    temp_path.replace(path)
    print(f"성공 {len(fixed)}개 / 대상 {len(bad_ids)}개 | 이번 비용 ${spent:.3f}")


def main():
    # 작은 실험: 정답 항목 + 무작위 500개 항목의 청크에 문맥을 만든다.
    selected, counts = select_entries(500)
    client = anthropic.Anthropic()
    run_contexts(client, selected, budget=20.0)


if __name__ == "__main__":
    main()
