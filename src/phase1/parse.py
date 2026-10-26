"""항목 JSON(data/raw)을 섹션으로 나눈다.
"""

import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/sections.jsonl")
HEADING_RE = re.compile(r"^(#+)\s*(.*?)\s*$")  # \s는 전각 공백(　)도 공백으로 본다


def split_sections(body: str) -> list[dict]:
    sections = []
    stack = []  # [(소제목 단계, 제목)]
    lines = []

    def flush():
        text = "\n".join(lines).strip()
        if text:
            sections.append({"path": [t for _, t in stack], "text": text})
        lines.clear()

    for line in body.replace("\r", "").split("\n"):
        m = HEADING_RE.match(line)
        if not m:
            lines.append(line)
            continue
        flush()
        level, title = len(m.group(1)), m.group(2)
        while stack and stack[-1][0] >= level:  # 같거나 깊은 단계는 닫는다
            stack.pop()
        stack.append((level, title))
    flush()
    return sections


def parse_item(item: dict) -> dict:
    sections = []
    if item["summary"].strip():
        sections.append({"kind": "summary", "path": [], "text": item["summary"].strip()})
    sections += [{"kind": "body", **s} for s in split_sections(item["body"])]
    if not any(s["kind"] == "body" for s in sections):
        sections.append({"kind": "definition", "path": [], "text": item["definition"].strip()})
    return {
        "eid": item["eid"],
        "headword": item["headword"],
        "origin": item["origin"],
        "url": item["url"],
        "field": item["field"],
        "era": item["era"],
        "primary_type": item["primaryType"],
        "writer": item["writerInfo"],
        "modified": item["lastModifiedTime"],
        "aliases": [a["word"] for a in item["articleAliases"]],
        "sections": sections,
    }


def parse_all(raw_dir: Path = RAW_DIR, out_path: Path = OUT_PATH) -> int:
    """raw_dir의 모든 항목을 parse_item으로 정리해 out_path에 한 줄에 항목 하나씩 쓴다. 항목 수를 돌려준다."""
    tmp = out_path.with_suffix(".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as out:
        for f in sorted(raw_dir.glob("*.json")):
            parsed = parse_item(json.loads(f.read_text(encoding="utf-8")))
            out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(out_path)  # 다 쓴 뒤 이름을 바꾼다. 쓰다가 끊겨도 반쪽 파일이 남지 않는다
    return n


if __name__ == "__main__":
    print(f"{parse_all()}개 항목 → {OUT_PATH}")
