"""data/sections.jsonl의 섹션을 청크로 만든다.
"""

import json
import re
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_PATH = Path("data/tokenizer/bge-m3-tokenizer.json")
SECTIONS_PATH = Path("data/sections.jsonl")
CHUNKS_PATH = Path("data/chunks.jsonl")
MAX_TOKENS = 512  # 청크 상한
MIN_TOKENS = 100  # 이보다 짧으면 다음 섹션과 합친다
MAX_ALIASES = 5  # embedding_text 접두사에 넣을 별칭 수

BLANK_LINE_RE = re.compile(r"\n\s*\n")
ENTRY_LINK_RE = re.compile(r"\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\(E\d{7}\)")  # [용어](E0072826), 용어 안에 [현 지명]이 들어간 [천마산[762m]](E…)도
FOOTNOTE_MARK_RE = re.compile(r"\[\^\w+\]")  # [^1]
BR_RE = re.compile(r"<br\s*/?>", re.I)
TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
LINE_END_SPACE_RE = re.compile(r"[ \t]+\n")
MULTI_NEWLINE_RE = re.compile(r"\n{2,}")

_tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))  # 17MB 파일이라 한 번만 읽는다


def count_tokens(text: str) -> int:
    """bge-m3 토크나이저로 text의 토큰 수를 센다. 모델이 앞뒤에 붙이는 특수 토큰은 세지 않는다."""
    return len(_tokenizer.encode(text, add_special_tokens=False).ids)


def split_paragraphs(text: str) -> list[str]:
    """text를 빈 줄 기준으로 문단 목록으로 나눈다. HTML 표도 따로 다루지 않는다."""
    return [p.strip() for p in BLANK_LINE_RE.split(text) if p.strip()]


def chunk_section(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """섹션 글을 max_tokens 이하 덩어리로 나눈다. 문단 경계에서만 끊고, 문단 하나가 max_tokens를 넘으면 그대로 둔다."""
    chunks, current, current_tokens = [], [], 0
    for paragraph in split_paragraphs(text):
        tokens = count_tokens(paragraph)
        if current and current_tokens + tokens > max_tokens:
            chunks.append("\n\n".join(current))  # 이 문단을 넣으면 넘치니 지금까지 것을 확정
            current, current_tokens = [], 0
        current.append(paragraph)
        current_tokens += tokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def merge_short(pieces: list[dict], min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS) -> list[dict]:
    merged = []
    current = None
    for piece in pieces:
        if current is not None:
            heading = "#" * len(piece["path"]) + " " + piece["path"][-1] + "\n" if piece["path"] else ""
            candidate = current["text"] + "\n\n" + heading + piece["text"]
            if count_tokens(current["text"]) < min_tokens and count_tokens(candidate) <= max_tokens:
                current["text"] = candidate
                continue
            merged.append(current)
        current = dict(piece)  # 원본 덩어리를 바꾸지 않도록 복사
    if current is not None:
        merged.append(current)
    return merged


def clean_text(text: str) -> str:
    text = ENTRY_LINK_RE.sub(r"\1", text)
    text = FOOTNOTE_MARK_RE.sub("", text)
    text = BR_RE.sub("\n", text)
    text = TAG_RE.sub(" ", text)  # 표 셀이 붙어 버리지 않게 공백으로 바꾼다
    text = MULTI_SPACE_RE.sub(" ", text)
    text = LINE_END_SPACE_RE.sub("\n", text)  # 표 태그를 지운 자리에 남는 줄 끝 공백
    text = MULTI_NEWLINE_RE.sub("\n", text)  # 표 셀 사이의 빈 줄. 문단 경계도 줄바꿈 하나가 되지만 의미는 같다
    return text.strip()


def make_embedding_text(text: str, headword: str, aliases: list[str], path: list[str],
                        max_aliases: int = MAX_ALIASES) -> str:
    prefix = headword
    if aliases:
        prefix += " (" + ", ".join(aliases[:max_aliases]) + ")"
    if path:
        prefix += " > " + " > ".join(clean_text(t).replace("\n", " ") for t in path)
    return prefix + "\n" + clean_text(text)


def chunk_item(item: dict) -> list[dict]:
    pieces = []  # (kind, path, text)
    body_pieces = []
    for sec in item["sections"]:
        if sec["kind"] == "body":
            body_pieces += [{"path": sec["path"], "text": t} for t in chunk_section(sec["text"])]
        else:
            pieces.append((sec["kind"], sec["path"], sec["text"]))
    pieces += [("body", p["path"], p["text"]) for p in merge_short(body_pieces)]

    chunks = []
    for n, (kind, path, text) in enumerate(pieces):
        embedding_text = make_embedding_text(text, item["headword"], item["aliases"], path)
        chunks.append({
            "chunk_id": f"{item['eid']}-{n:03d}",
            "eid": item["eid"],
            "headword": item["headword"],
            "url": item["url"],
            "field": item["field"],
            "era": item["era"],
            "primary_type": item["primary_type"],
            "kind": kind,
            "path": path,
            "text": text,
            "embedding_text": embedding_text,
            "tokens": count_tokens(embedding_text),
        })
    return chunks


def chunk_all(sections_path: Path = SECTIONS_PATH, out_path: Path = CHUNKS_PATH) -> int:
    """sections_path의 모든 항목을 chunk_item으로 청크로 만들어 out_path에 한 줄에 청크 하나씩 쓴다. 청크 수를 돌려준다."""
    tmp = out_path.with_suffix(".tmp")
    n = 0
    with sections_path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as out:
        for line in src:
            for chunk in chunk_item(json.loads(line)):
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n += 1
    tmp.replace(out_path)  # 다 쓴 뒤 이름을 바꾼다. 쓰다가 끊겨도 반쪽 파일이 남지 않는다
    return n


if __name__ == "__main__":
    print(f"청크 {chunk_all()}개 → {CHUNKS_PATH}")
