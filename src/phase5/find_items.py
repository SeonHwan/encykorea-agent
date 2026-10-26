"""평가 질문을 만들 때 쓰는 확인 도구: 항목명·별칭으로 data/sections.jsonl에서 항목을 찾는다.
프로젝트 루트에서 실행한다: python src/phase5/find_items.py 대동법 왕건 ...
--show EID ...  항목의 섹션 본문 앞부분을 보여 준다. --dup  역사 분야에서 같은 이름의 항목이 둘 이상인 것을 보여 준다."""

import json
import sys
from collections import defaultdict
from pathlib import Path

SECTIONS_PATH = Path(__file__).parent.parent.parent / "data" / "sections.jsonl"


def load_sections():
    items = []
    with open(SECTIONS_PATH, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def find_by_name(items, name):
    # 항목명이 같거나, 별칭이 같거나, 별칭에 이름이 들어 있는 항목을 모두 돌려준다.
    found = []
    for item in items:
        if item["headword"] == name:
            found.append((item, "항목명"))
        elif name in item["aliases"]:
            found.append((item, "별칭"))
        else:
            for alias in item["aliases"]:
                if name in alias:
                    found.append((item, "별칭 일부"))
                    break
    return found


def show_item(item):
    print("=" * 70)
    print(item["headword"], item["eid"], "|", item["era"], "|", item["field"], "| 별칭", item["aliases"][:5])
    for section in item["sections"]:
        label = " > ".join(section["path"]) or section["kind"]
        text = section["text"][:230].replace("\n", " ")
        print(f"  [{label}] {text}")


def main():
    items = load_sections()
    args = sys.argv[1:]

    if len(args) > 0 and args[0] == "--show":
        by_eid = {}
        for item in items:
            by_eid[item["eid"]] = item
        for eid in args[1:]:
            if eid not in by_eid:
                print(f"[목록에 없음] {eid} (본문 링크로만 나오는 항목일 수 있다; API 목록 밖은 본문이 없다)")
                continue
            show_item(by_eid[eid])
        return

    if len(args) > 0 and args[0] == "--dup":
        groups = defaultdict(list)
        for item in items:
            if item["field"].startswith("역사/"):
                groups[item["headword"]].append((item["eid"], item["era"], item["primary_type"]))
        for name, group in groups.items():
            if len(group) >= 2:
                print(name, group[:4])
        return

    for name in args:
        found = find_by_name(items, name)
        if len(found) == 0:
            print(f"[없음] {name}")
        for item, how in found[:6]:
            print(f"[{how}] {name} → {item['headword']} {item['eid']} | {item['era']} | {item['field']} | 섹션 {len(item['sections'])}")


if __name__ == "__main__":
    main()
