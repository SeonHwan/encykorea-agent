# encykorea-agent

「한국민족문화대백과사전」(API로 받을 수 있는 15,836건)을 근거로 한국사 질문에 출처를 달아 답하는 에이전트.
RAG(검색 증강 생성)와 도구를 쓰는 에이전트를 처음부터 만들며 배우는 학습 프로젝트다.

## AI Agent 단계별 프로젝트

AI Agent를 단계별로 만들어 가는 5개 프로젝트 중 **3번째**입니다. 앞 프로젝트에서 남은 한계가 다음 프로젝트를 시작한 이유입니다.

| 순서 | 프로젝트 | 다룬 것 | 남은 한계 → 다음 단계 |
|---|---|---|---|
| 1 | [game-wiki-rag](https://github.com/SeonHwan/game-wiki-rag) | 가상 게임 위키 4문서로 RAG 기본 파이프라인, 하이브리드·리랭커·Contextual Retrieval·GraphRAG, tool-use 에이전트, MCP, LangGraph 기반 Adaptive/Self-RAG | 데이터와 평가 질문이 너무 작아(문서 4개, 질문 12개) 오케스트레이션, 답변 품질 개선, 여러 RAG 기법의 효과를 제대로 확인하기 어려움 → 실제 방대한 데이터로 다시 |
| 2 | [dart-filing-agent](https://github.com/SeonHwan/dart-filing-agent) | DART 사업보고서 5개사 15건(수만 개 청크): 표 행 단위 청킹, bge-m3 dense+sparse 하이브리드, 리랭커 측정, XBRL 값 조회 스토어 | 문서가 전문용어 중심이라 답변이 나아졌는지 확인하려면 용어부터 이해해야 해서 학습 진도가 더딤 → 일반 상식에 가까운 데이터로 |
| 3 | **[encykorea-agent](https://github.com/SeonHwan/encykorea-agent)** | 한국민족문화대백과사전 15,836건(청크 75,122개): hybrid 검색, 답변 생성 방식 비교, 직접 구현한 도구 호출 에이전트 | RAG 성격이 강해 에이전트 역량을 보여 주는 증거로 약하다고 판단 → 멀티 에이전트·오케스트레이션·LangGraph를 배우는 별도 프로젝트 |
| 4 | [logistics-agent](https://github.com/SeonHwan/logistics-agent) | 공개 물류 데이터(KTDB 화물 OD)와 정답이 계산되는 합성 시나리오: LangGraph 멀티 에이전트(supervisor, 병렬 분석가, 취합), 검증 노드, 사람 승인, Langfuse 관측, RAGAS 채점 | 병렬로 도는 에이전트를 더 늘려 멀티 에이전트의 효과를 눈으로 확인하고, MCP 분리와 서빙도 함께 다루려고 → 새 프로젝트로 |
| 5 | [kitchen-agent](https://github.com/SeonHwan/kitchen-agent) | 가상 주방 데이터와 6명의 전문 에이전트(LangGraph), 도구의 MCP 분리, RAGAS 평가, FastAPI 서빙 | — |

### 이전 단계([dart-filing-agent](https://github.com/SeonHwan/dart-filing-agent))에서 남은 한계
공시 문서는 전문용어가 대부분이라, 답변이 개선됐는지 확인하려면 용어부터 이해해야 했고 학습 진도가 더뎠습니다.

### 이번 단계에서 바꾼 것
일반 상식에 가까운 개념이 많은 한국민족문화대백과사전(API로 받을 수 있는 15,836건)으로 데이터를 바꿔, 답변 내용을 이해하면서 개선 여부를 확인하기 쉽게 했습니다. 이 데이터로 hybrid 검색과, 답변 생성 방식(단일 호출 대 직접 구현한 도구 호출 에이전트)을 비교했습니다.

### 이 프로젝트의 한계가 다음 단계로 이어진 이유
이 프로젝트는 검색과 답변 생성(RAG)의 비중이 커서 에이전트 역량을 보여 주는 증거로는 약하다고 판단했습니다. 그래서 멀티 에이전트·오케스트레이션·LangGraph를 직접 다루는 별도 프로젝트 [logistics-agent](https://github.com/SeonHwan/logistics-agent)로 넘어갔습니다.

## 구성

질문 → 모델이 `search_encyclopedia` 도구를 호출해 검색어를 만듦 → hybrid 검색(dense + sparse를 RRF로 합침) → 근거 글 → 답변 + `출처: 「항목명」, ⓒ한국민족문화대백과사전`

| 단계 | 내용 | 위치 |
|---|---|---|
| 청킹 | 수집된 15,836건 → 섹션 → 청크 75,122개 | `src/phase1/` |
| 임베딩·적재 | bge-m3 dense·sparse → Qdrant | `src/phase2/` |
| 검색 평가 | 질문 30개로 dense·sparse·hybrid 등 비교 | `src/phase3/` |
| 답변 생성 | 단일 호출 답변, Claude·GPT·로컬 모델 비교 | `src/phase4/` |
| 에이전트 | 직접 구현한 도구 호출 루프, 평가셋 30개 | `src/phase5/` |

## 실행

```bash
docker compose -f infra/compose.yaml up -d      # Qdrant 실행 (127.0.0.1:6333)
uv run python src/phase2/index.py               # data/chunks.jsonl을 bge-m3로 인코딩해 Qdrant에 적재
uv run python src/phase5/run_eval.py A          # 에이전트로 평가 질문 실행 (모델 호출 비용 발생)
```

`.env`에 `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`가 필요하다. 수집된 데이터가 프로젝트 루트의 `data/`에 있다고 가정하며, `data/`와 `.env`는 저장소에 없다. 코드가 읽는 파일은 `data/raw/`, `data/sections.jsonl`, `data/chunks.jsonl`, `data/tokenizer/bge-m3-tokenizer.json`이다.

## 배운 것과 한계

- 기본 검색은 hybrid로 정했다 (질문 30개 측정, `docs/decisions.md` 3-8).
- 답변 방식은 에이전트로 정했다. 정답 항목을 검색해 온 비율이 단일 호출 20/23, 에이전트 23/23이다. 비용은 입력 토큰 약 2.7배다 (5-5).
- **한계**: 답변 규칙 3개와 질의 재작성만의 효과는 가려내지 못했다. 에이전트가 사전 안의 서로 어긋나는 서술을 알아채지 못한 사례가 있다.
- **하지 않은 것**: 관측성 도구(Langfuse, 필요해지면 적용), 웹 UI·배포(5-7), 정형 사실 데이터(4-A, 필요가 확인되지 않음).

자세한 결정과 측정 근거는 `docs/decisions.md`, 단계별 계획은 `docs/ROADMAP.md`.

## 출처

텍스트는 공공누리 제1유형(출처 표기 조건)으로 이용한다. 미디어는 쓰지 않는다.
