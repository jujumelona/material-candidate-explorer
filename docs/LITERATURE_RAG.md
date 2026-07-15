# Live Literature RAG와 근거 기반 탐색 분기

이 기능은 논문을 후보의 물성 점수로 사용하지 않습니다. 최신 문헌은 **어디를 탐색할지 정하는 prior**이고, 생성 후보의 합격·탈락은 MatterSim·CHGNet·DFT·결합·독성·ADMET 등 해당 분야 전문 평가기가 결정합니다.

## 실행 흐름

```text
사용자 목표
  ↓
RAG 모델 검색 계획
  ├─ 핵심 개념·동의어
  ├─ 표적·기전
  ├─ 실패·독성·상충 근거
  └─ 조성·도핑·압력·합성 조건
  ↓
PubMed / Europe PMC / OpenAlex / Crossref / arXiv 병렬 검색
  ↓
DOI·PMID·arXiv ID·정규화 제목으로 중복 제거
  ↓
출처 문장에 실제 존재하는 claim만 구조화
  ↓
Evidence Knowledge Graph
  ↓
Evidence Branch Planner
  ├─ 최신 보고 성분·재료 재검증
  ├─ 유도체·유사체
  ├─ 동일 표적·기전의 다른 구조
  ├─ 새 조성·도핑·압력·합성 조건
  ├─ 실패·독성 회피
  └─ 여러 논문 관계를 연결한 미탐색 가설
  ↓
1차 생성 → 전문 모델 평가
  ↓
실제 objective 개선·구조 붕괴·실패율로 분기 가중치 갱신
  ↓
좋은 분기 확대 / 실패 분기 축소 → 다음 라운드 생성
```

모든 claim은 원문 제목·초록의 연속 문자열인 `support_text`를 보존합니다. RAG 모델이 원문에 없는 문장을 만들면 claim은 폐기됩니다. 상충하는 긍정·부정 관계는 하나로 덮지 않고 conflict edge로 남깁니다.

## RAG 모델 연결

OpenAI-compatible `chat/completions` JSON endpoint를 사용합니다. 모델은 검색 계획과 claim 구조화를 담당할 뿐, 후보 물성값을 생성하지 않습니다.

```bash
export RAG_MODEL_API_URL="https://YOUR-ENDPOINT/v1"
export RAG_MODEL_NAME="YOUR-MODEL"
export RAG_MODEL_API_KEY="YOUR-TOKEN"       # 필요한 경우만
export RAG_MODEL_TIMEOUT_SECONDS="180"
```

모델 연결이 없으면 결정론적 검색 계획·보수적 claim 추출 fallback을 사용합니다. 모델 없이는 충분한 관계 추출 품질을 보장할 수 없으므로 운영 실행에서는 `--require-model` 또는 `--rag-require-model`을 권장합니다.

## 학술 검색 설정

```bash
export LITERATURE_CONTACT_EMAIL="researcher@example.com"
export NCBI_API_KEY="..."                    # 선택, NCBI rate limit 완화
export OPENALEX_API_KEY="..."                # OpenAlex 사용 시 필요
export LITERATURE_USER_AGENT="DiscoveryOS/0.3 researcher@example.com"
```

각 제공자의 성공·부분 성공·건너뜀·실패 상태, 실제 검색식, endpoint, 결과 수와 오류는 bundle에 기록됩니다. 한 제공자가 실패해도 다른 제공자의 근거를 숨기지 않습니다.

## 최신 근거 bundle 생성

```bash
discovery-os rag-update \
  --prompt "KRAS G12D 췌장암에서 최근 보고된 억제제, 독성 회피 골격과 실패 임상 조합을 찾아라" \
  --from-date 2024-01-01 \
  --max-results 40 \
  --max-branches 30 \
  --require-model \
  --index .discovery/evidence-index \
  --output runs/pancreatic-cancer/rag_bundle.json
```

출력에는 검색 계획, 소스 상태, 중복 제거된 논문, source-grounded claim, 지식그래프, 생성기 힌트가 포함된 탐색 분기가 들어갑니다.

## 실제 Fusion search에 직접 연결

```bash
discovery-os fusion-search \
  --search-id pancreatic-001 \
  --goal goal.json \
  --parent parent.json \
  --run-config on.json \
  --generator reinvent4 \
  --rounds 8 \
  --rag-prompt "KRAS G12D 췌장암용 신규 저분자와 독성 회피 유도체를 찾아라" \
  --rag-from-date 2024-01-01 \
  --rag-max-results 40 \
  --rag-max-branches 30 \
  --rag-require-model \
  --rag-index .discovery/evidence-index \
  --artifacts runs/pancreatic-cancer
```

재료 탐색에서는 generator를 MatterGen으로 지정합니다. 저분자·의약 탐색에서는 REINVENT4/Chemformer 계열 generator hint가 전달됩니다. 실제로 연결되지 않은 생성기나 평가기는 성공으로 위장하지 않고 기존 fail-closed 경계에 따라 실패합니다.

## 분기 적응 정책

`LiteratureEvidencePolicy`는 각 evidence branch를 round/worker에 배분합니다. 초기에는 우선순위와 미탐색 보너스를 사용하고, 이후에는 전문 평가기에서 관찰된 objective 개선, collapse, 실행 실패를 반영합니다. 논문의 최신성·인용 수·LLM confidence를 material/drug score에 더하지 않습니다.

## 한계

이 검색 계층은 구성된 색인의 제목·초록·메타데이터를 검색합니다. 유료 원문 전체나 색인되지 않은 논문까지 “전 세계 모든 논문”을 보장하지 않습니다. 그래서 source별 누락과 실패를 명시적으로 보존하며, 중요한 후보의 최종 판단에는 원문 검토, 독립 계산, DFT/실험 검증이 필요합니다.
