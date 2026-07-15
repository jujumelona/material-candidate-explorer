# Material Candidate Explorer

여러 전문 모델을 연결해 새로운 **물질 후보를 탐색하고 검증 우선순위를 관리하는** 모델 비종속 오케스트레이션 엔진입니다. 의약 저분자, 무기·결정 소재, 초전도체, 배터리, 촉매, 고분자 및 일반 소재를 다룹니다. 내부 Python 패키지와 기존 CLI 명령은 호환성을 위해 `discovery_os`·`discovery-os` 이름을 유지합니다.

> [!IMPORTANT]
> 이 프로젝트가 내놓는 계산 결과는 **탐색 리드(lead)**이지 신물질 발견의 증명이 아닙니다. 합성, 시료 동정, 조건이 명시된 물성 측정, 대조군·반복 측정과 독립 재현을 통과하기 전에는 `계산적으로 지지됨` 이상으로 표현하지 않습니다. 의약 후보의 안전성·유효성 또는 초전도성도 계산만으로 선언할 수 없습니다.

## 무엇을 만들고 있는가

모델의 역할은 가설, 후보 생성 요청, 예측, 검증 **제안**을 구조화된 JSON으로 내는 것입니다. 실제 실행 권한은 코드에만 있습니다.

```text
사용자 목표
  ↓
DiscoveryModel: 가설·후보·ValidationPlan 제안
  ↓  엄격한 Pydantic 스키마 검사
결정론적 PlanCompiler
  ├─ 공통 표현·소분자 RDKit·조성 화학식 sanity call 자동 삽입
  ├─ ToolRegistry allowlist와 operation별 조건 검사
  ├─ 후보 참조·의존성·시간·자원 예산 검사
  └─ 허용된 typed task만 생성
  ↓
코드 소유 어댑터 실행
  ↓
EvidenceRecord 정규화 + 버전·해시·artifact 기록
  ↓
결정론적 증거 정책과 게이트 판정
  ├─ 전체 도메인 프로필을 실제 EvidenceRecord와 대조
  └─ 필수 증거 누락은 insufficient_evidence로 차단
  ↓
판정된 사실·충돌·불확실성만 모델에 다시 전달
```

`tool_name`과 `operation`은 셸 명령이나 import 경로가 아닙니다. 컴파일러가 정적으로 등록된 조합과 일치시킬 때만 실행됩니다. 모델이 출력한 Python, 셸, 파일 경로, URL, 임의 환경 변수 또는 장비 명령은 실행 입력으로 받아들이지 않습니다. 모델의 우선순위·정보 이득·결과 해석·중지 의견도 참고값이며, 필수 검증과 최종 증거 등급을 건너뛸 권한은 없습니다.

## 현재 범위

이 문서에서는 상태를 다음처럼 구분합니다.

| 표기 | 의미 |
|---|---|
| **실행 가능 MVP** | 이 저장소에서 로컬로 실행하고 결과를 재현할 수 있는 통합 경로 |
| **격리 sidecar 실행 경로** | 저장소가 제공하는 `/v1/features`·`/v1/generate` 서버가 설치된 upstream 모델을 실제 호출하는 경로 |
| **연결 규격 준비·backend 필요** | 계약과 실패 경계는 실행 가능하지만 모델별 checkpoint/config 또는 사용자 Fusion backend가 더 필요한 경로 |
| **커넥터 자리** | 계약과 연결 위치만 정의됐거나 향후 어댑터가 필요한 계산·DB·모델 기능 |
| **실험실 증거** | 이 소프트웨어가 생성할 수 없으며, 승인된 실험과 출처가 있는 결과를 가져와야 하는 단계 |

현재 MVP의 목적은 실제 AI 모델이나 고비용 시뮬레이터 없이도 전체 계약, allowlist, 증거 경계와 실패 처리를 검증하는 것입니다.

- **실행 가능 MVP**: 엄격한 버전형 스키마, mock 모델·생성기, 결정론적 계획 컴파일과 registry 실행, 정규화된 증거, artifact·해시·캐시, 도메인 프로필 조회, 기본 화학식 검사와 선택적 RDKit 저분자 기본 검사.
- **격리 sidecar 실행 경로**: MatterGen, REINVENT4, Uni-Mol, UMA, MatterSim, CHGNet, Chemprop, ESM3, RNA-FM, PySCF, Boltz, scGPT, QHNet 어댑터가 실제 공개 API/CLI 또는 manifest에 고정된 source facade를 호출합니다. lazy checkpoint 로딩, 요청 크기·동시성·queue·timeout 제한, strict payload/provenance 검사를 포함합니다.
- **선택적 원격 backend·별도 checkpoint**: `/v1/fuse`·`/v1/revise`를 제공하는 학습된 Fusion AI는 선택 사항이며, URL이 없으면 로컬 Evidence controller가 동작합니다. 모델별 수동/관리 checkpoint는 별도입니다. Boltz 2.2.1은 고정 CLI/YAML/result codec으로 실제 계산합니다. scGPT 0.2.4는 고정 model/vocab/config bundle로 실제 cell embedding을 계산합니다. QHNet은 고정 AIRS source와 사용자가 선택한 checkpoint/config bundle로 전체 Hamiltonian을 계산하며, 필수 파일이나 dataset 범위가 맞지 않으면 합성 결과 없이 실패합니다.
- **커넥터 자리**: 데이터베이스 신규성 검색, OpenMM·pymatgen·spglib·DFT·Quantum ESPRESSO/EPW 및 phonon 기능은 catalog/profile에 `unavailable` placeholder로 나타납니다. 이름이 있다는 것은 설치 또는 구현됐다는 뜻이 아닙니다.
- **실험실 증거**: 합성, 시료 identity·purity·상 분율 확인, 생물학적 assay, 독성·안전성, 전기·자기·전기화학·기계·열 물성 측정과 독립 연구실 재현. MVP는 이 증거를 만들어 내거나 흉내 내지 않습니다. `ExperimentalEvidenceImporter`는 원자료 attachment와 애플리케이션 소유 verifier가 확인한 제출만 `success`로 기록하며, verifier가 없거나 검증에 실패한 제출은 `partial/unverified` 또는 `failed/rejected`로 남아 게이트를 통과하지 못합니다.

분야별 필수·권장 게이트와 현재 구현 경계는 [검증 행렬](docs/VALIDATION_MATRIX.md)에 정리되어 있습니다. `profiles` 명령에 프로필이 보인다는 사실은 그 프로필에 적힌 모든 외부 검증기가 연결됐다는 뜻이 아닙니다.

학습 중인 실제 모델의 8개 로컬/HTTP endpoint와 JSON 출력 규격은 [모델 연결 계약](docs/MODEL_CONTRACT.md)에 정리되어 있습니다.

## Unified Scientific Fusion Core

전문 모델의 가중치를 하나의 거대 모델로 합치지 않습니다. 각 엔진은 충돌하지 않는 독립 환경/API에 남고, 엄격한 의미 정보가 붙은 특징만 사용자가 학습한 Fusion AI로 전달됩니다.

```text
ScientificWorkspace
  ├─ primary candidate: 수정할 분자·결정·서열·세포 상태
  ├─ target/context/environment: 표적 단백질, 용매, assay 조건 등
  └─ typed relations: binds_to, evaluated_in, interacts_with 등
        ↓ 각 엔터티를 명시적 ExpertRoute로 전달
Uni-Mol / Boltz / ESM / RNA-FM / scGPT / QHNet / UMA / MatterSim / Chemprop
        ↓ ExpertFeaturePayload + FeatureSemantics
로컬 Evidence controller 또는 원격 FusionBackend (`/v1/fuse`, `/v1/revise`)
        ↓ UnifiedLatentStateRef + FusionRevisionProposal
허용된 생성기 (`/v1/generate`)가 부모를 인용한 새 Candidate 생성
        ↓
전문 특징 재추출 → 이전 latent와 결합 → 다음 latent 갱신
```

`FeatureSemantics`에는 tensor 역할, projection ID, entity/mask, pooling·정규화, 좌표계·basis·단위 의미가 들어갑니다. 로컬 Evidence controller는 tensor를 결합하지 않으며, 원격 Fusion AI를 선택한 경우에도 의미가 다른 tensor를 차원만 같다는 이유로 섞지 않도록 이 정보를 검사해야 합니다.

`FusionLoopRunner`는 후보 배치에 대해 **revision → generator → 새 후보 특징 재추출 → latent 갱신**을 실행합니다. `FusionSearchRunner`는 안정성·목표 물성·신규성·전문가 불일치·Pareto 풀을 분리하고 원본 전문가 출력을 `ExpertEvidenceStore`에 보존합니다. `WorkspaceBenchmarkRunner`도 같은 `candidate_count`의 OFF/ON 후보 배치를 명시적 `pair_slot`으로 짝지어 전부 재평가하며, 단일 후보 호출에는 기존 단수 접근자를 유지합니다. 두 arm의 batch seed·runtime parameter hash·provenance·누락 expert·OOD가 맞지 않으면 비교를 닫습니다. embedding, latent, Workspace ON 변화량은 모두 `diagnostic_only`이며 `EvidenceRecord`나 발견 증거가 아닙니다.

`FUSION_API_URL`이 없으면 학습 가중치 없이 전문가 평가 요약만 탐색 상태로 만드는 결정론적 `EvidenceDrivenFusionBackend`를 자동 사용합니다. URL을 설정하면 기존 원격 Fusion AI와 선택적 `FUSION_API_TOKEN`을 그대로 사용합니다. 생성기는 manifest에 고정된 `MATTERGEN_API_URL` 또는 `REINVENT_API_URL`과 대응 토큰으로 연결합니다. 자세한 구조는 [Fusion Core 규격](docs/FUSION_CORE.md), wire 형식은 [Expert/Fusion/Generator API 계약](docs/EXPERT_API_CONTRACT.md)에 있습니다.

서비스를 연결한 뒤에는 `fusion-iterate`가 한 번의 후보 배치 폐루프를, `fusion-search`가 다중 가지 반복 탐색을, `fusion-pair`가 실제 OFF/ON 생성·재평가 쌍을 실행합니다. 저장된 snapshot만 다시 비교할 때도 artifact 원본을 반드시 재검증합니다.

```powershell
discovery-os fusion-iterate --goal goal.json --parent parent.json --run-config on.json --generator mattergen --artifacts runs/fusion
discovery-os fusion-search --search-id run-001 --goal goal.json --parent parent.json --run-config on.json --generator mattergen --rounds 5 --artifacts runs/fusion
discovery-os fusion-pair --goal goal.json --parent parent.json --off-config off.json --on-config on.json --generator mattergen --artifacts runs/fusion
discovery-os fusion-compare --goal goal.json --off-snapshot off.json --on-snapshot on.json --artifact-root runs/fusion
```

충돌하는 Python·Torch·CUDA·NumPy 버전은 모델별 환경으로 분리합니다. Windows에서 공개 구성요소를 한 번에 **설치·다운로드 시도**하는 명령은 다음과 같습니다. Linux 전용 구성요소가 포함되면 구성된 WSL 배포판을 사용할 수 있습니다.

```powershell
.\bootstrap.ps1 -Profile all-open -Accelerator cuda -IncludeWeights -AcceptLicense esm
```

설치가 끝난 환경에서 각 모델 서버를 기동하고 중앙 클라이언트용 URL 환경 파일을 생성합니다.

```powershell
.\start-sidecars.ps1 -Component mattergen,unimol,esm,rnafm,pyscf -Backend auto
. .\.discovery\wsl\sidecars.env.ps1   # 실제 InstallRoot에 생성된 파일 사용
```

Linux에서는 같은 구성요소를 `./start-sidecars.sh --component ...`로 지정합니다. `all-open` 전체 기동은 실패 닫힘 모델 또는 수동 checkpoint가 하나라도 남으면 시작 전에 거부됩니다. 모델별 실제 지원 범위와 필수 checkpoint 환경변수는 [sidecar 실행 문서](docs/SIDECARS.md)에 있습니다.

UMA까지 받으려면 Hugging Face 접근 승인과 토큰이 별도로 필요합니다. 설치기가 약관을 자동 수락하거나 token을 파일에 저장하지 않습니다.

```powershell
$env:HF_TOKEN = "<사용자 토큰>"
.\bootstrap.ps1 -Profile all -Accelerator cuda -IncludeWeights -AcceptLicense esm,uma
```

다운로드 전 계획과 고정 버전을 확인할 수 있습니다. 완전성이 필요하면 `-RequireAll`도 지정하십시오. 수동 checkpoint, gated weight, upstream 관리 weight 또는 지원되지 않는 플랫폼이 남으면 성공으로 숨기지 않고 부분 상태로 끝나야 합니다.

```powershell
.\bootstrap.ps1 -Profile all-open -Accelerator cuda -DryRun -AcceptLicense esm
python scripts/bootstrap.py verify-manifest
discovery-os integrations --profile all-open
discovery-os experts
```

정확한 버전·commit·가중치 revision·archive SHA-256·라이선스·API 환경변수는 [격리 환경 및 설치 문서](docs/DEPENDENCIES.md)와 [`integrations/manifest.v1.json`](integrations/manifest.v1.json)에 있습니다.

> [!NOTE]
> 원클릭 설치는 패키지·검증된 source·허용된 weight와 공통 sidecar runtime을 모델별 환경에 준비합니다. 서버 기동은 별도 `start-sidecars` 명령이며, 수동/관리 checkpoint와 라이선스·credential은 사용자가 명시해야 합니다. 설치 후 생성되는 `*.freeze.txt`는 결과 스냅샷이지 hash가 포함된 사전 dependency lock이 아닙니다.

## Live Literature RAG와 최신 근거 기반 탐색

실행 시점의 최신 문헌을 검색해 **탐색 공간을 다시 구성**할 수 있습니다. RAG 모델이 프롬프트를 학술 검색식으로 확장하고, PubMed·Europe PMC·OpenAlex·Crossref·arXiv 결과를 통합한 뒤, 원문 제목·초록에 실제 존재하는 문장만 claim으로 보존합니다. 지식그래프에서 최신 성분·표적·기전·실패/독성 근거·조성·도핑·압력 조건과 논문 간 연결 가설을 탐색 분기로 만들고, 1차 생성 전과 라운드 사이에 `FusionSearchRunner`로 전달합니다.

논문 근거는 물성 점수에 합산하지 않습니다. 실제 생성 후보의 우열은 기존 전문 평가기의 objective 결과가 결정하고, `LiteratureEvidencePolicy`는 그 개선·붕괴·실패를 관찰해 다음 라운드에서 좋은 근거 분기를 확대하고 실패 분기를 축소합니다.

```bash
discovery-os rag-update --prompt "최근 KRAS G12D 췌장암 억제제와 독성 회피 골격을 찾아라" --from-date 2024-01-01 --require-model --output runs/rag_bundle.json

discovery-os fusion-search --search-id run-001 --goal goal.json --parent parent.json --run-config on.json --generator reinvent4 --rounds 8 --rag-prompt "최근 KRAS G12D 췌장암 억제제와 유도체를 탐색하라" --rag-from-date 2024-01-01 --rag-require-model --artifacts runs/fusion
```

RAG 모델과 학술 API 환경변수, source provenance, fail policy와 전체 데이터 흐름은 [Live Literature RAG 문서](docs/LITERATURE_RAG.md)에 있습니다.

## 빠른 시작

Python 3.11 또는 3.12를 사용합니다. Windows PowerShell 예시입니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[chem,dev]"
```

RDKit가 필요 없는 최소 설치는 다음과 같습니다.

```powershell
python -m pip install -e ".[dev]"
```

macOS/Linux에서는 활성화 명령만 `source .venv/bin/activate`로 바꾸면 됩니다.

등록된 도메인 프로필과 wire schema를 확인합니다.

```powershell
discovery-os profiles
discovery-os schema ValidationPlan
```

API 키나 실제 모델 없이 1회 mock 사이클을 실행합니다.

```powershell
discovery-os demo --goal "표적 단백질 결합 가능성이 있는 저분자 후보를 검토한다" --domain medicinal_chemistry --max-cycles 1 --output runs
```

`--domain`을 생략하면 mock 목표 컴파일러가 지원 범위 안에서 추론합니다. 명시할 수 있는 도메인은 다음과 같습니다.

```text
medicinal_chemistry   inorganic_materials   superconductors
batteries             catalysts             polymers
general_materials
```

이 demo는 통합 시험입니다. 고정된 mock 제안을 스키마로 검증하고, 허용된 로컬 어댑터만 실행해 `runs` 아래에 provenance와 증거를 남깁니다. 결과에 등장하는 후보·점수·속성은 과학적 발견이나 실험 결과가 아닙니다. 외부 실험 증거가 없으므로 demo가 `experimentally_validated` 또는 독립 재현 완료로 승격되어서는 안 됩니다.

테스트는 다음처럼 실행합니다.

```powershell
pytest
```

### 물질 후보 탐색 정확한 실행 순서

결정·재료 후보를 실제 sidecar로 탐색할 때는 아래 순서를 따릅니다.

```powershell
# 1) 의존성 계획 확인
python scripts/bootstrap.py verify-manifest

# 2) MatterGen·전문 평가 sidecar를 별도 환경에서 기동
.\start-sidecars.ps1 -Component mattergen,mattersim,chgnet -Backend auto
. .\.discovery\wsl\sidecars.env.ps1

# 3) 생성기와 평가 expert URL이 설정됐는지 확인
discovery-os integrations --profile materials-open
discovery-os experts

# 4) ON 폐루프 탐색 실행
discovery-os fusion-search `
  --search-id material-run-001 `
  --goal goal.json `
  --parent parent.json `
  --run-config run-config.json `
  --generator mattergen `
  --rounds 4 `
  --frontier-width 6 `
  --artifacts runs/material-run-001
```

`goal.json`에는 목표 물성과 단위를, `parent.json`에는 CIF 또는 화학식 후보를, `run-config.json`에는 `workspace_mode=on`, `candidate_count`, evaluator panel, MatterGen checkpoint provenance를 넣습니다. `FUSION_API_URL`을 설정하지 않으면 로컬 `EvidenceDrivenFusionBackend`가 자동 사용됩니다. 생성 결과와 CIF·전문가 원본 결과·latent·분기 이력은 `runs/material-run-001` 아래에 저장됩니다.

이 실행은 신물질 발견 확정이 아니라 후속 DFT·구조 이완·실험으로 넘길 후보 shortlist를 만드는 단계입니다. 실제 `energy_above_hull` 계산과 대칭 허용오차 중복 제거는 별도 고정밀 connector가 필요합니다.

## Adaptive alpha·temperature 후보 루프

`fusion-search`는 기본적으로 한 부모 후보를 한 번만 생성하지 않습니다. 각 라운드에서 adaptive scheduler의 현재 제어값을 중심으로 여러 alpha·temperature 지점을 실행하고, 생성된 모든 후보를 하나의 공용 후보 풀에 합칩니다.

기본 지점은 다음 세 개입니다.

```text
alpha=0.25, temperature=1.40  탐색 폭 확대
alpha=0.50, temperature=1.00  기준점
alpha=0.75, temperature=0.70  조건 집중
```

이 값은 다음 라운드에서 고정 반복되지 않습니다. 직전 라운드의 개선도, 구조 붕괴율, 전문가 불일치에 따라 branch별 scheduler가 이동한 뒤 같은 상대 지점을 다시 적용합니다. 각 시도는 별도 generator seed와 provenance를 가지며, 일부 제어점이 실패해도 다른 제어점이 성공하면 해당 부모 후보의 탐색은 계속됩니다.

```bash
discovery-os fusion-search \
  --search-id run-001 \
  --goal goal.json \
  --parent parent.json \
  --run-config run-config.json \
  --generator mattergen \
  --rounds 4 \
  --frontier-width 6 \
  --control-point 0.20:1.60 \
  --control-point 0.50:1.00 \
  --control-point 0.80:0.60 \
  --max-control-variants 3 \
  --ranking-limit 50
```

최종 `FusionSearchReport.ranked_candidates`에는 후보 하나가 아니라 여러 후보가 순위대로 들어갑니다. 각 행에는 다음 정보가 보존됩니다.

- 후보 구조와 immutable candidate reference
- 생성 라운드, branch, alpha, temperature 및 generation warnings
- Pareto·안정성·목표물성·신규성·전문가 불일치 branch별 순위와 점수
- 전문가별 원본 예측값, 단위, 불확실성
- 통합 우선순위와 산정 근거

통합 우선순위는 물성이 다른 원본 수치를 평균하지 않습니다. branch별 순위에 weighted reciprocal-rank fusion을 적용한 진단용 탐색 우선순위이며, 실제 과학적 판정은 후보별 원본 전문 모델 예측과 후속 고정밀 검증을 사용해야 합니다. MatterGen v1에서는 alpha가 guidance에 적용되지만 temperature는 직접 지원되지 않으므로 해당 사실이 generation warning에 남습니다. REINVENT 계열 생성기에서는 temperature가 실제 sampling에 적용됩니다.

## 모델 연결 규격

모델 구현은 같은 `DiscoveryModel` 계약을 따릅니다. 핵심 단계는 다음과 같습니다.

```python
class DiscoveryModel(Protocol):
    def compile_goal(self, request: GoalCompileRequest) -> DiscoveryGoal: ...
    def propose_hypotheses(self, request: HypothesisRequest) -> HypothesisBatch: ...
    def propose_candidates(self, request: CandidateProposalRequest) -> CandidatePlan: ...
    def predict_candidates(self, request: PredictionRequest) -> PredictionBatch: ...
    def plan_validation(self, request: ValidationPlanningRequest) -> ValidationPlan: ...
    def analyze_results(self, request: ResultAnalysisRequest) -> ResultAnalysis: ...
    def revise_candidates(self, request: RevisionRequest) -> RevisionPlan: ...
    def decide_stop(self, request: StopDecisionRequest) -> StopDecision: ...
```

모든 요청·응답은 `schema_version`이 있는 JSON-safe Pydantic 계약이며 알 수 없는 필드를 거부합니다. 모델의 자연어 설명은 `reason`이나 `rationale`일 뿐 실행 코드로 해석하지 않습니다.

### 로컬 모델

**연결 규격 준비·backend 필요** 단계입니다. 로컬 추론 래퍼는 실행 가능한 연결부를 제공하지만 실제 학습 모델과 tokenizer/backend는 사용자가 주입해야 합니다. 각 메서드에서 요청을 직렬화하고, constrained/structured generation으로 해당 응답 schema의 JSON만 생성한 뒤 `model_validate_json()`을 호출해야 합니다. 파싱 실패, 추가 필드, 비유한 숫자, timeout은 계획 실패로 기록하며 임의 텍스트를 복구해 실행하지 않습니다.

```python
raw = structured_generate(request_json, response_schema=ValidationPlan)
proposal = ValidationPlan.model_validate_json(raw)
compiled = plan_compiler.compile(proposal, goal=goal, candidates=candidates)
```

### HTTP 모델

**연결 규격 준비·backend 필요** 단계입니다. HTTP 클라이언트와 wire schema는 실행 가능하지만 실제 원격 endpoint는 사용자가 제공해야 합니다. 현재 어댑터는 고정 endpoint, timeout, 헤더 검증과 strict schema 재검증을 제공합니다. 운영 배포에서는 프록시나 전송 계층에서 응답 크기, 인증·비밀 관리와 재시도/멱등성 정책도 추가해야 합니다. `/plan-validation`의 응답은 검증되지 않은 제안이며, 곧바로 도구 runtime에 전달하지 않고 반드시 같은 `PlanCompiler`를 통과시킵니다.

로컬과 HTTP 구현을 교체해도 registry, 어댑터, artifact, 증거 정책과 검증 프로필은 바뀌지 않는 것이 이 규격의 핵심입니다.

## 어댑터 추가 원칙

외부 검증기나 생성기를 연결할 때는 다음 순서를 지킵니다.

1. 도구·버전·지원 도메인·후보 타입·operation·생성 속성·evidence kind·fidelity·조건 schema·기본 예산을 `ToolDescriptor`로 선언합니다.
2. operation별 조건을 별도의 엄격한 타입으로 검증합니다. 자유 형식 `conditions`를 셸 인자, URL 또는 파일 경로로 직접 연결하지 않습니다.
3. 애플리케이션 시작 코드에서 어댑터 인스턴스를 정적으로 registry에 등록합니다. 모델이 모듈 이름이나 실행 파일을 선택해 동적 로드하게 하지 않습니다.
4. 짧은 rule validator 외의 사용 가능한 도구는 실행 프로세스/컨테이너를 실제로 종료한 뒤에만 `TimeoutError`를 반환하는 `run_with_timeout()` 계약을 구현해야 registry에 등록됩니다. 생성기도 같은 의미의 `generate_with_timeout()`이 필요합니다. `normalize()`은 성공뿐 아니라 `partial`, `failed`, `timeout`, 경고와 수렴 실패도 `EvidenceRecord`로 보존합니다.
5. 도구 버전, 파라미터, seed, 후보 버전, 입력·출력 해시와 content-addressed immutable artifact 경로를 남깁니다. 실행 성공은 과학적 주장 통과를 뜻하지 않습니다.
6. 정상·비정상 입력, timeout, 손상된 출력, 캐시 재현성, 계산 증거가 실험 증거로 승격되지 않는지를 테스트합니다.

실험 연결은 일반 계산 어댑터와 분리합니다. 초기에는 승인된 protocol 요청 내보내기와 서명·출처가 있는 결과 가져오기만 허용하는 것이 안전합니다. 모델이 로봇·장비를 직접 제어하거나 위험한 합성 절차를 자동 실행하는 기능은 이 MVP의 범위가 아닙니다.

검증된 실험 record를 실제 판정에 사용하려면 같은 신뢰 검사를 importer, store, gate evaluator에 명시적으로 연결합니다. 기본값은 fail-closed이며, 이 연결이 없으면 `success` 실험 record의 저장 또는 게이트 통과가 거부됩니다.

```python
importer = ExperimentalEvidenceImporter(artifact_store, verifier=lab_verifier)
store = JsonDiscoveryStore(
    "runs",
    experimental_record_verifier=importer.verify_record,
)
gate_evaluator = GateEvaluator(
    experimental_record_verifier=importer.verify_record,
)
```

`lab_verifier`는 모델이 아니라 애플리케이션이 소유하며 기관 권한, 서명/attestation, protocol, 시료, 원자료 hash를 검사해야 합니다. Importer는 검증된 record hash와 attachment hash를 content-addressed attestation artifact로 영속화해 재시작 후에도 재검증합니다. 여러 서버·기관을 잇는 운영 환경에서는 이 로컬 commitment에 더해 영속적인 외부 서명/기관 검증기를 사용합니다.

## 증거와 주장 규칙

후보의 증거 수준(`ClaimLevel`)은 다음 경계를 지킵니다.

```text
generated
  → computationally_plausible   계산 증거와 적용 범위·불확실성 존재
  → experimentally_observed     필수 실험 게이트와 대조·반복 통과
  → independently_replicated    독립 시료·팀/시설의 사전 정의 기준 통과
```

별도의 report 상태는 `unvalidated`, `computationally_supported`, `experimentally_validated`, `inconclusive`, `rejected`로 표현합니다. 기본 프로필에서 `experimentally_validated`는 마지막 필수 `independently_replicated` gate까지 통과한 경우에만 부여합니다.

- 계산 방법을 두 번 돌렸다고 독립 실험 재현이 되지 않습니다.
- 데이터베이스에 없다는 결과는 해당 DB·snapshot 안에서의 미검색일 뿐 세계 최초의 증명이 아닙니다.
- `success`는 프로그램 실행 성공 상태이며 속성 주장 합격 상태가 아닙니다.
- 실패, timeout, OOD, 미수렴, 누락 증거는 자동 합격으로 변환하지 않습니다.
- 후보 구조·조성·공정이 바뀌면 새 후보 버전으로 취급하고 과거 증거를 자동 승계하지 않습니다.
- 의약품 승인에는 비임상·임상·규제 심사가 별도로 필요합니다. 이 시스템의 어떤 상태도 의료 사용 승인이 아닙니다.

## 권위 있는 출발점

- 저분자 구조 sanitization과 descriptor의 범위: [The RDKit Book](https://www.rdkit.org/docs/RDKit_Book.html)
- 의약품 발견 이후 비임상·임상·심사 단계: [FDA — The Drug Development Process](https://www.fda.gov/patients/learn-about-drug-and-device-approvals/drug-development-process)
- 경쟁상과 convex hull을 이용한 열역학 안정성 및 계산 한계: [Materials Project — Phase Diagrams](https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/phase-diagrams-pds)
- 격자 동역학 계산 도구의 사용 범위: [Quantum ESPRESSO — PHonon User's Guide](https://www.quantum-espresso.org/Doc/user_guide_PDF/ph_user_guide.pdf)
- 초전도성의 자기장 배제와 vortex/pinning 맥락: [NIST — Flux Lattice in Superconductors and Melting](https://www.nist.gov/ncnr/flux-lattice-superconductors-and-melting)

이 링크들은 검증 설계를 위한 출발점입니다. 특정 후보의 타당성, 안전성, 성능 또는 신규성을 보증하지 않습니다.
