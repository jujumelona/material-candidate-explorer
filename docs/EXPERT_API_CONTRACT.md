# Expert/Fusion/Generator HTTP API contract v1

모든 endpoint와 base URL은 코드 또는 운영 설정 소유입니다. 후보, planner, expert, Fusion 모델의 응답이 임의 endpoint를 지정할 수 없습니다. 요청과 응답은 `schema_version: "1.0"`을 명시한 JSON object이며 추가 필드, NaN/Infinity, 타입 강제 변환, code fence를 거부합니다.

현재 CLI에 공개된 JSON Schema는 다음처럼 확인합니다.

```powershell
discovery-os schema ExpertFeaturePayload
discovery-os schema FusionRequest
discovery-os schema FusionOutput
discovery-os schema FusionRevisionProposal
discovery-os schema FusionWorkspaceSnapshot
```

나머지 정확한 요청·응답 타입은 `src/discovery_os/fusion_schemas.py`의 strict Pydantic schema가 단일 기준입니다.

## 공통 전송 규칙

- 클라이언트는 canonical 요청 SHA-256을 `Idempotency-Key`로 보냅니다.
- redirect는 따르지 않고 거부합니다.
- 응답의 `schema_version`을 명시적으로 검사한 뒤 strict validation을 다시 수행합니다.
- expert와 Fusion 응답은 기본 16 MiB, generator 응답은 기본 32 MiB로 제한합니다.
- 인증이 있는 비-loopback HTTP는 허용하지 않습니다. 운영 서비스는 HTTPS를 사용합니다.
- token은 URL에 넣지 않고 Authorization header로 전달합니다.

## 1. 전문 특징 서비스: `expert-feature-v1`

```text
POST /v1/features
Content-Type: application/json
Accept: application/json
Idempotency-Key: <canonical request sha256>

request:  ExpertFeatureRequest
response: ExpertFeaturePayload
```

요청의 핵심 필드는 `workspace_entity_id`, 전체 `Candidate`, `DiscoveryGoal`, 명시적인 `modality`, `feature_space`, `cycle`, `seed`입니다. candidate에는 현재 내용과 일치하는 `CandidateRef`가 있어야 합니다. route는 코드 소유 `ExpertDescriptor.routes`에서 허용한 candidate type과 representation 조합이어야 합니다.

응답에는 다음 값이 요청과 정확히 일치해야 합니다.

- `candidate_ref`
- `workspace_entity_id`
- `expert_id`와 provenance의 `expert_id`
- `modality`, `feature_space`
- descriptor와 같은 `adapter_version`
- 요청과 같은 provenance `seed`

`status=success`이면 `tensor`와 `semantics`가 모두 필수입니다. `status=failed`이면 tensor와 diagnostic property를 포함할 수 없습니다. `partial` tensor는 runtime에 들어갈 수 있지만 경고로 남습니다.

### 필수 provenance

- `expert_id`, `adapter_version`, `model_version`
- immutable code revision과 weight revision
- 선택적 dataset/projection version
- 입력 parameter hash, device, seed

### FeatureSemantics

tensor의 숫자만 보내서는 안 됩니다. 다음 의미 정보가 함께 와야 합니다.

- `tensor_role`: global/token/atom/cell embedding, Hamiltonian 또는 custom
- `projection_id`: Fusion 입력 projection의 버전형 ID
- 선택적 `entity_type`, `entity_ids`, `mask`
- `pooling`, `normalization`
- 선택적 `coordinate_frame`, `basis`, `unit_semantics`

`entity_ids`가 있으면 tensor의 첫 축 길이와 같아야 하며 mask 길이도 일치해야 합니다. 서로 다른 projection·normalization·basis·단위를 가진 tensor를 같은 공간으로 취급해서는 안 됩니다.

대형 원자·잔기·세포 tensor를 JSON 한도까지 밀어 넣지 마십시오. 필요한 projection 또는 pooling을 sidecar에서 수행하고, 원자료는 검증된 별도 content-addressed 저장소에 보존하는 방식을 권장합니다.

## 2. 사용자 Fusion 서비스: `fusion-v1`

### 특징 결합

```text
POST /v1/fuse
request:  FusionRequest
response: FusionOutput
```

`FusionRequest.workspace`는 하나의 primary candidate와 0개 이상의 target/context/environment/assay 엔터티, 그리고 엔터티 관계를 포함합니다. 각 `FusionFeatureInput`과 payload는 동일한 `workspace_entity_id`를 명시하고, 그 엔터티의 정확한 `candidate_ref`를 가리켜야 합니다. 같은 CandidateRef가 homodimer A/B처럼 여러 역할에 놓여도 feature ID와 역할이 섞이지 않습니다. 실패 feature 또는 tensor가 없는 feature는 요청에 들어갈 수 없습니다.

`previous_latent`와 `previous_state_id`가 있다면 새 후보 cycle의 연속 갱신입니다. backend는 feature semantics와 provenance를 검사하고, 이전 latent를 어떤 방식으로 사용했는지 자체 모델 계약으로 고정해야 합니다.

`FusionOutput`은 입력된 모든 `feature_id`를 `used_feature_ids` 또는 `ignored_feature_ids` 중 정확히 하나에 기록해야 합니다. 알 수 없는 ID를 인용하거나 입력 하나라도 누락하면 전체 응답이 거부됩니다. 출력에는 latent 외에도 `backend_id`, `backend_version`, immutable code/weight revision이 필수입니다.

### 수정 의도

```text
POST /v1/revise
request:  FusionRevisionRequest
response: FusionRevisionProposal
```

응답은 받은 primary `CandidateRef`와 `state_id`를 정확히 인용해야 합니다. `desired_changes`는 다음과 같은 제한된 축과 방향만 기술합니다.

- 원소 분포, 3D 좌표, 격자
- 분자 구조, 단백질/RNA 서열, 세포 상태, 전자구조
- 목표 물성의 increase/decrease/target/preserve/explore

응답에는 generator 선호, confidence, rationale, safety note를 넣을 수 있지만 Python, shell, import path, URL, credential 또는 임의 artifact path를 넣는 필드는 없습니다.

`FUSION_API_URL`이 없으면 동일한 Python protocol을 구현한 로컬 결정론적 `EvidenceDrivenFusionBackend`를 사용하므로 HTTP 호출은 없습니다. URL이 있으면 기존 원격 계약과 선택적 `FUSION_API_TOKEN`을 사용합니다. loopback HTTP 개발은 별도 insecure-local opt-in이 필요합니다.

`FusionRequest.decision_context`, `failed_expert_ids`, `missing_expert_ids`와 `FusionRevisionRequest.decision_context`는 로컬 결정론적 controller를 위한 확장 필드입니다. `RemoteFusionBackend`는 strict legacy 서버와 호환되도록 기본 wire payload에서 이 필드를 제외합니다. 서버가 확장 스키마를 지원한다고 별도로 확인한 경우에만 `FUSION_SEND_EXTENDED_REQUEST_CONTEXT=1`로 전송을 opt-in합니다.

## 3. 후보 생성 서비스: `generator-v1`

```text
POST /v1/generate
request:  FusionGenerationRequest
response: FusionGenerationResponse
```

OFF 요청에는 goal, parent candidate, workspace, `workspace_mode=off`, `WorkspaceRunConfig`만 들어갑니다. latent와 revision을 넣으면 거부됩니다.

ON 요청에는 같은 항목과 함께 `revision_proposal`, `latent_state`, 실제 `latent_payload`가 모두 들어가야 합니다. payload dtype·shape와 content hash는 state metadata와 정확히 일치해야 합니다. run config의 goal hash, parent ref와 mode도 요청에 정확히 맞아야 합니다. 일반 반복/search와 OFF/ON paired benchmark는 모두 `candidate_count` 1~1024를 지원하고 응답 후보 수를 정확히 검사합니다. paired benchmark 응답은 후보마다 `pair_slot`, 전체 batch에 실제로 사용한 `batch_seed`, generator stream 위치를 반환합니다. runner는 두 arm을 `pair_slot`으로 결합하고 slot 누락·중복·재정렬, 서로 다른 runtime parameter hash나 seed를 거부한 뒤 각 후보를 전문가로 다시 평가합니다.

생성 응답의 `Candidate`는 다음 조건을 만족해야 합니다.

- 새 내용과 일치하는 `candidate_ref.content_hash`
- 정확한 부모 ID를 `parent_candidate_ids`에 기록
- 정확한 부모 ID/version/content hash를 `parent_candidate_refs`에 기록
- 부모와 candidate ID가 같으면 더 큰 version 사용

`GeneratorProvenance`의 generator ID/version, code/weight revision, parameter hash와 seed는 `WorkspaceRunConfig`와 정확히 일치해야 합니다. 클라이언트와 `FusionLoopRunner`가 둘 다 이 조건을 재검사합니다.

manifest에 등록된 현재 generator URL은 `MATTERGEN_API_URL`과 `REINVENT_API_URL`이며 token 이름은 각각 `MATTERGEN_API_TOKEN`, `REINVENT_API_TOKEN`입니다.

## 설치 패키지와 sidecar의 경계

통합 manifest와 bootstrap은 upstream package, source archive, 허용된 weight와 공통 sidecar runtime을 엔진별 환경에 준비합니다. `start-sidecars.ps1`/`.sh`는 저장소가 구현한 `/v1/features`·`/v1/generate` 서버를 각 환경에서 기동하고 URL env 파일을 만듭니다. 전처리, lazy checkpoint loading, bounded worker/queue, 요청·응답 strict 검사와 provenance binding은 포함됩니다.

자동으로 제공하지 않는 범위는 사용자가 학습한 `/v1/fuse`·`/v1/revise` backend, 수동/관리 checkpoint 선택, 운영 TLS·secret manager·외부 queue·rate limit·process supervisor, 장시간 작업의 강제 GPU kernel 취소입니다. Boltz 2.2.1은 고정 CLI/YAML/result codec을 제공합니다. scGPT 0.2.4는 명시적 `genes`/`values`/`value_semantics` 입력과 고정 bundle에서 실제 cell embedding을 반환합니다. `raw_counts`는 전체 vector에 `normalize_total=10000 → log1p`를 적용하고, `normalized_log1p`는 이 변환이 이미 끝났음을 뜻합니다. QHNet은 검증된 AIRS source, checkpoint/config bundle과 정확히 일치하는 XYZ/SDF 분자에 한해 실제 full Hamiltonian을 반환하며 overlap이 없으면 orbital energy/coefficients를 만들지 않습니다.

따라서 package 설치 또는 `status=lazy` health만으로 추론 수락을 선언하면 안 됩니다. 모델별 고정 fixture 추론, projection semantics, 실제 checkpoint provenance를 확인한 뒤 URL 환경변수를 중앙 프로세스에 적용해야 합니다. 세부 행렬은 [실제 모델 sidecar 실행](SIDECARS.md)에 있습니다.

비동기 계산기 sidecar를 구현할 경우 다음 운영 endpoint를 둘 수 있습니다.

```text
GET    /healthz
GET    /readyz
GET    /v1/capabilities
POST   /v1/jobs
GET    /v1/jobs/{job_id}
DELETE /v1/jobs/{job_id}
```

이 job API는 현재 `expert-feature-v1`/`generator-v1` 동기 클라이언트가 자동 호출하는 계약이 아닙니다. PySCF는 현재 동기 `expert-feature-v1` sidecar에서 CPU RHF/UHF 계산을 수행하며 장시간 비동기 job API나 DFT를 제공하지 않습니다.

## 보안·증거 규칙

- token은 환경변수 또는 secret manager에서만 읽고 manifest, artifact, freeze, audit log에 쓰지 않습니다.
- model/weight/code/dataset/projection revision과 입력·출력 hash를 보존합니다.
- 서비스는 deadline과 idempotency key를 존중해야 합니다.
- 생물학·의약 후보를 합성 주문, 로봇 또는 실험 장비로 자동 전달하지 않습니다.
- embedding, latent, diagnostic property, OFF/ON 개선량을 계산 증거 또는 실험 증거로 승격하지 않습니다.
- OOD, partial, failed 또는 누락 expert를 성공으로 바꾸지 않습니다.
