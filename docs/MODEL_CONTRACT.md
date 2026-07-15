# 실제 발견 모델 연결 계약

이 문서는 학습 중인 모델을 Discovery OS에 연결할 때 지켜야 할 wire contract입니다. 모델은 과학적 판단을 구조화해 반환하지만 파일, 셸, GPU, 외부 계산기 또는 실험 장비를 직접 제어하지 않습니다.

현재 연결 경계는 서로 다른 세 역할로 나뉩니다.

| 역할 | 계약 | 책임 |
|---|---|---|
| 탐색 planner | `DiscoveryModel` 8개 메서드 | 목표·가설·후보 생성 작업·예측·검증 계획·결과 해석·중지 의견 |
| 사용자 Fusion AI | `FusionBackend.fuse/propose_revision` | multi-entity 전문 특징을 latent로 결합하고 제한된 수정 의도 제안 |
| 전문 생성기 | `FusionCandidateGenerator.generate` | 수정 의도를 실제 새 `Candidate`로 decode하고 부모·버전·provenance 기록 |

이 셋은 같은 모델일 필요가 없으며 가중치를 merge할 필요도 없습니다. 각각 독립 Python 객체 또는 고정 HTTP endpoint로 배포할 수 있습니다.

## 8개 메서드

| operation | 요청 | 응답 | HTTP endpoint |
|---|---|---|---|
| `compile_goal` | `GoalCompileRequest` | `DiscoveryGoal` | `POST /compile-goal` |
| `propose_hypotheses` | `HypothesisRequest` | `HypothesisBatch` | `POST /propose-hypotheses` |
| `propose_candidates` | `CandidateProposalRequest` | `CandidatePlan` | `POST /propose-candidates` |
| `predict_candidates` | `PredictionRequest` | `PredictionBatch` | `POST /predict-candidates` |
| `plan_validation` | `ValidationPlanningRequest` | `ValidationPlan` | `POST /plan-validation` |
| `analyze_results` | `ResultAnalysisRequest` | `ResultAnalysis` | `POST /analyze-results` |
| `revise_candidates` | `RevisionRequest` | `RevisionPlan` | `POST /revise-candidates` |
| `decide_stop` | `StopDecisionRequest` | `StopDecision` | `POST /decide-stop` |

모든 요청·응답은 JSON object이며 `schema_version: "1.0"`을 포함합니다. 알 수 없는 필드, `NaN`/`Infinity`, 잘못된 enum, 문자열로 위장한 숫자 등은 strict validation에서 거부됩니다.

현재 JSON Schema는 코드에서 직접 생성합니다.

```powershell
discovery-os schema ValidationPlan
```

다른 계약은 `src/discovery_os/schemas.py`의 각 Pydantic 모델에서 `model_json_schema()`로 얻을 수 있습니다.

## 로컬 backend

권장 backend signature는 다음과 같습니다.

```python
class MyStructuredBackend:
    def generate_structured(
        self,
        *,
        operation: str,
        request_json: str,
        response_schema: type,
    ) -> str:
        # constrained decoding으로 response_schema에 맞는 JSON object 생성
        return generated_json
```

연결은 한 줄입니다.

```python
from discovery_os import LocalDiscoveryModel

model = LocalDiscoveryModel(MyStructuredBackend())
```

backend가 반환한 문자열은 실행되거나 `eval()`되지 않고 해당 operation의 정확한 응답 타입으로 다시 검증됩니다. Markdown code fence나 JSON 앞뒤의 자연어도 허용하지 않습니다.

## HTTP backend

각 endpoint는 요청 모델을 JSON body로 받고 표의 응답 모델 하나를 JSON body로 반환해야 합니다.

```python
from discovery_os import RemoteDiscoveryModel

model = RemoteDiscoveryModel(
    "https://model.internal.example/v1",
    timeout=(10, 300),
    auth_headers={"Authorization": "Bearer ..."},
)
```

endpoint URL은 실행 설정에서 고정됩니다. 모델 응답이 URL을 바꿀 수 없습니다. 인증키는 모델 prompt나 Evidence에 넣지 않습니다. 운영 환경에서는 TLS, 응답 크기 제한, 멱등성 키, rate limit 및 비밀 저장소를 별도로 구성해야 합니다.

## 검증 계획의 권장 출력

가능하면 모델은 구체적인 실행 명령보다 `ValidationIntent`를 반환합니다.

```json
{
  "schema_version": "1.0",
  "intents": [
    {
      "schema_version": "1.0",
      "intent_id": "screen-validity",
      "candidate_refs": [
        {
          "schema_version": "1.0",
          "candidate_id": "MOL-101",
          "version": 1,
          "content_hash": "..."
        }
      ],
      "requested_properties": ["validity"],
      "required_evidence_kind": "computational",
      "minimum_fidelity": "cheap",
      "preferred_method_classes": ["rule_based"],
      "conditions": {},
      "priority": 1.0,
      "reason": "고비용 계산 전에 표현 유효성을 확인한다.",
      "max_runtime_seconds": 120,
      "resource_budget": {
        "schema_version": "1.0",
        "cpu_cores": 1,
        "gpu_count": 0,
        "memory_gb": 1,
        "storage_gb": 0,
        "estimated_cost": 0,
        "extras": {}
      }
    }
  ],
  "calls": [],
  "expected_information_gain": {"screen-validity": 0.8},
  "plan_reason": "가장 저렴한 필수 검증부터 수행한다."
}
```

하위 호환을 위해 `ToolCall`도 받을 수 있지만 실행 권한은 아닙니다. `PlanCompiler`가 다음을 모두 다시 확인한 뒤에만 실행 계획이 됩니다.

- 현재 registry에서 `available=true`인 도구와 operation인지
- 후보 ID뿐 아니라 `CandidateRef.version/content_hash`가 맞는지
- 도메인·후보 타입·evidence kind·fidelity가 호환되는지
- 요청 property와 condition이 operation descriptor에 선언됐는지
- CPU/GPU/memory/time/cost 정책을 넘지 않는지
- 사람 승인 또는 실험 작업이 아닌지
- 의존성 graph에 누락이나 cycle이 없는지
- 공통 표현, RDKit, 조성식 등 필수 sanity gate가 빠지지 않았는지

긴 계산 어댑터는 Python thread 취소에 의존하지 않습니다. 실제 worker process/container의 종료를 확인하는 `run_with_timeout()` 또는 `generate_with_timeout()`을 구현해야 `available=true`로 등록할 수 있으며, timeout이 난 작업은 종료 확인 후에도 자동 재시도하지 않습니다.

## 후보 생성 계획

모델은 분자·결정 생성 코드를 반환하지 않고 `CandidatePlan.tasks`만 반환합니다. `generator_name`은 요청의 `available_generators`에 있는 이름 중 하나여야 합니다. 현재 demo의 `dummy_generator`는 알려진 fixture만 생성합니다. 실제 모델이 완성되면 `internal_core` 어댑터를 등록하거나 GenMol/MatterGen 등의 고정 어댑터를 연결해도 엔진 계약은 바뀌지 않습니다.

수정된 후보는 반드시 새 불변 `CandidateRef`와 새 content hash를 가져야 합니다. 같은 `candidate_id`를 재사용하면 version이 증가해야 합니다. `parent_candidate_ids`에는 부모 ID를, `parent_candidate_refs`에는 부모의 정확한 ID/version/content hash를 기록해야 합니다. 부모 후보의 Evidence는 자동 승계되지 않습니다.

## Fusion AI와 전문 생성기 계약

Fusion 계층은 planner의 8개 메서드와 별도입니다. `ScientificWorkspace` 하나에 수정할 primary candidate와 target/context/environment/assay 엔터티 및 관계를 넣습니다. runtime은 각 엔터티를 `ExpertDescriptor.routes`에 따라 전문 서비스로 보내고 `ExpertFeaturePayload`를 수집합니다.

성공한 tensor에는 다음 의미 정보가 있는 `FeatureSemantics`가 필수입니다.

- tensor role과 projection ID
- 선택적인 entity IDs/mask와 entity type
- pooling·normalization
- 좌표계, basis와 단위 의미

Fusion AI는 숫자 shape만 보고 특징을 섞지 말고 이 semantics와 model/code/weight/projection provenance를 함께 검사해야 합니다.

```python
class FusionBackend(Protocol):
    def fuse(self, request: FusionRequest) -> FusionOutput: ...
    def propose_revision(
        self,
        request: FusionRevisionRequest,
    ) -> FusionRevisionProposal: ...

class FusionCandidateGenerator(Protocol):
    def generate(
        self,
        request: FusionGenerationRequest,
    ) -> FusionGenerationResponse: ...
```

HTTP 배포 시 endpoint는 다음과 같습니다.

```text
POST /v1/fuse      FusionRequest           → FusionOutput
POST /v1/revise    FusionRevisionRequest   → FusionRevisionProposal
POST /v1/generate  FusionGenerationRequest → FusionGenerationResponse
```

`FUSION_API_URL`이 없으면 학습 가중치 없는 로컬 `EvidenceDrivenFusionBackend`를 사용합니다. URL이 있으면 원격 Fusion 서비스를 `FUSION_API_URL`/`FUSION_API_TOKEN`으로 연결합니다. generator는 integration manifest에 allowlist된 `MATTERGEN_API_URL` 또는 `REINVENT_API_URL`과 대응 `*_API_TOKEN`으로만 연결합니다. endpoint URL을 모델 응답에서 받지 않습니다.

로컬 탐색 제어용 decision context와 failed/missing expert 목록은 legacy 원격 요청에 기본 전송하지 않습니다. 확장 `fusion-v1` 요청을 실제로 지원하는 서버만 `FUSION_SEND_EXTENDED_REQUEST_CONTEXT=1`로 명시적으로 opt-in합니다.

`FusionLoopRunner`는 revision → generator → 새 후보의 전문 특징 재추출 → 이전 latent를 사용한 다음 latent 갱신을 수행합니다. 연속 cycle에는 같은 goal hash, seed, workspace context와 backend revision을 요구하고, 새 primary가 직전 후보를 정확히 부모로 인용하는지 검사합니다.

OFF/ON 실험은 `WorkspaceBenchmarkRunner`를 사용합니다. mode 이외의 목표·부모·seed·generator/decoder/후처리·예산·evaluator 설정이 다르면 paired run으로 실행하지 않습니다. expert 누락/실패 또는 projection/model provenance가 달라지거나 물성이 OOD이면 비교 가능 판정도 닫힙니다. 이 결과는 항상 `diagnostic_only`이며 Evidence가 아닙니다.

정확한 wire 필드와 sidecar 책임은 [Expert/Fusion/Generator API 계약](EXPERT_API_CONTRACT.md), 실행 구조는 [Fusion Core](FUSION_CORE.md)를 따릅니다.

## 예측과 Evidence의 경계

- `predict_candidates()` 출력은 `model_prior`이며 Evidence가 아닙니다.
- tool 실행 성공(`status=success`)과 속성 기준 통과(`meets_criterion=true`)는 서로 다릅니다.
- 계산 결과는 `evidence_kind=computational`; 실험 결과로 바꿀 수 없습니다.
- 실험 결과는 모델이나 ToolRuntime이 만들지 않습니다. 사람이 `ExperimentalEvidenceImporter`에 sample, protocol, laboratory, source와 원자료 attachment를 명시해 제출하고, 애플리케이션이 구성한 `ExperimentalEvidenceVerifier`가 서명·기관 권한·attachment hash를 확인해야 합니다. verifier가 없는 제출은 보존되지만 `partial/unverified`라서 게이트에 반영되지 않습니다.
- 검증된 제출도 해당 importer/영속 attestation authority의 `verify_record`를 `JsonDiscoveryStore`와 `GateEvaluator` 양쪽에 주입해야 합니다. 기본 store/evaluator는 외부에서 직접 만든 `verified` 문자열을 신뢰하지 않습니다.
- 서로 다른 두 record만으로 독립 재현이 되지 않습니다. profile이 요구하면 서로 다른 명시적 `source_id`가 필요합니다.
- 모델의 `confirmed_findings`와 `finish`는 의견입니다. 최종 상태는 코드 소유 validation profile과 gate evaluator가 결정합니다.

## 연결 전 계약 시험

실제 모델 backend를 연결할 때 최소한 다음을 자동 시험합니다.

1. 8개 endpoint가 각 JSON Schema를 정확히 반환한다.
2. extra field, 잘못된 enum, `NaN`, code fence가 거부된다.
3. 존재하지 않는 tool/operation과 임의 condition이 컴파일되지 않는다.
4. 후보 수정 후 이전 content hash의 Evidence가 거부된다.
5. 계산 결과만으로 `experimentally_validated`가 되지 않는다.
6. timeout·실패·부분 결과가 prompt에 손실 없이 다시 전달된다.
7. checkpoint에서 재시작해도 후보·도구·입출력 hash가 유지된다.

Fusion/전문 서비스와 생성기를 연결한다면 다음 시험도 추가합니다.

8. 성공 특징에 tensor semantics와 immutable model/code/weight/projection provenance가 있다.
9. multi-entity feature가 올바른 workspace entity를 인용하며 잘못된 route가 거부된다.
10. 생성 후보가 정확한 부모 ref를 인용하고 같은 ID라면 version이 증가한다.
11. revision 후 새 후보의 특징이 실제로 재추출되고 이전 latent와 lineage가 이어진다.
12. OFF 요청에는 latent/revision이 전달되지 않고 ON에만 전달된다.
13. OFF/ON의 generator·expert provenance, context, 누락/실패 패널 또는 OOD가 다르면 개선 결론을 내리지 않는다.

저장소의 mock 통합 시험은 이 경계를 실제 RDKit 및 조성식 어댑터까지 포함해 검증합니다.
