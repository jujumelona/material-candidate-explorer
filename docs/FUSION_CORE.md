# Unified Scientific Fusion Core

이 계층은 MatterGen·Uni-Mol·Boltz·ESM·RNA-FM·scGPT·QHNet·PySCF·UMA·MatterSim·CHGNet·Chemprop 같은 전문 모델의 가중치를 하나로 합치지 않습니다. 각 모델은 호환되는 독립 Python/CUDA 환경 또는 독립 서비스에 남습니다. 코드는 각 서비스의 출력을 엄격한 특징 계약으로 정규화하고, 기본 로컬 Evidence controller 또는 선택적인 원격 Fusion AI에 전달합니다.

```text
DiscoveryGoal
  + ScientificWorkspace
      ├─ primary_candidate: 이번 단계에서 수정할 후보
      ├─ target/context/environment/assay 엔터티
      └─ 엔터티 사이의 typed relation
            ↓ 코드 소유 ExpertRoute
  각 전문 서비스의 ExpertFeaturePayload
      ├─ NumericTensor
      ├─ FeatureSemantics
      ├─ DiagnosticProperty
      └─ model/code/weight/projection provenance
            ↓
  로컬 Evidence controller 또는 원격 FusionBackend
      ├─ fuse 또는 /v1/fuse   → UnifiedLatentStateRef
      └─ revise 또는 /v1/revise → FusionRevisionProposal
            ↓
  허용된 generator-v1 sidecar
      └─ /v1/generate → 부모를 정확히 인용한 새 Candidate
            ↓
  모든 workspace 엔터티의 전문 특징 재추출
            ↓
  이전 latent + 새 특징 → 다음 latent
            ↓
  별도의 계산·실험 EvidenceRecord 검증 경로
```

## 바뀌지 않는 경계

- 전문 모델 embedding, 공통 latent, 정렬 점수, 진단 물성, OFF/ON 차이는 `EvidenceRecord`가 아닙니다.
- Fusion AI가 반환할 수 있는 것은 제한된 `FusionRevisionProposal`뿐입니다. Python·shell·URL·credential·임의 파일 경로를 실행하는 필드는 계약에 없습니다.
- 생성된 후보는 내용 해시가 맞는 불변 `CandidateRef`를 가져야 하며, `parent_candidate_ids`와 `parent_candidate_refs`에 정확한 부모를 함께 기록해야 합니다.
- 새 후보는 같은 ID를 유지한다면 version이 증가해야 합니다. 부모 후보의 Evidence는 자동 승계되지 않습니다.
- tensor와 보고서는 `ArtifactStore`에 content-addressed artifact로 보존됩니다. 참조의 SHA-256과 실제 bytes가 다르면 읽기를 거부합니다.
- 실제 효능, 독성, 안정성, 초전도성, 합성 가능성 또는 신규성 결론은 기존 validation profile과 독립된 계산·실험 증거 게이트를 통과해야 합니다.

## 다중 엔터티 ScientificWorkspace

하나의 분자만 latent로 만드는 대신, 한 작업에 함께 필요한 과학적 대상을 명시적으로 묶습니다. 예를 들어 약물 설계 workspace는 다음처럼 구성할 수 있습니다.

```text
primary   small_molecule  role=primary_candidate
target    protein        role=target
assay     cell_state     role=assay

primary --binds_to-----> target
primary --evaluated_in-> assay
```

`WorkspaceEntityInput`은 실제 `Candidate`를 runtime에 제공하고, wire에 남는 `ScientificWorkspace`는 각 엔터티의 불변 `CandidateRef`만 보존합니다. `WorkspaceRelation`의 양 끝은 반드시 같은 workspace 안에 있어야 하며 primary는 정확히 하나여야 합니다.

runtime은 primary뿐 아니라 context 엔터티도 각각 호환되는 expert로 보냅니다. 따라서 분자 특징, 표적 단백질 특징, 세포 상태 특징을 같은 `FusionRequest.features`에 넣을 수 있습니다. 다만 서로 다른 엔터티의 tensor를 의미 없이 이어 붙이지 않도록 Fusion AI가 `candidate_ref`, modality, feature space와 아래 semantics를 함께 사용해야 합니다.

latent를 다음 cycle로 넘길 때는 같은 goal hash, seed, workspace ID와 context/relations가 유지되어야 합니다. 새 primary는 직전 state의 후보와 같거나, `parent_candidate_refs`로 그 후보를 정확히 인용한 자식이어야 합니다. cycle은 정확히 1 증가합니다. backend ID/version/code/weight revision과 latent dtype/shape도 기본적으로 고정되며, 명시적으로 runtime 호환성 완화를 켜지 않는 한 변경이 거부됩니다.

## 특징의 과학적 의미

`ExpertFeaturePayload`의 성공 tensor에는 `FeatureSemantics`가 필수입니다.

각 요청·payload·저장된 feature ref에는 `workspace_entity_id`도 필수입니다. 따라서 같은 `CandidateRef`를 서로 다른 역할이나 복합체 인스턴스로 재사용해도 어느 엔터티에서 나온 특징인지 보존됩니다.

| 필드 | 의미 |
|---|---|
| `tensor_role` | global/token/atom/cell embedding, Hamiltonian 또는 custom |
| `projection_id` | Fusion 입력 공간으로 보내는 projection의 불변 ID |
| `entity_type`, `entity_ids`, `mask` | 첫 tensor 축이 나타내는 원자·잔기·세포와 유효 항목 |
| `pooling` | none/mean/sum/cls/attention/custom |
| `normalization` | scaling, centering, whitening 등 적용 규칙 |
| `coordinate_frame` | 좌표계·주기 경계·정렬 기준 |
| `basis` | 전자구조 basis, orbital convention 등 |
| `unit_semantics` | 값 또는 축의 물리 단위 의미 |

`entity_ids`가 있으면 tensor 첫 축 길이와 정확히 같아야 합니다. projection ID, normalization, basis 또는 단위가 다른 특징은 차원이 같아도 같은 공간이라고 가정할 수 없습니다. 이 스키마는 의미를 전달할 뿐 자동으로 과학적 호환성을 증명하지 않으므로, Fusion 모델 학습·평가 시 허용 조합을 고정해야 합니다.

`ExpertDescriptor.routes`는 `(modality, feature_space, representation kind, candidate type)` 조합을 명시합니다. runtime은 descriptor 목록의 첫 항목을 임의로 고르지 않고 일치하는 route만 호출합니다. 현재 환경 registry는 다음 계열을 연결 대상으로 선언합니다.

- 결정·재료·원자계: UMA, MatterSim, CHGNet
- 분자 2D/3D: Uni-Mol, Chemprop, Boltz의 ligand route
- 단백질: ESM, Boltz
- RNA·세포: RNA-FM, scGPT
- 전자구조: QHNet, PySCF

URL이 설정되지 않은 expert는 `available=false`로 남으며 실행 가능한 서비스로 간주되지 않습니다.

| expert | base URL 환경변수 | bearer token 환경변수 | feature space |
|---|---|---|---|
| Uni-Mol | `UNIMOL_API_URL` | `UNIMOL_API_TOKEN` | `unimol-cls-v1` |
| Boltz | `BOLTZ_API_URL` | `BOLTZ_API_TOKEN` | `boltz-structure-v1` |
| ESM | `ESM_API_URL` | `ESM_API_TOKEN` | `esm-sequence-v1` |
| RNA-FM | `RNAFM_API_URL` | `RNAFM_API_TOKEN` | `rnafm-t12-v1` |
| scGPT | `SCGPT_API_URL` | `SCGPT_API_TOKEN` | `scgpt-cell-v1` |
| QHNet source | `QHNET_API_URL` | `QHNET_API_TOKEN` | `qhnet-hamiltonian-v1` |
| UMA | `UMA_API_URL` | `UMA_API_TOKEN` | `uma-atomic-v1` |
| MatterSim | `MATTERSIM_API_URL` | `MATTERSIM_API_TOKEN` | `mattersim-atomic-v1` |
| CHGNet | `CHGNET_API_URL` | `CHGNET_API_TOKEN` | `chgnet-atomic-v1` |
| Chemprop | `CHEMPROP_API_URL` | `CHEMPROP_API_TOKEN` | `chemprop-mpn-v1` |
| PySCF | `PYSCF_API_URL` | `PYSCF_API_TOKEN` | `pyscf-orbital-v1` |

token 이름은 각 URL 환경변수의 `_API_URL`을 `_API_TOKEN`으로 바꾼 이름입니다. manifest component ID가 `qhnet-source`여도 QHNet token은 `QHNET_API_TOKEN`입니다.

## Fusion backend 연결

Fusion AI는 두 메서드를 구현합니다.

```python
class FusionBackend(Protocol):
    def fuse(self, request: FusionRequest) -> FusionOutput: ...
    def propose_revision(
        self,
        request: FusionRevisionRequest,
    ) -> FusionRevisionProposal: ...
```

`FUSION_API_URL`이 없거나 공백이면 `build_fusion_backend_from_environment()`는 로컬 `EvidenceDrivenFusionBackend`를 자동 선택합니다. 이 backend는 서로 다른 전문가 embedding을 평균하지 않고 평가 성공·실패 수, 목표 utility, 전문가 불일치, 개선 여부, 붕괴 비율과 현재 alpha만 결정론적 탐색 상태에 기록합니다. 학습 가중치나 원격 서비스는 필요하지 않습니다.

로컬 latent는 과학 예측 벡터가 아니라 다음 순서가 고정된 8개 탐색 제어 값입니다.

```text
[cycle, 성공 expert 수, 비성공 expert 수, 최악 objective utility,
 expert 간 불일치, 이전 라운드 개선 여부, 구조 붕괴 비율, guidance alpha]
```

주 후보의 `SUCCESS` property만 utility·불일치·revision에 사용합니다. 비성공 축에는 실패·누락·`PARTIAL` expert를 포함하되, 누락을 실제 실행 실패라고 기록하지는 않습니다. 이 때문에 불완전한 evaluator panel은 revision confidence도 낮아집니다. context 후보는 보간하거나 성공으로 간주하지 않습니다. 동일 expert의 여러 route가 서로 다른 값을 냈을 때도 그것을 expert 간 불일치로 부풀리지 않습니다. tensor 값은 읽거나 평균내지 않습니다.

`propose_revision()`이 낼 수 있는 조건 이름은 `chemical_system`, `space_group`, `dft_mag_density`, `dft_band_gap`, `ml_bulk_modulus`, `hhi_score`, `energy_above_hull`뿐입니다. 명시적인 goal target을 우선하고, 근거가 없거나 단위·형식이 맞지 않으면 숫자를 만들어내지 않습니다. `chemical_system`은 검증된 원소 기호를 정렬하고 `space_group`은 1~230 정수만 허용합니다. hull 값은 `eV/atom`으로 정규화하며 `meV/atom`은 명시적으로 변환하고, 단위가 없는 혼합 panel은 닫습니다. 안정 후보가 부족하면 hull 목표를 `0.00`으로 좁히고, 안정 후보가 확보된 뒤에는 분기에 따라 `0.00`, `0.03`, `0.05`, `0.08`을 사용합니다. 불일치 분기의 제안은 낮은 confidence로 유지해 추가 평가 대상으로 보존합니다. hull 이외 조건은 명시 target 또는 명시 range 중간값만 쓰며 전문가 극값을 새 생성 목표로 복사하지 않습니다.

MatterSim·UMA의 `energy_per_atom`은 총 potential energy를 원자 수로 나눈 `eV/atom` 값이며 CHGNet의 같은 이름 property와 함께 전문가별 축으로 보존할 수 있습니다. 그러나 raw MLIP energy의 절대 기준은 모델과 조성에 의존하므로 Pareto dominance와 불일치는 같은 reduced composition 안에서만 비교합니다. 이것은 `energy_above_hull`이 아닙니다. convex-hull 안정성에는 구조 이완, 일관된 기준 상 에너지와 조성별 hull 계산을 수행하는 별도 고정밀 connector가 필요하며, 이 계층은 `energy_per_atom`에서 hull 값을 추정해 만들지 않습니다.

원격 Fusion AI를 사용할 때만 운영 endpoint와 선택적 token을 환경변수로 연결합니다.

```powershell
$env:FUSION_API_URL = "https://fusion.internal.example"
$env:FUSION_API_TOKEN = "<secret>"
```

```python
from discovery_os.artifacts import ArtifactStore
from discovery_os.configured_experts import build_expert_registry_from_environment
from discovery_os.configured_fusion import build_fusion_backend_from_environment
from discovery_os.fusion_runtime import FusionRuntime

registry = build_expert_registry_from_environment()
backend = build_fusion_backend_from_environment()
runtime = FusionRuntime(
    registry,
    backend,
    ArtifactStore("runs/fusion-artifacts"),
)
```

원격 주소는 HTTPS가 기본입니다. loopback HTTP 개발은 `FUSION_ALLOW_INSECURE_LOCAL_HTTP=1` 또는 `DISCOVERY_ALLOW_INSECURE_LOCAL_HTTP=1`을 명시해야 합니다. URL이 있을 때의 기존 bearer token 및 provenance 검사는 그대로 유지하며, 토큰은 manifest, artifact 또는 설치 상태에 쓰지 않습니다.

로컬 controller용 `decision_context`, `failed_expert_ids`, `missing_expert_ids`는 기존 strict `fusion-v1` 서버의 요청 스키마를 깨지 않도록 원격 POST에서 기본 제외됩니다. 확장 요청을 지원한다고 확인된 서버에만 `FUSION_SEND_EXTENDED_REQUEST_CONTEXT=1`을 명시합니다. 이 opt-in은 endpoint·token·응답 provenance 검사를 완화하지 않습니다.

`MeanFusionBackend`는 테스트 fixture에만 남은 배선 확인용 구현이며 운영 fallback이 아닙니다. 로컬 `EvidenceDrivenFusionBackend`의 latent도 과학 예측값이나 후보 점수가 아닙니다. 실제 탐색 선택은 `DeterministicExplorationSelector`와 `AdaptiveGenerationScheduler`가 `ExpertEvidenceStore`에 보존된 전문가별 원본 property vector를 사용해 수행합니다.

## revision → generation → 재추출 폐루프

`FusionLoopRunner.iterate()`가 한 번의 폐루프를 실행합니다.

1. 현재 primary와 모든 context에서 전문 특징을 추출합니다.
2. 선택된 backend의 `fuse()`로 탐색 상태 latent를 만들고 `propose_revision()`으로 수정 의도를 받습니다. 원격 backend일 때만 `/v1/fuse`와 `/v1/revise`가 호출됩니다.
3. `FusionCandidateGenerator.generate()`에 goal, workspace, run config, latent와 수정 의도를 전달합니다.
4. 생성기가 부모를 정확히 인용한 `candidate_count`개의 새 후보와 generator provenance를 반환합니다.
5. 모든 새 후보와 같은 context의 특징을 다시 추출합니다. 동일 후보·expert·목표의 불변 평가 입력만 goal hash가 포함된 persistent cache에서 검증 후 재사용합니다.
6. 후보별로 직전 latent를 입력해 다음 latent를 만들고 `FusionBatchIterationReport`를 저장합니다.

manifest에 등록된 생성기는 환경변수로 고정 연결합니다.

```powershell
$env:MATTERGEN_API_URL = "https://mattergen.internal.example"
$env:MATTERGEN_API_TOKEN = "<secret>"
# 또는 REINVENT_API_URL / REINVENT_API_TOKEN
```

```python
from discovery_os.configured_fusion import build_generator_from_environment
from discovery_os.fusion_loop import FusionLoopRunner

generator = build_generator_from_environment("mattergen")
loop = FusionLoopRunner(runtime, generator)
iteration = loop.iterate(
    goal=goal,
    parent_candidate=parent_candidate,
    cycle=0,
    run_config=on_run_config,
    context_entities=[target_entity],
    relations=[binds_to_relation],
)
```

이 저장소는 MatterGen/REINVENT용 `generator-v1` sidecar와 여러 실제 expert sidecar를 제공합니다. checkpoint·decoder·후처리 revision/hash는 여전히 `WorkspaceRunConfig`, 기동 환경과 응답 provenance에 정확히 일치해야 합니다. 모델별 지원 범위와 실패 닫힘 항목은 [sidecar 실행 문서](SIDECARS.md)를 따릅니다.

## 증거 보존형 다중 가지 탐색

`FusionSearchRunner`는 후보 하나만 직렬로 잇지 않고 다음의 분기별 bounded frontier와 누적 elite pool을 독립적으로 유지합니다. 새 자식과 과거 elite를 함께 다시 선택하고, 한 분기 실행이 실패하면 그 분기의 마지막 안전 frontier를 보존해 다음 round에서 재시도합니다. 동일 후보가 여러 분기에 있어도 과학적 identity만 중복 제거하며 continuation latent는 분기별로 유지합니다.

- 안정성
- 목표 물성 maximin
- 전문가 property 공간 신규성
- 전문가 불일치
- 비지배 Pareto

선택기는 전문가 출력을 평균내지 않습니다. 각 `ExpertFeaturePayload`와 원래 `ExpertFeatureRef`를 content-addressed `ExpertEvidenceStore`에 저장하고, failed/partial/OOD/누락 evaluator/단위 불일치 후보를 보간하지 않고 제외합니다. 후보 JSON과 CIF·SMILES·서열 원문, 생성 provenance, 반복 history, latent state, 매 단계 생성 control도 함께 저장합니다.

`AdaptiveGenerationScheduler`는 raw expert objective utility의 반복 간 개선량, 구조 붕괴 비율과 불일치 후보를 관찰합니다. 일반적인 backend에서는 기존 alpha 의미를 유지하지만, MatterGen에서는 `gamma = alpha * guidance_max`인 classifier-free guidance 의미를 사용합니다. 따라서 MatterGen은 개선이 지속되면 조건 집중을 위해 alpha를 높이고, 정체되면 탐색 폭을 넓히기 위해 alpha를 낮춥니다. 붕괴가 증가하면 guidance·temperature·mutation을 낮춥니다. MatterGen v1이 실제로 적용하지 않는 temperature·mutation·diversity는 requested-but-ignored provenance로 남으며, sidecar가 적용한 척하지 않습니다.

최종 보고서의 `validation_handoff_candidate_refs`는 Pareto 후보를 우선하고 안정성 후보를 이어 붙인 후 정확히 같은 과학 표현을 중복 제거한 고비용 검증 입력 목록입니다. 이 필드는 DFT·구조 이완·phonon·실험을 실행했다는 뜻이 아닙니다. 결정 후보 파이프라인은 별도의 `discovery_os.crystal_identity` 단계에서 pymatgen `StructureMatcher`와 표준화를 실행하며, hard dedup은 species-preserving·unscaled strict match에만 허용합니다. volume scaling이 필요한 동일 prototype 관계와 애매한 비교는 삭제하지 않습니다.

## Workspace OFF/ON 비교

`WorkspaceBenchmarkRunner.run_pair()`는 같은 부모에서 두 arm의 후보 배치를 생성하고 새 후보를 전부 다시 평가합니다. 생성기가 반환한 명시적 `pair_slot`으로 OFF/ON 쌍을 결합하며 slot 누락·중복·재정렬, 서로 다른 batch seed나 runtime parameter hash를 거부합니다. `candidate_count=1`일 때는 기존 단수 snapshot/comparison 접근자도 사용할 수 있습니다.

- OFF: expert 진단은 추출할 수 있지만 Fusion latent나 revision을 생성기에 주지 않습니다.
- ON: 같은 입력 조건에서 Fusion latent와 revision을 생성기에 줍니다.

두 `WorkspaceRunConfig`는 mode를 제외하고 goal hash, parent, pair key, seed, cohort, generator ID/version/code/weight, generator parameter hash, decoder·후처리·자원 예산·evaluator panel hash, metric version과 후보 수가 모두 같아야 합니다. runner는 다르면 생성 전에 거부합니다.

snapshot 비교는 context/relations, 누락·실패 expert 목록, adapter/model/code/weight/projection provenance와 feature status도 대조합니다. 하나라도 다르면 `paired_configuration=false`가 되어 Workspace 효과로 해석할 수 없습니다. 물성 단위가 다르거나 어느 arm이라도 OOD이면 해당 `ObjectiveDelta.comparable=false`입니다.

비교기는 `ArtifactStore`에서 모든 feature와 latent를 SHA-256으로 다시 읽고, feature payload로 aggregate property를 재계산합니다. artifact root를 주지 않으면 결과에 미검증 caveat를 남기고 `paired_configuration=false`로 닫습니다. CLI에서도 `--artifact-root`가 필수입니다.

```powershell
discovery-os fusion-pair --goal goal.json --parent parent.json --off-config off.json --on-config on.json --generator mattergen --artifacts runs/fusion
discovery-os fusion-compare --goal goal.json --off-snapshot off.json --on-snapshot on.json --artifact-root runs/fusion
```

진단치는 다음을 포함합니다.

- 원소 분포 total variation과 Jensen–Shannon divergence
- 원자 label과 순서가 정확히 같을 때만 계산하는 좌표 RMS displacement
- 3×3 격자 Frobenius 거리
- SMILES/SELFIES/InChI 값의 변화 여부
- 단백질·RNA normalized edit distance
- 목표 방향을 반영한 물성 signed improvement와 양쪽 uncertainty

좌표 진단은 회전·병진·원자 permutation·결정 대칭을 정렬하지 않습니다. 실제 구조 비교는 RDKit atom mapping, pymatgen `StructureMatcher`, TM-score/lDDT 같은 분야별 검증 도구를 별도로 실행해야 합니다. 모든 비교 보고서의 `scientific_claim`은 고정값 `diagnostic_only`입니다.
