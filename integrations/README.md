# Integration manifest

[`manifest.v1.json`](manifest.v1.json)은 실행 코드가 아니라 strict 설치 데이터입니다. 구성요소별로 다음 항목을 기록합니다.

- Python minor, exact top-level package version과 제약
- exact source commit, archive URL·byte size·SHA-256
- exact Hugging Face weight revision 또는 manual/managed 상태
- 라이선스 URL, 명시적 acceptance ID/env와 token env
- 지원 platform/accelerator와 CPU/GPU/storage hint
- `expert-feature-v1`, `generator-v1` API protocol과 base URL 환경변수
- 설치 profile과 component dependency DAG

manifest에는 shell/Python command, credential 또는 실행할 임의 파일 경로가 없습니다. identifier와 destination은 bootstrap이 다시 검증합니다.

## 검증과 조회

```powershell
python scripts/bootstrap.py verify-manifest
python scripts/bootstrap.py plan --profile all-open --accelerator cuda --accept-license esm
discovery-os integrations
discovery-os integrations --profile all-open
```

`manifest_revision`은 자신을 제외한 canonical JSON의 SHA-256입니다. 내용이 바뀌면 이 값도 다시 계산해야 하며 불일치 시 설치 전에 중단합니다.

저장소의 기본 manifest는 bootstrap 코드에 고정된 trusted revision과도 일치해야 합니다. custom manifest는 명시적인 `--manifest ... --allow-custom-manifest` opt-in이 필요합니다. custom 파일의 자체 SHA-256은 손상·우발적 변경은 잡지만 작성자 신뢰나 공급망 진위를 증명하지 않습니다. 검토·서명한 파일만 사용하십시오.

`generated_at`과 `resolution_cutoff`은 timezone이 있는 RFC3339 값이어야 하며 cutoff ≤ generated ≤ 현재 시각을 만족해야 합니다. cutoff는 의존성 resolver의 시간 경계이지 이미 만들어진 transitive lock을 의미하지 않습니다.

## install과 runtime service는 분리되어 있다

manifest의 package/source/weight 항목은 독립 환경을 준비하기 위한 명세입니다. `api.protocol`과 `base_url_env`는 연결 위치를 고정합니다. bootstrap은 공통 sidecar 코드까지 각 API 환경에 설치하지만 프로세스를 자동 시작하지 않습니다. 시작은 `start-sidecars.ps1`/`.sh`가 담당합니다.

- 전문 특징 `POST /v1/features`
- 사용자 Fusion `POST /v1/fuse`, `POST /v1/revise`
- 후보 생성 `POST /v1/generate`
- 장시간 계산기 job/cancellation endpoint

따라서 package가 설치되거나 source가 준비되어도 API URL이 없으면 expert registry는 `available=false`입니다. 실제 지원 adapter, 수동 checkpoint, projection semantics와 strict fixture 수락 범위는 [sidecar 실행 문서](../docs/SIDECARS.md)에 있습니다. Boltz 2.2.1은 공식 CLI codec으로, scGPT 0.2.4는 고정 model/vocab/config bundle로 연결됩니다. QHNet은 bootstrap이 검증한 AIRS source와 명시적 checkpoint/config bundle을 공식 graph path에 연결하며, dataset 범위가 맞지 않으면 합성 출력 없이 실패합니다.

## weight 상태

- `huggingface`: exact 40자 revision으로 자동 받을 수 있지만 license/credential 조건을 먼저 만족해야 합니다.
- `manual`: 목표별 checkpoint 선택이 필요하므로 `manual_download_required`로 남습니다.
- `managed`: upstream이 실행 시 관리하므로 `managed_by_upstream`으로 남습니다.
- gated weight: 접근 승인과 token이 없으면 `credential_required`입니다.

`--include-weights`를 사용하지 않으면 `not_requested`이며 구성요소 설치 완료 여부와 분리합니다. `--include-weights --require-all`에서는 manual/managed/license/credential 미해결 상태를 완전 성공으로 간주하지 않습니다.

설치 후 생성되는 `inventories/*.freeze.txt`는 관찰된 package inventory입니다. hash가 포함된 사전 dependency lock이나 모든 플랫폼에서 재생 가능한 lockfile이 아닙니다.

자세한 설치·신뢰·수동 항목 경계는 [격리 환경 문서](../docs/DEPENDENCIES.md), wire 규격은 [Expert/Fusion/Generator API 계약](../docs/EXPERT_API_CONTRACT.md)을 참조하십시오.
