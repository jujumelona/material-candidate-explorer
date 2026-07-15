# 격리 환경과 한 명령 설치

전문 엔진은 Python·Torch·CUDA·NumPy 요구사항이 서로 충돌하므로 하나의 거대한 환경에 설치하지 않습니다. bootstrap은 manifest를 검증한 뒤 구성요소별 독립 환경, source와 weight 저장소를 준비합니다.

## 가장 짧은 명령

Windows PowerShell에서 공개 구성요소 전체를 설치·다운로드 시도합니다.

```powershell
.\bootstrap.ps1 -Profile all-open -Accelerator cuda -IncludeWeights -AcceptLicense esm
```

UMA까지 포함하려면 Hugging Face에서 먼저 접근 승인을 받고 약관 수락과 token을 명시합니다. 설치기는 약관을 대신 수락하거나 token을 상태 파일에 저장하지 않습니다.

```powershell
$env:HF_TOKEN = "<사용자 토큰>"
.\bootstrap.ps1 -Profile all -Accelerator cuda -IncludeWeights -AcceptLicense esm,uma
```

Linux에서는 다음과 같습니다.

```bash
./bootstrap.sh --profile all-open --accelerator cuda --include-weights --accept-license esm
```

이 명령은 package, 검증된 source archive, 자동 다운로드 가능한 weight와 공통 sidecar 코드를 각 격리 환경에 설치합니다. 설치 후 `start-sidecars.ps1`/`.sh`로 구현된 endpoint를 기동할 수 있습니다. 수동/관리 checkpoint 선택, 라이선스·credential, 모델별 고정 fixture 추론 수락은 여전히 별도입니다.

## 설치 전에 계획 확인

다운로드·설치 없이 manifest와 플랫폼별 계획을 확인할 수 있습니다.

```powershell
python scripts/bootstrap.py verify-manifest
.\bootstrap.ps1 -Profile all-open -Accelerator cuda -DryRun -AcceptLicense esm
python scripts/bootstrap.py plan --profile all-open --accelerator cuda --accept-license esm
```

완전성이 필요한 자동화에서는 `-RequireAll`/`--require-all`을 추가합니다. 선택한 구성요소가 unsupported platform/accelerator, license·credential 필요, 설치 실패 또는 disk 부족 상태이면 exit code 2가 됩니다. `-IncludeWeights`도 지정했다면 manual, gated, credential/license 필요, upstream-managed weight도 미완료로 취급합니다. weight를 요청하지 않았다면 `not_requested`이고 구성요소 설치의 완료 판정에는 포함하지 않습니다.

```powershell
.\bootstrap.ps1 -Profile molecule-modern -Accelerator cpu -RequireAll
```

`all-open -IncludeWeights -RequireAll`은 scGPT/QHNet/Chemprop/REINVENT의 수동 checkpoint와 MatterSim/CHGNet의 upstream-managed weight 때문에 의도적으로 완전 성공하지 않을 수 있습니다. 이것은 실패를 숨기지 않기 위한 동작입니다.

## Windows, WSL과 설치 위치

PowerShell의 `-Backend auto`는 Linux 중심 프로필에서 실제 사용 가능한 WSL 배포판과 `python3`/`venv`를 확인한 뒤 WSL을 사용합니다. 조건을 만족하지 않으면 native 계획으로 돌아가고 지원되지 않는 구성요소를 명시합니다. `-Backend wsl`을 직접 지정했는데 usable 배포판이 없으면 즉시 오류입니다.

기본 설치 위치의 공간이 부족하면 다른 드라이브를 명시할 수 있습니다. workspace 밖 경로는 실수로 쓰지 않도록 별도 opt-in이 필요합니다.

```powershell
.\bootstrap.ps1 `
  -Profile all-open `
  -InstallRoot "I:\DiscoveryOS\scientific-envs" `
  -AllowExternalRoot `
  -Accelerator cuda `
  -AcceptLicense esm
```

설치기는 쓰기 전에 manifest의 resource hint를 합산해 보수적인 여유 공간을 검사합니다. cache, 임시 파일과 환경도 지정 root 아래에 둡니다. CUDA/MPS를 선택했더라도 해당 구성요소가 CPU만 지원하면 구성요소 단위로 CPU fallback을 사용할 수 있습니다. 예를 들어 CUDA 프로필에 포함된 PySCF를 accelerator 불일치만으로 건너뛰지 않습니다.

## bootstrap이 수행하는 일

1. strict manifest schema, identifier/path, default manifest의 신뢰된 revision을 검사합니다.
2. RFC3339 `generated_at`/`resolution_cutoff` 순서와 미래 시점을 검사하고 profile dependency DAG를 풉니다.
3. 플랫폼·구성요소별 accelerator, 라이선스, credential과 예상 저장공간을 계획합니다.
4. 고정된 uv bootstrap과 필요한 Python minor를 사용해 구성요소별 환경을 만듭니다.
5. 정확한 top-level package version과 manifest cutoff를 사용해 의존성을 풉니다.
6. GitHub archive의 기록된 byte size와 SHA-256을 확인하고 symlink, path traversal, 과도한 파일 수·단일 파일·압축 해제 크기를 제한합니다.
7. Hugging Face snapshot은 manifest의 정확한 40자 revision으로 받습니다.
8. 실제 설치된 package 목록을 `inventories/*.freeze.txt`에 기록하고 secret이 없는 install state를 원자적으로 저장합니다.

기본 manifest는 bootstrap 코드에 고정된 trusted revision과 일치해야 합니다. 다른 manifest를 쓰려면 `-Manifest`/`--manifest`와 `-AllowCustomManifest`/`--allow-custom-manifest`를 명시해야 합니다. custom manifest의 자체 hash 검사는 전송 무결성만 확인할 뿐 작성자의 신뢰성을 증명하지 않습니다. 조직에서 검토·서명한 manifest만 사용하십시오.

## freeze inventory는 lock이 아니다

`inventories/<component>.freeze.txt`는 **설치 후 실제 환경을 관찰한 inventory**입니다. 사전에 모든 transitive dependency와 artifact hash를 고정한 replayable lock이 아닙니다. 상태에도 `replayable_lock: false`로 기록됩니다.

따라서 현재 재현성 보장은 다음 범위입니다.

- manifest의 정확한 top-level package version
- exact source commit과 검증된 archive SHA-256/size
- exact Hugging Face revision
- resolution cutoff와 사용한 index/find-links
- 설치 후 package inventory 및 component state

장기 보존·규제·배포 재현성이 필요하면 지원할 OS/architecture/CUDA별로 별도 검증된 lockfile 또는 wheelhouse/container digest를 만들고 서명해야 합니다. `*.freeze.txt`를 그런 lock으로 부르거나 그대로 재설치 명세로 사용해서는 안 됩니다.

## 왜 환경이 여러 개인가

아래는 manifest가 고정하는 주요 top-level 버전과 환경 경계입니다. transitive dependency 전체 목록은 설치 후 inventory에서 확인합니다.

| 구성요소 | 고정 버전/소스 | Python | 핵심 환경 조건 | 기본 플랫폼 |
|---|---:|---:|---|---|
| Discovery OS | 0.2.0 | 3.11 | Pydantic 2.13.4, requests 2.34.2, RDKit 2026.3.3; sidecar FastAPI 0.139.0/Uvicorn 0.51.0 | Windows/Linux/macOS |
| MatterGen | 1.0.3 | 3.10 | cu118 index, MatterSim 1.1.2 constraint | Linux CUDA |
| Uni-Mol Tools | 0.1.6 | 3.11 | 독립 modern 환경 | Windows/Linux/macOS |
| Boltz | 2.2.1 | 3.11 | 선택형 CUDA extra | Linux |
| ESM | 3.2.3 | 3.12 | 명시적 ESM license 수락 | 로컬/API |
| RNA-FM | 0.2.2 | 3.10 | legacy 생태계와 분리 | Windows/Linux/macOS |
| scGPT | 0.2.4 | 3.11 | checkpoint는 수동 선택 | Windows/Linux/macOS |
| QHNet | exact AIRS archive/commit | 3.10 | source는 직접 import하지 않고 marker/실행 파일 digest 검증; Torch 2.2.0, PyG 2.5.3, e3nn 0.5.1 호환 환경 | Linux CUDA |
| PySCF | 2.13.1 | 3.11 | native Windows 미지원 | Linux/macOS, CPU fallback |
| UMA | fairchem-core 2.21.0 | 3.12 | gated weight와 HF token 필요 | Linux CUDA |
| MatterSim | 1.2.5 | 3.12 | checkpoint는 upstream 관리 | Linux |
| CHGNet | 0.4.2 | 3.12 | packaged/upstream-managed 0.3.0 model | Windows/Linux/macOS |
| Chemprop | 2.2.4 | 3.11 | task checkpoint는 사용자 제공 | Windows/Linux/macOS |
| REINVENT4 | exact v4.8 archive/commit | 3.11 | cu126 index, prior는 수동 선택 | Linux/macOS |

정확한 package, constraint, index, source commit, archive SHA-256/size, weight revision, 라이선스 URL, resource hint와 API 환경변수는 [`integrations/manifest.v1.json`](../integrations/manifest.v1.json)에 있습니다. manifest에는 실행 가능한 shell/Python command 필드가 없습니다.

## 프로필

| 프로필 | 내용 |
|---|---|
| `core` / `fusion` | 공통 스키마, RDKit, Fusion/API client |
| `molecule-modern` | Uni-Mol + Chemprop |
| `molecule-generation` | REINVENT4 source/environment |
| `materials-open` | MatterGen + 최신 MatterSim + CHGNet, 서로 다른 환경 |
| `biology-open` | Boltz + ESM + RNA-FM + scGPT |
| `electronic-open` | PySCF + QHNet 고정 source/격리 runtime |
| `uma` | fairchem-core + 승인 필요한 UMA weight |
| `all-open` | UMA를 제외한 모든 공개 source/package |
| `all` | `all-open` + UMA |

`all-open`은 모든 라이선스·weight가 무조건 자동 사용 가능하다는 뜻이 아닙니다. ESM은 명시적인 조건 수락이 필요하며 설치기는 약관을 자동 수락하지 않습니다.

## 자동화할 수 없거나 의도적으로 남기는 항목

| 구성요소 | 상태 | 사용자가 해야 할 일 |
|---|---|---|
| scGPT | `manual_download_required` | 목표 dataset/task의 `args.json`·`vocab.json`·`best_model.pt`를 한 디렉터리에 두고 `SCGPT_CHECKPOINT_DIR`로 지정; launcher가 전체 inventory SHA-256 기록 |
| QHNet | `manual_download_required` | 목표 MD17 checkpoint와 그 dataset 범위를 적은 `QHNET_CONFIG_PATH`를 함께 지정; launcher가 두 파일을 `bundle-sha256`으로 결합 검증 |
| Chemprop | `manual_download_required` | task `.ckpt`를 선택하고 `CHEMPROP_CHECKPOINT_PATH`, 실제 SHA-256, checkpoint 출력 순서와 같은 `CHEMPROP_PROPERTY_NAMES`·`CHEMPROP_PROPERTY_UNITS` 지정 |
| REINVENT4 | `manual_download_required` | 사용할 prior 선택, checksum·라이선스 고정 |
| MatterSim | `managed_by_upstream` | 첫 실행 cache/weight 동작과 실제 revision을 별도 기록 |
| CHGNet | `managed_by_upstream` | packaged/upstream model 이름과 실제 runtime attestation을 별도 기록 |
| UMA | `credential_required`/`license_required` | HF 접근 승인, `HF_TOKEN`, UMA license 수락 |
| 시스템 | 수동 | WSL 배포판, Docker, NVIDIA driver 등 관리자 권한·재부팅 항목 설치 |

이 상태들은 `partial` 결과와 state에 그대로 남습니다. 강제 복구하거나 임의 checkpoint를 대신 고르지 않습니다.

## 설치 후 해야 할 일

1. `doctor`와 environment inventory로 import 상태를 확인합니다.
2. 목표에 맞는 checkpoint를 고르고 immutable revision/hash를 기록합니다.
3. [sidecar 실행 문서](SIDECARS.md)에 따라 구현된 runtime과 checkpoint 환경변수를 연결합니다. Boltz 2.2.1 adapter는 검증된 snapshot과 공식 CLI를 사용합니다. scGPT 0.2.4는 `SCGPT_CHECKPOINT_DIR`의 model/vocab/config bundle 전체 inventory hash를 검증합니다. QHNet은 bootstrap의 고정 AIRS source와 사용자가 지정한 `QHNET_CHECKPOINT_PATH`·`QHNET_CONFIG_PATH`를 모두 검증한 뒤 공식 graph/model 경로를 호출합니다.
4. `/health`와 고정 fixture strict-schema 추론을 모두 통과시킵니다. `status=lazy`만으로 모델 checkpoint 성공을 판단하지 않습니다.
5. `UNIMOL_API_URL`, `BOLTZ_API_URL`, `MATTERGEN_API_URL` 같은 manifest/code-owned 환경변수로 sidecar를 연결합니다. 원격 Fusion AI를 선택할 때만 `FUSION_API_URL`을 설정하며, 없으면 로컬 `EvidenceDrivenFusionBackend`가 자동 사용됩니다.

이 단계까지 끝나야 “설치됨”이 아니라 “Discovery OS에서 사용 가능한 expert/generator”라고 할 수 있습니다.
