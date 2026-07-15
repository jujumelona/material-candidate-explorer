# 실제 모델 sidecar 실행

각 전문 모델은 자기 Python/CUDA 환경에서 실행되고 중앙 Fusion Core와는 JSON 계약으로만 통신합니다. 공통 서버 구현은 `python -m discovery_os.sidecars --model <id>`이며 다음 endpoint를 제공합니다.

```text
GET  /health
POST /v1/features   # expert-feature-v1 모델
POST /v1/generate   # generator-v1 모델
```

서버는 모델을 첫 요청에서 한 번만 lazy-load하고, 실패한 load도 재시작 전까지 기억합니다. 요청 크기, 생성 batch 크기, 동시 worker, 대기 queue와 timeout을 제한하며 strict schema·후보 hash·부모 lineage·model/code/weight provenance를 검사합니다. 전문가 요청은 현재 한 후보씩 처리되고, 생성기는 `candidate_count`만큼의 결과를 한 요청에서 반환합니다.

## 구현 상태

| 모델 | 포트 | 현재 sidecar가 실제 호출하는 경로 | 중요한 경계 |
|---|---:|---|---|
| MatterGen | 8101 | MatterGen `CrystalGenerator` | 명시적 목표/수정 target을 condition으로 변환. raw latent, 부모 구조 mutation, temperature·mutation·diversity는 upstream 입력이 아니므로 경고 |
| REINVENT4 | 8112 | 고정 argv의 공식 CLI sampling | temperature·randomized SMILES·uniqueness와 선택적 Mol2Mol 부모 seed만 반영. raw latent/물성 수정은 소비하지 않음 |
| Uni-Mol | 8102 | `unimol_tools.UniMolRepr` | SMILES 또는 SDF/XYZ 원자·좌표를 공식 conformer dict로 변환 |
| UMA | 8109 | `fairchem.core.FAIRChemCalculator` | `UMA_TASK_NAME=omat`은 주기 CIF/POSCAR, `omol`은 비주기 XYZ/EXTXYZ/SDF만 허용. 안정된 hidden API가 없어 energy/force/stress observable을 반환 |
| MatterSim | 8110 | `MatterSimCalculator` | 주기 `CRYSTAL_MATERIAL + CIF/POSCAR`의 energy/force/stress만 반환 |
| CHGNet | 8113 | `CHGNet.predict_structure` | 주기 `CRYSTAL_MATERIAL + CIF/POSCAR`의 energy/force/stress/magnetic moment를 반환 |
| Chemprop | 8111 | Chemprop v2 `MoleculeDatapoint` MPNN load/encoding/predict | 현재 경로는 `SMALL_MOLECULE/CATALYST + SMILES`만 허용하며 task checkpoint와 property 이름을 사용자가 고정해야 함 |
| ESM3 | 8104 | Biohub `ESM3.encode`/`logits` | `PROTEIN_SEQUENCE` embedding만 반환. PDB 입력도 검증된 단일 chain 서열을 추출할 뿐 좌표는 사용하지 않음 |
| RNA-FM | 8105 | `fm.pretrained.rna_fm_t12` | `RNA_SEQUENCE` nucleotide embedding이며 3D RNA 특징으로 광고하거나 라우팅하지 않음 |
| PySCF | 8108 | 분자 RHF/UHF SCF | 비주기 `SMALL_MOLECULE/CUSTOM + XYZ/SDF`만 허용하는 CPU Hartree–Fock energy/orbital 결과이며 PBC·DFT가 아님 |
| Boltz | 8103 | 공식 2.2.1 `boltz predict` CLI | 단일 protein/RNA sequence 또는 SMILES만 자체 생성 YAML로 변환. model 0 confidence와 mmCIF 구조 요약을 observable tensor로 반환하며 hidden embedding을 합성하지 않음 |
| scGPT | 8106 | `TransformerModel` + `scgpt.preprocess.binning` | `args.json`·`vocab.json`·`best_model.pt` bundle을 inventory hash로 고정하고 실제 `<cls>` cell embedding 반환 |
| QHNet | 8107 | 고정 AIRS `models.get_model` + PyG `Data` | bootstrap source marker·실행 파일 digest와 수동 checkpoint/config bundle을 검증하고 전체 Hamiltonian만 반환. checkpoint에 overlap 출력이 없으므로 orbital energy/coefficients는 합성하지 않음 |

`/health`가 `status=lazy`인 것은 runtime과 identity가 기동됐다는 뜻이며 checkpoint 추론 성공을 뜻하지 않습니다. 실제 배포 수락에는 모델별 고정 fixture를 `/v1/features` 또는 `/v1/generate`로 통과시켜야 합니다. scGPT와 QHNet은 bundle 정적 검증 후 lazy 상태로 기동하고 첫 실제 요청에서 checkpoint를 로드합니다.

Boltz adapter는 [공식 v2.2.1 prediction 계약](https://github.com/jwohlwend/boltz/blob/v2.2.1/docs/prediction.md)에 맞춰 `confidence_request_model_0.json`, `request_model_0.cif`, 선택적 `affinity_request.json`만 크기 제한과 함께 읽습니다. 현재 중앙 `ExpertFeatureRequest`는 workspace 전체가 아니라 한 entity만 전달하므로 protein-ligand complex affinity 입력은 아직 표현할 수 없습니다. 따라서 SMILES 단독 route를 결합 affinity로 표시하지 않으며, 단백질은 외부 MSA 경로나 임의 YAML을 받지 않고 공식 `msa: empty` 단일서열 모드로 실행합니다.

## 한 번에 설치하고 기동하기

Windows PowerShell에서 Linux 전용 모델이 포함된 프로필은 `-Backend auto`가 사용 가능한 WSL을 선택합니다. 저장 공간이 작은 시스템에서는 충분한 다른 드라이브를 명시합니다.

```powershell
.\bootstrap.ps1 `
  -Profile all-open `
  -Backend auto `
  -InstallRoot "I:\DiscoveryOS\scientific-envs" `
  -AllowExternalRoot `
  -Accelerator cuda `
  -IncludeWeights `
  -AcceptLicense esm

.\start-sidecars.ps1 `
  -Component mattergen,unimol,boltz,esm,rnafm,pyscf `
  -Backend auto `
  -InstallRoot "I:\DiscoveryOS\scientific-envs" `
  -AllowExternalRoot

. "I:\DiscoveryOS\scientific-envs\sidecars.env.ps1"
```

Linux에서는 다음과 같습니다.

```bash
./bootstrap.sh --profile all-open --accelerator cuda --include-weights --accept-license esm
./start-sidecars.sh --component mattergen --component unimol --component boltz --component esm \
  --component rnafm --component pyscf
. ./.discovery/sidecars.env.sh
```

위 시작 예시는 자동/고정 weight로 동작하는 구현부터 선택합니다. MatterSim·CHGNet·Chemprop·REINVENT·scGPT·QHNet을 함께 시작하려면 아래 수동/관리 checkpoint 변수를 먼저 채운 뒤 `--component`/`-Component`에 추가하십시오. 필수 bundle이나 검증된 bootstrap source가 빠진 선택은 preflight에서 즉시 거부되며 어떤 프로세스도 부분 기동하지 않습니다.

기동기는 loopback 포트 충돌과 기존 live PID state를 거부하고, stdout/stderr log와 PID/provenance state를 `InstallRoot` 아래에 기록합니다. Windows 프로세스는 숨김 창으로 실행됩니다. readiness가 실패하면 프로세스를 몰래 종료하지 않고 state와 log 경로를 남깁니다.

## 수동 또는 관리 checkpoint

manifest의 `manual`/`managed` weight는 임의 파일을 자동 선택하지 않습니다. 시작 전에 최소한 다음처럼 실제 파일과 immutable revision을 명시합니다.

Windows용 전체 입력 틀은 [`integrations/manual-checkpoints.env.example.ps1`](../integrations/manual-checkpoints.env.example.ps1)에 있습니다. 그대로 실행하지 말고 별도 운영 파일로 복사해 placeholder를 모두 교체하십시오. 자동 snapshot 모델의 내부 경로는 launcher가 검증된 `InstallRoot`에서 계산하므로 이 파일에서 임의로 덮어쓰지 않습니다.

```powershell
$env:CHEMPROP_CHECKPOINT_PATH = "D:\models\chemprop\task.ckpt"
$env:CHEMPROP_WEIGHT_REVISION = "sha256:..."
$env:CHEMPROP_PROPERTY_NAMES = "solubility,toxicity"
$env:CHEMPROP_PROPERTY_UNITS = "mol/L,dimensionless"

$env:REINVENT_MODEL_FILE = "D:\models\reinvent\prior.model"
$env:REINVENT_WEIGHT_REVISION = "sha256:..."

$env:MATTERSIM_WEIGHT_REVISION = "managed-unattested:upstream-release-and-cache-id"
$env:CHGNET_WEIGHT_REVISION = "managed-unattested:chgnet-0.3.0-builtin"

# Boltz는 bootstrap의 검증된 snapshot을 자동 바인딩한다. 아래는 운영 조정값이다.
$env:BOLTZ_CACHE = "I:\DiscoveryOS\scientific-envs\cache\boltz-runtime"
$env:BOLTZ_PROCESS_TIMEOUT_SECONDS = "840"
$env:BOLTZ_NO_KERNELS = "false"
```

Chemprop property 이름과 단위는 checkpoint 출력 순서와 정확히 같은 개수로 지정해야 합니다. 둘 중 하나가 없거나 출력 폭이 다르면 property를 조용히 버리지 않고 기동 또는 추론을 실패시킵니다. `REINVENT_WEIGHT_REVISION`은 launcher가 선택한 prior 파일의 실제 SHA-256과 대조합니다. prior의 배포 출처·DOI·라이선스는 파일 hash와 별개인 운영 기록이므로 모델 카드/배포 기록에도 함께 보존해야 합니다.

scGPT는 `SCGPT_CHECKPOINT_DIR` 안에 호환되는 `args.json`, `vocab.json`, `best_model.pt`가 모두 있어야 합니다. launcher와 sidecar는 bundle inventory SHA-256과 config/vocab/checkpoint digest를 provenance에 묶고, encoder 경로의 모든 tensor와 shape를 확인합니다. Torch의 안전한 `weights_only` 역직렬화를 지원하지 않는 runtime은 거부하며, 임의 checkpoint 선택이나 누락 weight의 무작위 초기화는 금지합니다.

```powershell
$env:SCGPT_CHECKPOINT_DIR = "D:\models\scgpt\whole-human"
$env:SCGPT_MAX_LENGTH = "1200"
$env:SCGPT_USE_FAST_TRANSFORMER = "false"

$env:QHNET_CHECKPOINT_PATH = "D:\models\qhnet\water_results.pt"
$env:QHNET_CONFIG_PATH = "$PWD\integrations\qhnet.water.config.example.json"
# 선택 사항: launcher가 계산한 bundle-sha256 값을 운영 설정에서 다시 고정할 때만 지정
# $env:QHNET_WEIGHT_REVISION = "bundle-sha256:..."
```

QHNet config는 checkpoint와 분리해서 해석할 수 없는 dataset 범위를 고정합니다. `model_version`, dtype, basis/단위, 중성·단일항 조건과 허용된 **원자번호 순서**를 모두 적어야 합니다. sidecar는 XYZ/SDF 좌표 Å를 upstream과 같은 `1.8897261258369282` 배율로 Bohr로 바꾸고, 해당 순서가 config에 없거나 full Hamiltonian이 65,536개 wire 값 한도를 넘으면 pooling/truncation 없이 거부합니다. 예시는 [`integrations/qhnet.water.config.example.json`](../integrations/qhnet.water.config.example.json)이며 실제 checkpoint dataset에 맞게 별도 파일로 복사해 고정해야 합니다.

scGPT cell-expression representation은 한 세포만 담는 다음 JSON 형식입니다.

```json
{
  "genes": ["TP53", "GAPDH"],
  "values": [12, 340],
  "value_semantics": "raw_counts"
}
```

`value_semantics="raw_counts"`는 전체 caller gene vector의 library size를 기준으로 `normalize_total=10000 → log1p → exact vocab filter → binning(51)`을 적용합니다. OOV gene count도 정규화 분모에 포함됩니다. 이미 같은 정규화와 `log1p`를 마친 값만 `value_semantics="normalized_log1p"`로 보내며, 이 경우 추가 normalize/log 변환 없이 vocab filter와 binning(51)만 수행합니다. 의미 표기가 없거나 raw count가 음수·분수이면 요청을 거부합니다.

## 개별 실행과 제한

```powershell
$env:SIDECAR_DEVICE = "cuda"
# 모델별 설정이 공통값보다 우선합니다. GPU 번호가 겹치지 않게 배치할 수 있습니다.
$env:MATTERGEN_DEVICE = "cuda:0"
$env:MATTERGEN_CUDA_VISIBLE_DEVICES = "0"
$env:UMA_DEVICE = "cuda:0"
$env:UMA_CUDA_VISIBLE_DEVICES = "1"
$env:SIDECAR_MAX_BATCH_SIZE = "16"
$env:SIDECAR_MAX_CONCURRENCY = "1"
$env:SIDECAR_MAX_QUEUE_SIZE = "2"
$env:SIDECAR_TIMEOUT_SECONDS = "900"
python -m discovery_os.sidecars --model unimol --host 127.0.0.1 --port 8102
```

운영 배포에서는 loopback 밖으로 직접 노출하지 말고 TLS, 인증, secret manager, process supervisor와 GPU 격리를 별도로 적용하십시오. 중앙 코드의 loopback HTTP 사용에는 생성된 env 파일이 `DISCOVERY_ALLOW_INSECURE_LOCAL_HTTP=1`을 명시합니다.
