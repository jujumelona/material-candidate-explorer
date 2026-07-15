# 도메인별 검증 행렬

이 문서는 Discovery OS가 후보를 어떻게 **탈락시키고, 계산 리드로 남기고, 실험 및 독립 재현으로 승격할지** 정의하는 설계 기준입니다. 여기에 적힌 검증 항목 전체가 현재 구현되어 있다는 뜻은 아닙니다.

## 상태 표기

| 표기 | 현재 의미 |
|---|---|
| **실행 가능 MVP** | 이 저장소의 mock 경로와 로컬 코드로 수행 가능한 계약·기초 검사·증거 기록 |
| **연결 규격 준비·backend 필요** | 모델 래퍼는 실행 가능하지만 실제 로컬 모델 또는 HTTP endpoint는 사용자가 제공 |
| **커넥터 자리** | 실제 도구, 데이터, 모델, 라이선스 또는 계산 자원이 연결되어야 수행 가능 |
| **실험실 증거** | 합성·측정 기관에서 만든 원자료와 provenance를 가져와야 하며 소프트웨어가 생성할 수 없음 |

프로필은 필요한 검증 **recipe와 gate**를 선언합니다. 프로필이 등록되어 있어도 표에서 `커넥터 자리` 또는 `실험실 증거`로 표시한 항목이 자동으로 실행되거나 충족되는 것은 아닙니다. 실행 시에는 registry가 실제로 제공하는 operation만 컴파일할 수 있어야 하며, 없는 필수 게이트는 `insufficient_evidence`입니다.

## 공통 증거 등급과 판정

| 등급 | 허용되는 표현 | 최소 요건 |
|---|---|---|
| E0 | `generated`, `unvalidated` | schema와 후보 identity만 존재 |
| E1 | `screened` | 기초 유효성·중복·정책 검사; 성능 증명 아님 |
| E2 | `computationally_supported` | 적용 범위와 불확실성이 있는 하나 이상의 적절한 계산, 재현 정보, 필수 계산 게이트 통과 |
| E3 | `experimentally_observed` | 동정된 시료, 사전 정의한 조건·대조군·반복, 원자료와 측정 불확실성 |
| E4 | `independently_replicated` | 독립 시료와 독립 팀/시설이 같은 판정 기준을 통과하고 상충 결과가 해소됨 |

E0~E2는 **발견을 위한 계산 리드**입니다. 공개적인 신물질 발견 주장은 해당 분야의 E3 필수 실험을 통과한 뒤 E4 독립 재현까지 요구합니다. 의약 후보는 E4 이후에도 임상적 안전성·유효성이나 규제 승인을 뜻하지 않습니다.

### 모든 도메인에 적용하는 필수 게이트

| 게이트 | 검사 | 실패 또는 누락 시 정책 | 상태 |
|---|---|---|---|
| G0 계약 | schema version, 타입, 유한 숫자, 단위·조건, 알 수 없는 필드 | 계획 또는 증거 거부 | **실행 가능 MVP** |
| G1 실행 권한 | 등록된 tool/operation, 후보 참조, typed 조건, 의존성, timeout·자원 예산 | 컴파일 거부; 모델이 우회 불가 | **실행 가능 MVP** |
| G2 provenance | 후보 버전, 도구·모델·데이터 버전, 파라미터·seed, input/output hash, artifact | 증거 미충족 또는 품질 경고 | **실행 가능 MVP** |
| G3 계산 신뢰성 | 수렴, 반복성, 독립적인 방법, OOD/applicability, 불확실성, 음성 대조 | `computationally_supported` 승격 금지 | **커넥터 자리** |
| G4 시료·실험 | identity·purity/phase, protocol, 조건, 장비, 대조군, 반복, 원자료, 허가된 verifier의 attestation | 실험 관찰 주장 금지; 미검증 제출은 `partial/unverified` | **실험실 증거** |
| G5 독립 재현 | 별도 합성 batch와 가능한 한 독립 팀/시설·장비, 사전 정의 기준 | 발견 확정 표현 금지 | **실험실 증거** |
| G6 주장 범위 | 신규성 검색 범위, 측정 조건, 한계, 실패·상충 결과 공개 | 과장된 결론 차단 | MVP report + 외부 검토 |

`success`인 tool run은 프로그램이 정상 종료했다는 뜻입니다. gate 통과는 필요한 `EvidenceRecord`의 속성·조건·품질과 provenance를 코드 정책이 별도로 판정합니다. 모델의 `confirmed_findings`나 `finish` 제안만으로 gate를 통과할 수 없습니다.

## 1. 의약 저분자 (`medicinal_chemistry`)

[RDKit sanitization](https://www.rdkit.org/docs/RDKit_Book.html)은 허용 원자가, 방향족성, kekulization 등 구조 표현의 합리성을 검사하지만 실제 안정성·합성 가능성·약효·안전성을 증명하지 않습니다. 의약품은 발견 이후에도 비임상, 임상 및 규제 검토가 필요합니다. [FDA의 전체 개발 절차](https://www.fda.gov/patients/learn-about-drug-and-device-approvals/drug-development-process)를 기준 경계로 삼습니다.

| 단계 | 필수 또는 권장 검증 | 통과 증거 | 현재 상태 |
|---|---|---|---|
| 표현 기초 | SMILES parse/sanitize, canonical/isomeric 표현, 원자가·전하·fragment, 미지정 입체화학 경고 | 구조와 경고가 있는 계산 증거 | **실행 가능 MVP**: 선택적 RDKit 기본 검사 범위만 |
| 기초 descriptor | MW, cLogP, TPSA, HBD/HBA, 회전 결합 등 정확한 정의·버전 | 값·단위·RDKit 버전 | **실행 가능 MVP**: 어댑터가 제공하는 기본 항목만 |
| 중복·신규성 | tautomer/salt/stereoisomer 정책, exact·scaffold 유사도, 명시한 DB snapshot 검색 | 검색 DB·버전·날짜와 hit | **커넥터 자리**; 미검색은 세계 최초 증거가 아님 |
| 화학 위험·간섭 | 반응성·독성 구조 alert, PAINS/Brenk 등; 경고의 적용 범위와 오탐 검토 | alert와 전문가 판정 | **실행 가능 MVP**: RDKit PAINS/Brenk 경고만 제공하며 자동 탈락시키지 않음 |
| 3D·합성 | conformer/에너지 건전성, 합성 가능성 heuristic 또는 retrosynthesis, 시약·경로 위험 사전 gate | 방법·파라미터·불확실성 | **부분 실행 MVP**: RDKit SA heuristic만 제공; 3D·경로·위험 검증은 **커넥터 자리** |
| 표적 결합 계산 | docking/Boltz 등 구조 기반 순위화, pose·confidence·OOD, orthogonal 계산 | 계산 provenance와 비교 기준 | **커넥터 자리**; 결합 실험이 아님 |
| 동역학·고정밀 | OpenMM 반복 MD의 force-field coverage·수렴, 필요 시 FEP/QM | 반복 trajectory·수렴·오차 | **커넥터 자리** |
| ADME/Tox 예측 | endpoint 정의, 외부 성능, calibration, applicability domain, 불확실성 | endpoint별 모델·dataset version | **커넥터 자리**; 임상 안전성 증거가 아님 |
| 시료 확인 | 합성/조달, identity, purity, stereochemistry, 안정성 | 원자료, batch, 분석법 | **실험실 증거** |
| 생물학적 검증 | primary binding/functional assay, orthogonal assay, counterscreen, 농도-반응, 양성·음성 대조 | 반복값, effect size, uncertainty | **실험실 증거** |
| 후속 안전성 | 세포 활성·선택성, ADME, 독성 및 적절한 비임상 연구 | 승인된 protocol과 전체 결과 | **실험실 증거**; 의료 사용 근거와는 별도 |
| 독립 재현 | 별도 batch와 독립 연구실의 identity + 핵심 assay 반복 | 독립 provenance와 동일 판정 기준 | **실험실 증거** 필수 gate |

## 2. 무기·결정 소재 (`inorganic_materials`)

상 안정성은 formation energy 하나가 아니라 같은 계산 체계의 경쟁상으로 만든 convex hull과 비교해야 합니다. Materials Project도 계산 phase diagram이 주로 0 K/0 atm 근사이며 DFT와 유한 온도 한계가 있음을 설명합니다. [Materials Project phase diagram 방법론](https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/phase-diagrams-pds)을 참고합니다.

| 단계 | 필수 또는 권장 검증 | 통과 증거 | 현재 상태 |
|---|---|---|---|
| 조성 기초 | 화학식 token, 알려진 원소, 양의 유한 화학량론 계수, 조성 정규화 | canonical composition, parser version | **실행 가능 MVP** |
| 구조 기초 | CIF/POSCAR parse, 양의 cell volume, 유한 좌표, occupancy, 원자 간 거리·밀도 sanity | 표준화 구조와 경고 | **커넥터 자리**: pymatgen/spglib 등 필요 |
| 대칭·중복 | primitive/standard cell, tolerance 민감도, exact/StructureMatcher 중복 | 표준화 파라미터와 match | **커넥터 자리** |
| 화학 plausibility | 산화수·전하 중성·bond valence; 희귀 화학은 경고와 전문가 검토 | 규칙·예외·confidence | **커넥터 자리** |
| 구조 완화 | DFT 또는 검증된 MLIP relaxation, force/stress/energy 수렴, OOD | 초기/최종 구조와 수렴 기록 | **커넥터 자리** |
| 열역학 안정성 | 동일 functional·pseudopotential·보정 체계의 경쟁상, formation energy, energy above hull | reference snapshot과 hull artifact | **커넥터 자리**; DB에 경쟁상이 없으면 불충분 |
| 동적·기계 안정성 | phonon dispersion, imaginary mode 확인, elastic tensor와 안정 조건 | q-mesh·수렴·mode artifact | **커넥터 자리** |
| 조건 안정성 | 목표 온도·압력에서 free-energy 근사/AIMD, 분해·산화·수분 민감성 | 조건과 trajectory/오차 | **커넥터 자리** |
| 시료·상 확인 | 합성 protocol, 조성 분석, XRD/Rietveld·상 분율, 미세구조 | batch별 원자료 | **실험실 증거** |
| 목표 물성 | 온도·압력·방향·주파수 등 조건을 고정한 전기/광학/열/기계 측정 | 보정·대조·반복·불확실성 | **실험실 증거** |
| 독립 재현 | 독립 합성 batch/시설의 상 확인과 핵심 물성 | 독립 provenance | **실험실 증거** 필수 gate |

## 3. 초전도체 (`superconductors`)

계산상 높은 DOS, 전자-포논 결합 또는 예측 `Tc`는 초전도성의 실험 증거가 아닙니다. Quantum ESPRESSO의 [PHonon User's Guide](https://www.quantum-espresso.org/Doc/user_guide_PDF/ph_user_guide.pdf)는 선형 응답 격자 동역학 계산 도구의 범위를 설명합니다. NIST는 낮은 자기장에서의 flux expulsion인 [Meissner effect와 type-II vortex/pinning의 맥락](https://www.nist.gov/ncnr/flux-lattice-superconductors-and-melting)을 설명합니다.

| 단계 | 필수 또는 권장 검증 | 통과 증거 | 현재 상태 |
|---|---|---|---|
| 조성 기초 | 원소·화학량론 유효성, 후보·조건 identity | canonical composition | **실행 가능 MVP**; 초전도성 검사는 아님 |
| 상·구조 안정성 | DFT relaxation, 경쟁상 hull, phonon·elastic 안정성, 목표 압력 범위 | 수렴·reference phase·조건 | **커넥터 자리** |
| 전자구조 | 금속성, band/DOS at Fermi level, spin/SOC/U 민감도 | k-mesh·smearing·방법 비교 | **커넥터 자리** |
| phonon-mediated 가설 | 전자-포논 결합, `α²F`, `λ`, `ωlog`, Coulomb parameter 범위와 `Tc` 구간 | k/q mesh 수렴과 민감도 | **커넥터 자리**: QE PHonon/EPW 등; 비전통 후보의 탈락 단독 근거가 아님 |
| 계산 독립성 | 다른 코드/설정/방법의 재계산, 수치 오차와 applicability 검토 | 독립성 그룹과 상충 결과 | **커넥터 자리**; 여전히 계산 리드 |
| 시료·상 확인 | 합성, 조성, 결정구조, 상 분율, 압력/열처리 이력 | batch별 원자료 | **실험실 증거** |
| 전기 수송 | 4-probe `R(T,H,I)`, onset/midpoint/zero 기준, contact·self-heating·geometry 점검 | 원시 곡선, 검출 한계, 반복 | **실험실 증거** 필수 |
| 자기 검증 | DC/AC susceptibility, ZFC/FC, Meissner/diamagnetic shielding, demagnetization·volume fraction | 자기장·형상 보정과 원시 곡선 | **실험실 증거** 필수; 수송만으로 확정하지 않음 |
| 강건성·메커니즘 | 임계 자기장·전류, 가능하면 specific heat/isotope 및 다른 bulk probe | 조건별 전이의 일관성 | **실험실 증거** |
| 독립 재현 | 독립 시료·팀/시설에서 상 확인 + 수송 + 자기 검증 | 독립 provenance와 사전 기준 | **실험실 증거** 필수 gate |

초전도체 프로필의 계산 gate만 통과한 후보는 반드시 `computationally_supported`로 남습니다. 저항 감소만 관찰했거나 Meissner/자기 감수율 증거가 누락된 경우도 `insufficient_evidence`이며, 독립 재현 전에는 발견 확정으로 보고하지 않습니다.

## 4. 배터리 소재 (`batteries`)

| 단계 | 필수 또는 권장 검증 | 통과 증거 | 현재 상태 |
|---|---|---|---|
| 조성 기초 | 원소·화학량론, 후보 역할(전극/전해질/코팅 등), 목표 ion과 oxidation 범위 | canonical composition과 역할 | **실행 가능 MVP**: 화학식 범위만 |
| 구조·상 | 구조 sanity, relaxation, 경쟁상, 충·방전 상태별 상 변화 | 구조와 hull/phase artifact | **커넥터 자리** |
| 전압·용량 | redox 상태, 이론 용량, 평균/step 전압, 전자 전도성 | 반응식·단위·계산 조건 | **커넥터 자리** |
| 수송 | ion diffusion path/barrier, anisotropy, defect·농도·온도 의존성 | NEB/MD 수렴과 불확실성 | **커넥터 자리** |
| 계면·안전 | 전해질 안정 창, 계면 반응, dendrite/산소 방출·열 안정성 위험 | 조건별 반응·위험 flag | **커넥터 자리**; 실제 안전성 증거 아님 |
| 소재 확인 | 합성, 조성·상·입도·표면·수분 및 electrode formulation | batch와 분석 원자료 | **실험실 증거** |
| 셀 성능 | half-cell 이후 full-cell, areal loading, N/P, 전압 창, 온도, C-rate, formation protocol | capacity, CE, energy, rate, impedance, 원시 cycle data | **실험실 증거** |
| 수명·안전 | cycle/calendar life, 반복 cell 통계, post-mortem, 열·오용 시험 | failure 포함 전체 분포 | **실험실 증거** |
| 독립 재현 | 독립 batch/시설의 동일 셀 protocol 및 핵심 지표 | 독립 provenance | **실험실 증거** 필수 gate |

## 5. 촉매 (`catalysts`)

| 단계 | 필수 또는 권장 검증 | 통과 증거 | 현재 상태 |
|---|---|---|---|
| 표현 기초 | 분자/조성 유효성, 활성상·표면·지지체·결함·반응 조건 명시 | 후보 identity와 조건 | **실행 가능 MVP**: 분자/화학식 기초만 |
| 반응 열역학 | 흡착 에너지, 기준 상태, 용매·전위·온도·압력 보정 | 표면 모델·coverage·오차 | **커넥터 자리** |
| 반응 속도 | transition state·장벽, competing pathway, microkinetics와 민감도 | 경로·수렴·속도 결정 단계 | **커넥터 자리** |
| 선택성·열화 | 선택성, poisoning, sintering/leaching, 표면 재구성, 질량전달 한계 | 조건·시간에 따른 계산/모델 | **커넥터 자리** |
| 시료 확인 | 활성상, surface area, loading, oxidation state, morphology | batch별 characterization | **실험실 증거** |
| 성능 실험 | blank/지지체/양성 대조, conversion·selectivity·yield, TOF/STY의 명확한 분모, 물질수지 | 반복·error bar·원시 chromatogram 등 | **실험실 증거** |
| 안정성·operando | 장시간 성능, 회수 후 분석, 가능하면 operando로 실제 활성상 확인 | 시간축 데이터와 post-mortem | **실험실 증거** |
| 독립 재현 | 독립 합성 batch/시설에서 characterization + 핵심 반응 | 독립 provenance | **실험실 증거** 필수 gate |

## 6. 고분자 (`polymers`)

repeat unit 하나만으로 실제 고분자 시료를 정의할 수 없습니다. 분자량 분포, tacticity, branching/crosslink, 말단기, 첨가제, morphology와 가공 이력이 물성을 바꾸므로 후보와 측정 조건에 함께 기록해야 합니다.

| 단계 | 필수 또는 권장 검증 | 통과 증거 | 현재 상태 |
|---|---|---|---|
| 표현 기초 | repeat unit·연결점·조성 schema, 유효 원소, 금지 구조 경고 | canonical representation | **실행 가능 MVP**: 일반 표현/화학식 계약만; polymer 전용 완전 검증 아님 |
| 시료 정의 | 분자량/분포, tacticity, branching/crosslink, copolymer sequence, 첨가제·가공 조건 목표 | 누락 없는 후보 specification | **커넥터 자리** |
| 계산 예측 | conformer/MD 또는 QSPR로 density, `Tg/Tm`, modulus, permeability, dielectric 등; calibration/OOD | endpoint·조건·모델 버전·불확실성 | **커넥터 자리** |
| 합성·가공성 | 중합 경로, conversion, gelation, 용매·촉매·열 위험, 공정 window | 경로 feasibility와 위험 검토 | **커넥터 자리** |
| 구조·조성 확인 | NMR/FTIR 등 identity, GPC/SEC 분포, tacticity/branching, morphology | batch와 원자료 | **실험실 증거** |
| 열·기계·수송 | DSC/TGA/DMA, 인장/피로, 투과·전기 등 목표 시험; 습도·속도·두께·열 이력 명시 | 표준화 protocol, 반복, uncertainty | **실험실 증거** |
| 내구·수명 | aging, creep, solvent/UV/thermal stability, 재활용·분해와 첨가제 영향 | 시간축·failure 데이터 | **실험실 증거** |
| 독립 재현 | 독립 중합 batch/시설의 시료 정의 + 핵심 물성 | 독립 provenance | **실험실 증거** 필수 gate |

## 7. 일반 소재 (`general_materials`)

모든 소재에 통하는 만능 validator는 없습니다. 일반 프로필은 최소 공통 계약만 제공하며, 실제 후보 승격에는 도메인 플러그인이 property 정의, 단위·측정 조건, 안전 정책과 검증 recipe를 추가해야 합니다.

| 단계 | 필수 또는 권장 검증 | 통과 증거 | 현재 상태 |
|---|---|---|---|
| 공통 계약 | 후보 ID/version, representation kind, composition, parent lineage, schema·hash·artifact | 재현 가능한 provenance | **실행 가능 MVP** |
| 화학식 기초 | 지원되는 표현일 때 원소와 양의 화학량론 검사 | parser 결과 | **실행 가능 MVP** |
| 도메인 정의 | canonical property ID, 값 타입, 단위, 필수 조건, 목표 방향, 성공 threshold | versioned domain profile | **커넥터 자리**; 없으면 승격 금지 |
| 계산 recipe | 물리적으로 적절한 sanity→predictive→physics DAG, 수렴·uncertainty·OOD | 도메인별 evidence policy | **커넥터 자리** |
| 위험·윤리 | 독성·폭발성·환경·규제·생물학적 이중용도·자원 위험과 사람 승인 | 정책 버전과 승인 기록 | **커넥터 자리 + 사람 검토** |
| 실험 recipe | 시료 identity, 대조, 측정법, 조건, 반복과 acceptance criteria | 원자료가 있는 분야별 실험 | **실험실 증거** |
| 독립 재현 | 독립 시료·팀/시설에서 같은 claim과 조건 확인 | 독립 provenance | **실험실 증거** 필수 gate |

새 분야를 추가할 때 `general_materials`에 자유 문자열 도구를 붙이는 방식으로 확장하지 않습니다. 후보 schema, 속성 정의, 허용 operation, 필수 계산·실험 gate와 stop policy를 먼저 버전 관리한 뒤 정적 registry에 어댑터를 등록해야 합니다.

## 독립 재현의 최소 기준

독립 재현은 단순한 재실행 횟수가 아닙니다.

1. 후보 버전과 핵심 protocol/판정 기준을 실험 전에 고정합니다.
2. 별도 합성 또는 제조 batch를 사용합니다.
3. 가능한 한 최초 팀과 분리된 연구자·시설에서 측정합니다. 같은 시설뿐이라면 독립성 한계를 명시합니다.
4. 원자료, calibration, 실패한 반복과 sample exclusion을 함께 보존합니다.
5. 단위·조건·불확실성을 맞춘 뒤 사전 정의 threshold로 판정합니다.
6. 상충 결과는 평균으로 숨기지 않고 `inconclusive`로 유지하며 원인을 조사합니다.

따라서 계산 코드·seed·모델을 바꾼 반복은 G3의 계산 강건성을 높일 수는 있어도 G5 독립 실험 재현을 충족하지 않습니다.

## MVP가 의도적으로 하지 않는 것

- 모든 가능한 검증기법이 구현되었다고 주장하지 않습니다.
- 계산 `Tc`, 결합 affinity, 독성, 안정성, 전압, 활성, `Tg` 등을 실험값으로 표시하지 않습니다.
- 임의 셸·Python·URL·파일 경로·실험 장비 명령을 모델 출력에서 실행하지 않습니다.
- 외부 DB 미검색을 절대 신규성 또는 세계 최초 발견으로 바꾸지 않습니다.
- 실험 provenance가 없는 JSON을 `experimental`로 신뢰하지 않습니다.
- 사람의 위험 검토, 기관 승인, 비임상·임상·규제 절차를 자동화하거나 대체하지 않습니다.
