# Adaptation under Hardware-Frozen Base

## MCM-Synthesized SR Backbone with Constrained LoRA Side-Path

**Document version:** v8.1 (review-hardened, pre-implementation freeze)
**Status:** ACTIVE paper plan — v5 is archived as future-work seed
**Last updated:** 2026-05-02

---

## 0. One-line Thesis

> **We study the trade-off between adaptation capacity and multiplier budget when the SR backbone is hardware-frozen by MCM synthesis.**

본 논문은 _"MCM이 INT4보다 항상 좋다"_ 또는 _"multiplier-free network가 좋다"_ 를 주장하지 않는다. 대신 다음의 **구체적이지만 실용적인 silicon setting** 을 다룬다:

- SESR-class CNN SR backbone이 MCM logic 으로 합성되면, base weight는 deployment-time constant 가 된다.
- 이 _immutability constraint_ 하에서 sensor adaptation은 별도의 LoRA side-path로만 가능하다.
- 핵심 질문은 **"이 side-path를 어디까지 (INT4 / PoT / Ternary) 제약할 수 있는가, 그리고 image fidelity 가 언제 무너지는가"** 이다.

Motivation 은 두 층으로 분리한다:

1. **Greenfield efficiency question**: MCM SESR 이 INT4 SESR 대비 area / energy / latency / SRAM bandwidth 중 어느 조건에서 우위인가?
2. **Given-MCM deployment question**: 이미 MCM SR accelerator 가 SoC 에 들어간 상황에서, 새 sensor / domain 을 base 재합성 없이 어떻게 adaptation 할 것인가?

따라서 INT4 baseline 이 특정 node / utilization / throughput target 에서 이기더라도 논문의 핵심은 사라지지 않는다. 그 경우 paper 의 framing 은 "MCM beats INT4" 가 아니라 **"hardware-frozen MCM SoC 에 남은 유일한 adaptation channel 의 capacity/budget boundary"** 로 좁힌다.

---

## 1. Positioning Against Prior Work

### 1.1 Closest Prior Work (직접 차별화 필요)

세 개의 prior work 가 우리 논문의 framing 에 결정적이다.

#### **MCM-SR** [Lee et al., IEEE Access 2024]

- CNN-based SR streaming accelerator
- Multiple Constant Multiplication 으로 weight 합성, multiplier 제거
- 23% logic area 감소, 15% SRAM 감소
- **Implicit assumption**: "weights are fixed"
- **우리 논문의 출발점**: _"what if the fact that weights cannot change becomes a constraint, not a benefit?"_
- **차별점**: adaptation 없음 → static deployment 가정

#### **FixyNN** [Whatmough et al., NeurIPS-W 2018; arXiv:1812.01672 / 1902.11128]

- **이 논문이 사실상 우리 framing 의 시조**: fixed front-end feature extractor (FFE) + programmable back-end
- FFE 는 fixed-weight shift-add scaler 로 구현 (= MCM 의 한 형태)
- Transfer learning 으로 back-end 만 task-specific 학습
- 26.6 TOPS/W (4.81× iso-area programmable accelerator 대비)
- **차별점 1**: image classification 만 검증, **SR / low-level vision 검증 없음**
- **차별점 2**: back-end 가 일반 programmable CNN accelerator (MAC array) — **multiplier budget 제약 없음**
- **차별점 3**: **LoRA framework 없음** — back-end 가 task-specific 별도 학습되는 구조 (전체 네트워크 재학습)
- **우리 논문의 발전**: FixyNN 의 fixed front-end 컨셉을 (a) SR domain, (b) **multiplier-budget-constrained adaptation**, (c) **post-deployment LoRA-based adaptation** 으로 확장

#### **AdaptSR** [Korkmaz et al., arXiv:2503.07748, 2025]

- **이 논문이 가장 가까운 LoRA-SR prior work**
- Bicubic-trained SR model 위에 LoRA 로 real-world SR 적응
- Up to 4dB PSNR gain over GAN/diffusion methods
- 92% fewer trainable parameters vs full fine-tuning
- **차별점 1**: **hardware constraint 없음** — base 가 standard FP/INT8 model
- **차별점 2**: LoRA 가 standard FP — **multiplier budget 분석 없음**
- **차별점 3**: post-deployment adaptation 시나리오 없음 (offline merge 가정)
- **우리 논문의 발전**: AdaptSR 의 LoRA-SR concept 위에 **MCM-frozen base** 와 **constrained LoRA quantization** 축을 추가
- **핵심 방어선**: 본 연구는 단순히 "AdaptSR + quantized LoRA" 가 아니다. MCM base 에서는 `W + ΔW` merge 가 불가능하므로 LoRA 가 항상 parallel side-path 로 남고, base 의 quantization / synthesis error 는 adapter 가 직접 보정해야 하는 fixed residual field 가 된다. 이 학습 dynamics 를 Pilot 4 에서 gradient norm, adapter residual spectrum, quantization projection error 로 직접 비교한다.

### 1.2 Visual Positioning

|                 | **Hardware efficiency** | **Adaptation**       | **Multiplier budget on adapter**         |
| --------------- | ----------------------- | -------------------- | ---------------------------------------- |
| MCM-SR          | ✅ multiplier-free base | ❌                   | n/a                                      |
| FixyNN          | ✅ fixed FFE            | ✅ transfer learning | ❌ unconstrained (programmable accel)    |
| AdaptSR         | ❌ standard model       | ✅ LoRA              | ❌ unconstrained (FP LoRA)               |
| QA-LoRA / QLoRA | ✅ INT4 base            | ✅ LoRA              | ⚠️ INT4 (not multiplier-free)            |
| **Ours**        | ✅ MCM base             | ✅ LoRA              | ✅ **PoT/INT4/Ternary, explicit budget** |

→ 본 논문은 위 네 prior work 의 합집합이 아니다. **"hardware-frozen base 위에서 multiplier budget 제약 하의 adaptation 은 어디까지 가능한가"** 라는 새 축을 정의한다.

Reviewer defense 로는 두 실험 축을 본문에 넣는다:

- **AdaptSR-equivalent baseline**: standard FP/INT8 SESR + unconstrained LoRA, merge 가능한 일반 setting.
- **MCM-frozen baseline**: MCM SESR + parallel LoRA, merge 불가능한 hardware-frozen setting.

두 setting 이 같은 rank / dataset 에서 동일하게 행동하면 theory claim 은 낮추고 hardware-constraint trade-off paper 로 유지한다. 다르게 행동하면 "multiplier-free frozen base 위 LoRA 의 learning dynamics" 를 별도 분석 contribution 으로 승격한다.

### 1.3 Supporting Prior Work (참조용)

**Multiplier-free networks** (B7 baseline 으로 사용):

- AdderNet [Chen et al., CVPR 2020]
- DeepShift [Elhoushi et al., CVPR-W 2021] — power-of-two weight 학습. **PoT LoRA 학습성의 직접 근거.**
- ShiftAddNet [You et al., NeurIPS 2020]
- DenseShift [Li et al., ICCV 2023]
- ShiftCNN [Gudovskiy & Rigazio, 2017]

**Backbones** (primary / secondary):

- SESR [Bhardwaj et al., MLSys 2022] — primary backbone (collapsible linear blocks, mobile NPU 친화적)
- QuickSRNet [Berger et al., CVPR-W 2023] — secondary backbone (Qualcomm AI Research, mobile-quant 친화적)

**Quantized SR baselines**:

- Mobile AI 2021/2022/2025 Challenges [Ignatov et al.] — DIV2K INT8 SR baseline reference
- NAWQ-SR [Xu et al., 2022] — hybrid-precision NPU-aware SR

**Quantized LoRA**:

- QLoRA [Dettmers et al., NeurIPS 2023] — INT4 base + FP LoRA (not multiplier-free)
- QA-LoRA [Xu et al., ICLR 2024] — quantization-aware LoRA, **본 논문의 PoT LoRA 학습 방법론 참고**

**Datasets**:

- RealSR [Cai et al., ICCV 2019] — Canon 5D3 + Nikon D810 (2-sensor paired)
- DRealSR [Wei et al., ECCV 2020] — 5 DSLR cameras (**primary multi-sensor dataset**)
- RealSR-RAW [2024] — multi-smartphone with RAW (Plan B candidate)
- DIV2K [Agustsson & Timofte, CVPR-W 2017] — base pretraining

---

## 2. Problem Setting

### 2.1 Background

모바일 / 엣지 SR 가속기는 두 축으로 최적화:

| Axis            | Method                                      | Limitation                                         |
| --------------- | ------------------------------------------- | -------------------------------------------------- |
| Quantization    | INT8 / INT4 (Mobile AI Challenges, NAWQ-SR) | multiplier 잔존 → area / energy bottleneck 유지    |
| Multiplier-free | AdderNet, DeepShift, MCM-SR, FixyNN-FFE     | base 효율적이나 deployment 후 _adaptation 불가능_  |
| LoRA-SR         | AdaptSR, DSCLoRA                            | hardware-aware 하지 않음, multiplier budget 무제약 |

본 논문은 **3축의 교집합** 을 다룬다: multiplier-free base + LoRA adaptation + **explicit multiplier budget on adapter**.

### 2.2 Hardware-Frozen Base 의 의미

MCM (Multiple Constant Multiplication) 은:

- weight를 fixed coefficient set으로 보고
- shift-add network 로 logic synthesis 시점에 pre-compile
- runtime에는 weight를 _데이터로 fetch하지 않음_

결과:

- ✅ runtime multiplier 제거
- ✅ weight memory access 제거
- ❌ **deployment 후 weight 변경 불가능** (재합성 = 새 chip)

이는 FixyNN 의 FFE 와 동일한 hardware contract 이다. 차이는 우리가 **전체 SR backbone** 을 frozen 시키고, adaptation 을 **multiplier-budget-constrained LoRA** 로만 허용한다는 점.

### 2.3 Why This Matters: Sensor Domain Shift

Mobile SoC 위 SR 가속기는 다음과 같은 **post-deployment domain shift** 를 겪는다:

- 같은 base accelerator 가 여러 sensor variant 에 reuse 됨 (RealSR/DRealSR 가 보여주는 inter-sensor PSNR gap)
- Sensor 별로 noise statistic, color response, demosaic artifact 가 다름
- Hardware re-synthesis 는 불가능 (chip 은 이미 tape-out 됨)

### 2.4 Anticipated Reviewer Attack: "Why not multi-domain training?"

> "처음부터 multi-sensor 데이터로 robust base 를 학습하면 LoRA 가 필요 없지 않나?"

방어 (introduction motivation 에 명시):

1. **Performance**: Multi-domain robust training은 averaged representation 을 만들어 sensor-specific 으로는 suboptimal. (RealSR / DRealSR 이 specialized model 로 더 좋은 결과 보임)
2. **Deployment-time unknown**: 새 sensor 가 chip tape-out 후 추가될 수 있음. Training 시점에 알 수 없는 distribution.
3. **Storage / dispatch**: N개 sensor 에 대해 N개 작은 LoRA (각 ~수십 KB) vs 1개 대형 multi-domain model. Edge SoC 에선 전자가 dispatch cost 측면에서 유리.

### 2.5 Greenfield vs Already-Deployed MCM

이 논문의 hardware motivation 은 두 scenario 를 분리해서 쓴다.

| Scenario                         | Core question                                                   | INT4 baseline 이 이기면?                                      |
| -------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------- |
| **Greenfield accelerator design** | 새 SR accelerator 를 만들 때 MCM 이 INT4 보다 나은가?           | 논문 claim 을 area 우위가 아니라 energy / latency / SRAM 으로 좁힘 |
| **Already-deployed MCM SoC**      | 이미 MCM 으로 frozen 된 base 를 새 sensor 에 어떻게 맞출 것인가? | 논문 main motivation 유지. INT4 는 counterfactual baseline 으로 남김 |

이 분리를 introduction 에 명시하면 B4 (`INT4 SESR + INT4 LoRA`) 가 강하게 나와도 paper survival 이 baseline 승패 하나에 묶이지 않는다.

---

## 3. Core Insight

### 3.1 The Real Question

| Prior question                                         | Our question                                                                    |
| ------------------------------------------------------ | ------------------------------------------------------------------------------- |
| "How to fine-tune efficiently?" (LoRA, AdaptSR)        | "How to enable adaptation when base is hardware-frozen?"                        |
| "How to make SR efficient?" (SESR, QuickSRNet, MCM-SR) | "How to retain adaptability while base is multiplier-free?"                     |
| "How to quantize LoRA?" (QLoRA, QA-LoRA)               | "How to constrain LoRA's _multiplier budget_ under fully multiplier-free base?" |

### 3.2 Architectural Decision: Parallel, Not Merged

`y = MCM(W, x) + LoRA(x)` (parallel)

vs

`y = MCM(W + ΔW, x)` (merged) — **rejected**

이유:

- Merged 는 새 sensor마다 MCM 재합성 필요 → silicon 에서 infeasible
- Merged 는 LoRA의 online adaptation 장점 제거
- AdaptSR 가 채택한 merged 방식은 standard FP base 에서만 가능; MCM-frozen base 에서는 parallel 만이 유일한 옵션
- Parallel 만이 hardware-frozen base + post-deployment adaptation 을 동시에 만족

### 3.3 Why Constrain LoRA?

자연스러운 반박: _"LoRA 는 이미 작다 (parameter 1~5%). 왜 더 줄이나?"_

답: **Hardware cost 는 parameter ratio 가 아니라 multiplier cost 기반이다.**

| Path                       | Compute primitive   | Multiplier? |
| -------------------------- | ------------------- | ----------- |
| Base (MAC count 의 90~99%) | shift-add (MCM)     | ❌          |
| LoRA (MAC count 의 1~10%)  | INT8 GEMM (default) | ✅          |

→ MCM 으로 base 의 multiplier 를 0으로 만들면, LoRA path 가 multiplier budget 의 거의 100%를 차지한다.
→ FixyNN 이 back-end 를 programmable accelerator 로 둔 채 두지 않은 이 trade-off 를 우리는 정량화한다.
→ 따라서 LoRA 자체도 multiplier-light (PoT / Ternary) 로 제약해야 한다.

---

## 4. Architecture

### 4.1 Overview

```
RAW / LR Input
      │
      ▼
┌──────────────────────────┐
│  SESR backbone (frozen)  │   ← MCM-synthesized, weight = constant
│   ├─ early conv (frozen) │
│   ├─ mid conv blocks ────┼──→ + LoRA side-path (trainable)
│   └─ late conv (frozen)  │
└──────────────────────────┘
      │
      ▼
   HR Output
```

Per-layer formulation (mid conv blocks):

```
y_l = MCM(W_l, x_l)   +   α_l · B_l · A_l · x_l
       └─ frozen base ┘   └─ LoRA side-path ──┘
       multiplier-free      constrained multipliers
```

### 4.2 Backbone: SESR (primary)

선택 이유:

| Criterion             | Rationale                                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| Inference simplicity  | over-parameterized training이 단일 conv kernel 로 collapse [Bhardwaj et al., MLSys 2022]                   |
| MCM compatibility     | collapsed kernel = fixed coefficient → MCM 직접 합성 (MCM-SR 가 vanilla CNN 에서 보인 효과를 SESR 에 적용) |
| Synergy hypothesis    | collapsed weight 의 distribution 이 MCM shared subexpression 을 증가시킬 _가능성_ (Pilot 3 에서 검증)      |
| Mobile NPU validation | SESR 는 commercial Arm mobile NPU 에서 6×~8× FPS 향상 검증됨                                               |

Secondary backbone (ablation): **QuickSRNet** [Berger et al., CVPR-W 2023] — quantization-robust, residual collapse trick 없이 단순한 구조. 비교를 위함.

### 4.3 LoRA Insertion: Architectural Axis (not a default)

이전 버전에서는 _"1×1 default"_ 로 고정했지만, 이는 검증되지 않은 가정이었다.
최종 버전에서는 **1×1 vs 3×3 LoRA 를 primary architectural trade-off** 로 본다.

| Variant  | Capacity                | Hardware | Tile-friendly          | Risk                                    |
| -------- | ----------------------- | -------- | ---------------------- | --------------------------------------- |
| 1×1 LoRA | channel-only adaptation | 매우 low | ✅ no halo region      | spatial sensor artifact 못 잡을 수 있음 |
| 3×3 LoRA | channel + spatial       | higher   | ⚠️ halo alignment 필요 | budget 증가                             |

**Comparison protocol**: matched **multiplier budget** (not matched rank).

- Primary matching criterion: **side-path per-pixel multiplier count**.
- 1×1 with rank `r₁` vs 3×3 with rank `r₃` where `9 · r₃ ≈ r₁` under the same input/output channel width.
- 이 등식은 **MAC / multiplier-count matching** 이지, area / energy / latency matching 이 아니다.
- Area, energy, latency, SRAM traffic, halo buffering cost 는 별도 secondary metrics 로 보고한다.
- AdaptSR 가 채택한 conv LoRA 는 rank-only 비교 → 우리는 hardware-aware 로 한 단계 더 나아감.

### 4.4 Insertion Position

- **Frozen**: early layers (low-level feature extraction, task-generic), late layers (reconstruction head, task-generic)
- **LoRA-equipped**: mid conv blocks (sensor-specific statistics: noise, color, texture 가 주로 인코딩되는 영역)
- AdaptSR 의 layer-selective ablation 결과를 SESR 에 transfer 하여 검증

### 4.5 Scaling Factor α_l

`α_l` 도 multiplier budget 에 포함되므로:

- Training: float scalar (per-layer)
- Deployment: nearest power-of-two 로 projection → bit-shift 로 구현, multiplier 0개
- 이 design choice 는 PoT LoRA 와 일관성을 유지

### 4.6 LoRA Quantization Hierarchy

기존 _"INT8 / INT4 / PoT / Ternary 동등 비교"_ → 다음과 같이 **계층적 narrative** 로 재정렬:

| Tier        | Variant      | Role in paper                              | Reference                       |
| ----------- | ------------ | ------------------------------------------ | ------------------------------- |
| Upper bound | INT8 LoRA    | unconstrained reference                    | AdaptSR-equivalent              |
| Practical   | INT4 LoRA    | conventional baseline                      | QLoRA-equivalent                |
| **Main**    | **PoT LoRA** | **multiplier-light primary contribution**  | DeepShift principle on adapter  |
| Lower bound | Ternary LoRA | extreme constraint, "phase boundary" probe | Ternary quantization literature |

PoT 가 main 인 이유:

- power-of-two weight 는 shift 로 구현 → multiplier-free side-path 가능
- continuous low-rank space 의 일부를 보존 → ternary 보다 학습 가능성 높음
- INT4 와 multiplier-free 사이의 sweet spot 후보
- DeepShift 가 base network 에서 PoT 학습 성공을 보였으므로, **adapter 에 적용 시 학습성 가설은 합리적** (Pilot 2 에서 검증)

### 4.7 Tile-aware Scheduling

`1×1 LoRA` 의 hardware advantage 를 architecture 절에서 명시:

- Base path 가 tile streaming 으로 동작할 때 (MCM-SR 의 streaming datapath 와 동일 패러다임)
- 1×1 LoRA 는 receptive field = 1 → halo region 0 → **base tile boundary 와 직접 정렬**
- 3×3 LoRA 는 halo alignment 필요 → buffering overhead

→ 1×1 vs 3×3 trade-off 는 **단순한 표현력 비교가 아니라 datapath-aware decision** 이다.

Important caveat: `9 · r₃ ≈ r₁` 은 only first-order multiplier count proxy 이다. 실제 hardware 에서는 3×3 weight reuse, line-buffer reuse, halo-region buffering, tile size, SRAM banking 이 area / energy 를 바꾼다. 본문에서는 "matched multiplier budget" 과 "matched hardware cost" 를 구분해서 쓴다.

---

## 5. Research Questions

| RQ  | Question                                                                                                                                                         |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RQ1 | MCM 합성된 SESR base 의 area / energy / latency 가 INT4 SESR baseline 대비 어떤 조건에서 우위인가? (MCM-SR 의 23% area gain 이 SESR 에 transfer 되는가?)         |
| RQ2 | SESR collapsed weight 가 vanilla / random conv 대비 MCM shared subexpression 을 증가시키는가?                                                                    |
| RQ3 | Constrained LoRA (INT4 / PoT / Ternary) 가 hardware-frozen SESR base 위에서 **학습 가능한가**? (DeepShift 의 PoT 학습성이 adapter 에 transfer 되는가?)           |
| RQ4 | Sensor-domain shift 에서 constrained LoRA 가 회복하는 fidelity 는 unconstrained (INT8) LoRA / AdaptSR 대비 어느 수준인가?                                        |
| RQ5 | 동일 multiplier budget 하에서 1×1 high-rank vs 3×3 low-rank LoRA 중 어느 것이 sensor adaptation 에 유리한가?                                                     |
| RQ6 | Compute dominance (base vs side-path) 를 MAC count / area / energy / latency **네 metric 으로 분리** 측정했을 때, 각 metric 에서 dominance 비율은 어떻게 다른가? |
| RQ7 | MCM-frozen parallel LoRA 의 학습 dynamics 는 standard FP/INT8 base 위의 mergeable AdaptSR/QLoRA-style LoRA 와 어떻게 다른가?                                    |

---

## 6. Pilot Studies (Pre-Implementation)

본 실험 전에 반드시 통과해야 하는 검증 단계.
**순서가 중요** — paper survival 을 위협하는 failure mode 부터 제거한다.

### 6.1 Pilot 0: Dataset Feasibility (1~2 days)

**Goal**: Sensor adaptation evaluation 에 사용할 dataset 확보 가능성 확인.

**Primary candidates**:

- **DRealSR** [Wei et al., ECCV 2020] — 5 DSLR cameras, ~31,970 patches, 가장 강력한 multi-sensor SR dataset
- **RealSR** [Cai et al., ICCV 2019] — Canon 5D3 + Nikon D810, 595 paired (2-sensor)
- **RealSR-RAW** [2024] — multi-smartphone with RAW data

**Plan B (이미 구체화됨)**:

- Plan B-1: dataset domain shift (DIV2K → RealSR → BSDS) 시나리오로 후퇴
- Plan B-2: Degradation adaptation (bicubic → realistic blur+noise)
- Plan B-3: Synthetic-only with strong physical model
  - Heteroscedastic Gaussian noise (signal-dependent shot noise + read noise)
  - CCM perturbation grounded in DNG metadata
  - Tone curve / gamma variation

### 6.2 Pilot 1: MCM vs INT4 Hardware Viability (paper-and-pencil, 1~2 days, **CRITICAL**)

**Goal**: Motivation 의 정직성 확보. PoT LoRA 가 학습 가능해도 MCM hardware premise 가 숫자로 버티지 못하면 greenfield thesis 가 흔들리므로, 이 pilot 을 trainability 보다 먼저 수행한다.

**Action**:

- INT4 SESR base: 4-bit multiplier / MAC array area, energy, throughput, utilization proxy 산출.
- INT4 SESR + INT4 LoRA (B4): adapter 까지 포함한 multiplier-rich counterfactual 산출.
- MCM SESR base: collapsed weight 분포 → adder count / shared subexpression count → area estimate.
- MCM SESR + constrained LoRA: LoRA path 의 INT4 / PoT / Ternary 비용을 별도로 산출.
- SRAM bandwidth: base weight fetch 제거 효과와 LoRA weight fetch 비용을 분리.
- Throughput target: 1080p@60fps 및 4K@30fps 에서 각각 계산.

**Pre-validated framing decision** (결과 본 뒤 끼워 맞추지 않기):

| Hardware outcome                                 | Paper framing                                                                                   |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| MCM < INT4 in area and at least one runtime metric | Greenfield + given-MCM framing 모두 유지                                                        |
| MCM area 비슷 / runtime metric 우위               | Area claim 약화, **energy / latency / SRAM bandwidth** 중심으로 이동                            |
| INT4 가 area/runtime 대부분 우위                  | Greenfield efficiency claim 제거. **already-deployed MCM SoC 의 adaptation problem** 으로 좁힘 |

이 pilot 의 역할은 "MCM 이 INT4 를 반드시 이긴다" 를 증명하는 것이 아니라, 어떤 claim boundary 가 정직한지 먼저 정하는 것이다.

### 6.3 Pilot 2: PoT LoRA Trainability (이번 주, **CRITICAL**)

**Goal**: 논문 survival condition 검증.

**Setup**:

- Backbone: **SESR 자체** (ESPCN 같은 toy model 우회 — transfer risk 제거)
- Base: pretrained on DIV2K, frozen
- LoRA variants: FP, INT4, PoT, Ternary (DeepShift-Q / DeepShift-PS 학습 방법론 참고)
- Eval: PSNR/SSIM on RealSR
- Training: QA-LoRA 의 group-wise operator 트릭 적용 검토

**Pre-defined success thresholds** (confirmation bias 방지):

| PoT retention vs INT8 | Decision                                                     |
| --------------------- | ------------------------------------------------------------ |
| ≥ 95%                 | PoT 를 main contribution 으로 진행                           |
| 85 ~ 95%              | PoT vs INT4 dual main, trade-off framing 강화                |
| < 85%                 | **Reframe required** — PoT 는 lower bound, INT4 가 main 으로 |

Threshold 는 pilot 시작 _전에_ 정해놓아야 함.

### 6.4 Pilot 2.5: Sensor Adaptation Capacity (1 week max, **CRITICAL**)

**Goal**: PoT LoRA 가 loss 를 줄일 수 있다는 사실과, 실제 DRealSR/RealSR sensor adaptation 에서 의미 있는 PSNR gain 을 낸다는 사실은 다르다. Pilot 2 통과 직후 짧게 capacity 를 검증한다.

**Setup**:

- Base: DIV2K-pretrained SESR → MCM-frozen proxy.
- Source / target: DRealSR sensor split 우선. 최소 fallback 은 RealSR Canon 5D3 ↔ Nikon D810.
- Variants: MCM base only, MCM + INT8 LoRA, MCM + INT4 LoRA, MCM + PoT LoRA.
- Duration: 1 week 안에 끝나는 small-rank / small-epoch probe.

**Metrics**:

```
Δ_INT8 = PSNR(MCM + INT8 LoRA) - PSNR(MCM base)
Δ_PoT  = PSNR(MCM + PoT  LoRA) - PSNR(MCM base)
retention = Δ_PoT / Δ_INT8
```

**Pre-defined thresholds**:

| Outcome                                      | Decision                                                                 |
| -------------------------------------------- | ------------------------------------------------------------------------ |
| `Δ_PoT ≥ 0.3 dB` and `retention ≥ 70%`        | PoT LoRA remains primary contribution                                    |
| `Δ_PoT ≥ 0.3 dB` and `50% ≤ retention < 70%` | PoT vs INT4 trade-off paper; PoT becomes aggressive operating point      |
| `Δ_INT8 < 0.3 dB`                            | Dataset / adaptation protocol not strong enough; revisit sensor split    |
| `Δ_PoT < 0.3 dB` or `retention < 50%`         | INT4 main, PoT lower-bound / phase-boundary probe                        |

### 6.5 Pilot 3: SESR Collapse × MCM Synergy (1 day)

**Goal**: SESR 선택의 secondary contribution 검증.

**Experiment**:

- SESR collapsed weight, vanilla 3×3 conv weight, random weight 100개씩 샘플
- 각각을 MCM synthesis tool 로 합성 (MCM-SR 의 graph exploration 알고리즘 참고)
- Adder count, shared subexpression count 분포 비교

**Decision tree**:

- SESR collapsed > vanilla > random → **secondary contribution** 으로 강조
- SESR ≈ vanilla → backbone 선택 근거를 "inference simplicity + MCM compatibility" 만으로 축소
- SESR < vanilla → backbone 선택 재검토

### 6.6 Pilot 4: AdaptSR / QLoRA Differentiation Mini-Experiment (2~3 days)

**Goal**: Reviewer attack _"AdaptSR + QLoRA-style quantization 아닌가?"_ 에 대한 실험적 방어.

**Matched setup**:

- Standard setting: FP/INT8 SESR + LoRA, merge 가능 (`W + ΔW`).
- Frozen setting: MCM SESR proxy + parallel LoRA, merge 불가능 (`MCM(W, x) + LoRA(x)`).
- Same rank, same data, same target sensor, same training budget.

**Report**:

- LoRA gradient norm and variance across layers.
- Gradient cosine similarity between standard and frozen settings.
- Adapter residual spectrum: `LoRA(x)` 가 base quantization / synthesis residual 의 어느 frequency band 를 보정하는가?
- Quantization projection error: FP LoRA → INT4 / PoT projection 시 fidelity drop.

**Decision**:

- Dynamics 차이가 명확하면 RQ7 / discussion 에 반영.
- 차이가 약하면 "learning dynamics" claim 은 낮추고, paper 는 explicit hardware contract + cost/fidelity boundary 로 유지.

---

## 7. Main Experiments

### 7.1 Baseline Suite (반드시 모두 포함)

| #      | Baseline                                                     | Purpose                               | Reference                        |
| ------ | ------------------------------------------------------------ | ------------------------------------- | -------------------------------- |
| B1     | FP32 SESR                                                    | accuracy ceiling                      | Bhardwaj et al., MLSys 2022      |
| B2     | INT8 SESR                                                    | conventional accuracy reference       | Mobile AI Challenges             |
| B3     | INT4 SESR                                                    | "그냥 INT4 로 다 하면?" reviewer 방어 | NAWQ-SR style                    |
| B4     | **INT4 SESR + INT4 LoRA**                                    | **critical baseline**                 | QLoRA principle on SR            |
| B5     | MCM SESR (no LoRA)                                           | no-adaptation lower bound             | MCM-SR equivalent                |
| B6     | MCM SESR + INT8 LoRA                                         | unconstrained side-path baseline      | AdaptSR equivalent               |
| B7a    | DeepShift SESR + INT8 LoRA                                   | generality (PoT base)                 | Elhoushi et al., CVPR-W 2021     |
| B7b    | AdderNet SESR + INT8 LoRA                                    | generality (adder base)               | Chen et al., CVPR 2020           |
| B8     | FixyNN-style SESR (FFE + programmable back-end full retrain) | classical hardware-frozen approach    | Whatmough et al., NeurIPS-W 2018 |
| **P1** | **MCM SESR + PoT LoRA**                                      | **primary proposal**                  | —                                |
| P2     | MCM SESR + INT4 LoRA                                         | intermediate constraint               | —                                |
| P3     | MCM SESR + Ternary LoRA                                      | extreme lower bound                   | —                                |

B7a/b 가 중요 — 안 넣으면 _"MCM-specific 논문"_ 으로 보이고 generality 가 약해짐.
B8 (FixyNN) 추가 — _"왜 LoRA framework 가 필요한가"_ 의 직접 비교. FixyNN 은 back-end 전체 재학습; 우리는 small LoRA 로 충분함을 보임.
B4 는 가장 위험한 baseline 이므로 별도 강조한다. B4 가 proposal variants `P1/P2` 를 image quality 또는 area 에서 이기면 결과를 숨기지 않고, greenfield accelerator claim 을 낮춘 뒤 **given-MCM SoC adaptation** 과 SRAM/latency/energy trade-off 중심으로 재해석한다.

### 7.2 Compute Dominance Reporting (RQ6)

각 configuration 에 대해 **네 가지 metric 을 분리 측정**:

| Metric          | Base path | LoRA path | Notes                            |
| --------------- | --------- | --------- | -------------------------------- |
| MAC count ratio | %         | %         | 가장 단순                        |
| Area ratio      | %         | %         | gate count proxy                 |
| Energy ratio    | %         | %         | switching activity × capacitance |
| Latency ratio   | %         | %         | critical path                    |

핵심 finding 형식:

> "MCM SESR + PoT LoRA 에서 base path 는 MAC count 의 X% 를 차지하지만 area 의 Y% 만 차지하며, 이 비대칭이 PoT constraint 의 motivation 을 정량적으로 뒷받침한다."

### 7.3 Image Quality Suite

기본 SR metric:

- PSNR / SSIM / LPIPS

ISP-grade fidelity (SR output 에 적용):

- ΔE (color accuracy)
- Banding artifact rate
- Low-light region PSNR
- HDR clipping rate

### 7.4 Throughput Validation

Target: **1080p@60fps 또는 4K@30fps**

- Cycles per pixel
- SRAM bandwidth (line buffer + LoRA weight)
- Latency per frame
- Tile-based scheduling 시 base/LoRA path sync overhead
- QuickSRNet 의 2.2ms@1080p×2 baseline 대비 비교

### 7.5 Adaptation Evaluation Protocol

**Three settings** (논문 본문 명시):

| Setting                 | Description                                                  | Role                             |
| ----------------------- | ------------------------------------------------------------ | -------------------------------- |
| (A) Paired adaptation   | DRealSR sensor-pair source / target paired data 로 LoRA 학습 | **primary**, cleanest comparison |
| (B) Zero-shot transfer  | no LoRA, base 만 evaluation                                  | no-adaptation lower bound        |
| (C) Few-shot adaptation | target 의 limited paired data                                | practicality probe               |

**Synthetic shift physical grounding** (반드시 명시):

- Random noise / color jitter ❌
- Heteroscedastic noise model + CCM perturbation + tone curve ✅

### 7.6 Ablations

| Ablation                | Variants                               |
| ----------------------- | -------------------------------------- |
| LoRA spatial size       | 1×1 vs 3×3 (primary: matched side-path multiplier count; secondary: area / energy / SRAM / halo cost) |
| LoRA insertion position | early / mid / late / all               |
| LoRA rank               | r ∈ {2, 4, 8, 16}                      |
| α scaling               | FP vs PoT                              |
| Backbone                | SESR vs QuickSRNet                     |
| Multi-domain training   | as alternative reference (RQ defense)  |

---

## 8. Contributions (Final Form)

1. **New problem setting**: hardware-frozen base 위의 adaptation 문제 정식화. MCM 의 deployment-time immutability 가 LoRA 를 _"the only viable adaptation channel"_ 로 만드는 framing 을 명시. **FixyNN 의 fixed front-end 컨셉을 SR + adaptation domain 으로 확장**.

2. **Constrained LoRA framework**: multiplier budget 하의 LoRA 설계 공간 (INT4 / PoT / Ternary) 을 _trade-off study_ 로 정의. **AdaptSR / QLoRA / QA-LoRA 가 다루지 않은 multiplier-free side-path** 의 학습성과 fidelity boundary 를 정량화. 필요 시 RQ7 분석으로 MCM-frozen parallel LoRA 와 mergeable LoRA 의 gradient / residual dynamics 차이를 보고.

3. **1×1 vs 3×3 LoRA under matched multiplier budget**: hardware-aware adaptation capacity comparison. 여기서 matched budget 은 primary metric 으로 **side-path multiplier count** 를 의미하며, area / energy / latency / SRAM 은 secondary hardware metrics 로 따로 보고한다. Tile-based streaming 과의 정합성까지 고려한 datapath-aware decision.

4. **Compute dominance asymmetry analysis**: MAC count / area / energy / latency 네 metric 의 분리 측정을 통해 _"작은 LoRA 가 hardware 관점에선 dominant 일 수 있다"_ 를 정량화.

5. **(Conditional) SESR collapse × MCM synergy**: collapsed weight distribution 이 MCM shared subexpression 을 증가시킨다는 가설 — Pilot 3 결과에 따라 강조 강도 조정.

---

## 9. Assumptions and Limitations

### 9.1 Assumptions (introduction 에 명시 — 결과의 전제 조건)

- A1: PoT / low-bit LoRA 가 sensor adaptation 에 충분한 capacity 를 보유한다 (Pilot 2 및 Pilot 2.5 에서 검증, DeepShift 의 PoT base 학습성이 학습 가능성의 prior 근거).
- A2: MCM synthesis 가 SESR-class 모델의 weight distribution 에서 INT4 baseline 대비 area 또는 energy / latency / SRAM bandwidth 우위를 가질 수 있다 (Pilot 1 에서 검증, MCM-SR 의 23% area / 15% SRAM gain 이 출발점). 단, 이 assumption 이 약하면 main framing 은 already-deployed MCM SoC 로 이동한다.
- A3: Sensor-domain shift 가 mid conv layer 의 LoRA 로 충분히 표현 가능하다 (AdaptSR 의 layer-selective 결과 transfer).
- A4: PoT LoRA 의 trainability 는 실제 sensor adaptation gain 으로 이어진다 (Pilot 2.5 에서 별도 검증).

### 9.2 Limitations (결론 절 — 결과의 적용 범위)

- L1: 결과는 SESR-class compact SR 에 한정. 더 크고 복잡한 SR 모델 (Transformer-based, diffusion-based) 로의 일반화는 future work.
- L2: 평가는 2~5 sensor variant 수준 (RealSR / DRealSR scale). 수십 개 sensor scale-up 시 LoRA 저장 오버헤드 분석은 수행하지 않음.
- L3: Synthetic shift 는 physically grounded 이지만 real sensor 의 모든 idiosyncrasy 를 포착하진 않음.

---

## 10. Risk Register

| Risk                                                | Severity     | Mitigation                                                                                                                                                                             |
| --------------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B4 (`INT4 SESR + INT4 LoRA`) 가 proposal `P1/P2` 보다 강함 | **Critical** | Pilot 1 을 Pilot 2 보다 먼저 수행. Greenfield efficiency claim 은 결과 의존으로 두고, 패배 시 **already-deployed MCM SoC adaptation** 으로 main framing 이동.                         |
| MCM 이 INT4 대비 area 우위 없음                     | **Critical** | Area / energy / latency / SRAM bandwidth 를 모두 paper-and-pencil 로 선계산. 결과 보기 전에 fallback metric 이 실제로 가능한지 확인. 대부분 패배하면 greenfield claim 제거.             |
| PoT LoRA 학습 자체가 실패                           | **Critical** | Pilot 2 사전 수행, threshold 사전 정의, INT4 main 으로 reframe 준비. DeepShift 가 base 에서 성공한 prior 가 있어 선험적 가능성은 합리적.                                               |
| PoT LoRA 는 학습되지만 sensor adaptation gain 이 작음 | **High**     | Pilot 2.5 추가. `Δ_PoT ≥ 0.3 dB` 및 INT8 gain retention threshold 사전 정의. 실패 시 PoT 는 lower bound, INT4 를 main operating point 로 이동.                                         |
| Multi-sensor SR dataset 부재                        | **High**     | Plan B-1/2/3 사전 정의. **DRealSR 5-camera 가 primary 로 거의 확정** (이미 공개 dataset).                                                                                              |
| 1×1 vs 3×3 matched budget 정의가 불명확             | Medium       | Primary matching 은 side-path multiplier count 로 명시. Area / energy / latency / SRAM / halo buffering 은 별도 secondary metrics 로 보고.                                             |
| 1×1 LoRA 가 sensor spatial artifact 못 잡음         | Medium       | 3×3 LoRA 를 architectural axis 로 승격 (이미 반영). 3×3 low-rank 가 이기면 "tile-cost vs spatial-capacity" trade-off 로 해석.                                                          |
| Reviewer: "AdaptSR 와 차별점 불명확"                | Medium       | Multiplier budget axis + MCM-frozen non-mergeable LoRA 를 명시. Pilot 4 로 mergeable LoRA vs parallel LoRA 의 gradient / residual dynamics 를 비교.                                   |
| Reviewer: "FixyNN 와 차별점 불명확"                 | Medium       | FixyNN 은 back-end 전체 재학습 + classification only. 우리는 SR + LoRA + budget 제약. B8 baseline 으로 직접 비교                                                                       |
| Reviewer: "multi-domain training 으로 충분"         | Medium       | 3-axis 방어 (performance / unknown sensor / storage) introduction 명시                                                                                                                 |
| SESR collapse 가 MCM synergy 없음                   | Low          | Secondary contribution 강도 조정만 필요, main thesis 영향 없음                                                                                                                         |

---

## 11. Execution Plan

### Week 1

- **Day 1~2**: Pilot 0 (dataset feasibility survey — DRealSR / RealSR / RealSR-RAW 우선순위 결정)
- **Day 1~2 병렬**: Pilot 1 (MCM vs INT4 paper-and-pencil area / energy / latency / SRAM estimate)
- **Day 3**: Pilot 1 outcome 으로 greenfield vs given-MCM framing boundary 확정
- **Day 4~5**: Pilot 2 코드 작성 (SESR + PoT/INT4/Ternary LoRA, DeepShift / QA-LoRA 학습 방법론 참고)

### Week 2

- **Day 1~2**: Pilot 2 실행 + trainability threshold 비교 → PoT / INT4 main framing 결정
- **Day 3~5**: Pilot 2.5 sensor adaptation capacity probe (`Δ_PoT`, INT8 retention)
- **Day 4~5 병렬**: Pilot 3 (SESR collapse × MCM adder count) + Pilot 4 (AdaptSR / QLoRA differentiation mini-experiment)

### Week 3

- Pilot 결과 종합 → architecture 최종 확정
- Abstract + Figure 1 작성
- 본 실험 코드 framework 시작

### Week 4 onwards

- Main experiments (baseline suite + ablations)
- Paper writing in parallel

---

## 12. Out of Scope (Explicitly)

논문 v1 ~ v4 에서 한때 포함되었으나 **본 논문에서는 제외**:

- ❌ DyT (Dynamic Tanh) — ISP / SR 검증 부족, 별도 논문 가치
- ❌ Looped Transformer — latency budget 과 충돌
- ❌ Transformer-based SR backbone — MCM 적합성 낮음 (SwinIR, ESRT 등)
- ❌ Diffusion-based SR adaptation (S3Diff 등) — base 자체가 거대 multiplier-heavy
- ❌ Joint demosaic + denoise task — 평가 범위 비대화

이들은 **future work** 으로 후속 논문에서 다룬다.

### 12.1 Archived v5 Ideas Worth Preserving

`RecurISP_v5_proposal_1.md` 는 삭제하지 않고 후속 연구 seed 로 보존한다. v8 본문에는 넣지 않지만, 다음 아이디어는 later-cycle discussion 또는 별도 ISP paper 로 살릴 가치가 있다.

| v5 idea                                      | v8 decision                          | Future-work use                                                  |
| -------------------------------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| Iteration-conditioned LoRA specialization     | 본 SR paper 에서는 제외              | ISP 에서 denoise → texture → color → tone role specialization 검증 |
| DyT-HDR for RAW/HDR                           | 본 SR paper 에서는 제외              | RAW→sRGB / HDR ISP 에서 별도 normalization-free block paper 후보 |
| Cycle-level latency / energy measurement      | v8 에서는 throughput validation 중심 | ISSCC/JSSC extension 을 노릴 때 hardware measurement ambition 흡수 |

---

## 13. The One Sentence

> _"Once an SESR backbone is synthesized into multiplier-free MCM logic, its weights become hardware-frozen — and the question is no longer how to make the model efficient, but how much we can constrain its only remaining adaptation channel before image fidelity collapses."_

---

## Appendix A: Decision History

| Version       | Key change                                                                                                                      | Trigger                      |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| v1            | DyT + MCM + Loop + LoRA on Transformer ISP                                                                                      | Initial proposal             |
| v2            | "Multiplierless" → "dominant path multiplier-free + constrained side"                                                           | LoRA × MCM contradiction     |
| v3            | Transformer → SESR/QuickSRNet CNN SR                                                                                            | DyT/Loop validation risk     |
| v4            | Reframed as "adaptation under hardware-frozen base"                                                                             | LoRA 의 진짜 역할 재정의     |
| v5            | Honest about MCM not always beating INT4                                                                                        | Reviewer defense             |
| v6            | 1×1 vs 3×3 LoRA → architectural axis, PoT pilot priority                                                                        | Single-point-of-failure 봉쇄 |
| v7            | Pilot threshold 사전 정의, dataset Plan B, 3-axis multi-domain defense, B7 generality baseline, assumptions vs limitations 분리 | Pre-execution freeze         |
| v8            | Literature integration: MCM-SR / FixyNN / AdaptSR / DeepShift / QA-LoRA / DRealSR 정확한 차별화                                 | Reviewer-ready positioning   |
| **v8.1 (this)** | Pilot 1 을 hardware viability 로 재정렬, Pilot 2.5 / Pilot 4 추가, B4 fallback 과 given-MCM framing 강화                      | Claude review 반영           |

---

## Appendix B: Quick Decision Reference

질문이 생기면 이 표를 먼저 본다:

| Question                | Answer                                                                            |
| ----------------------- | --------------------------------------------------------------------------------- |
| LoRA 형태?              | 1×1 vs 3×3 둘 다 (matched multiplier budget)                                      |
| LoRA 위치?              | mid conv blocks only                                                              |
| LoRA quantization main? | PoT (Pilot 2 / 2.5 결과 따라 INT4 가능성 열어둠)                                  |
| α scaling?              | learned FP, deploy as PoT                                                         |
| Backbone?               | SESR primary, QuickSRNet ablation                                                 |
| Task?                   | Super-resolution (×2, ×4)                                                         |
| Adaptation scenario?    | Sensor domain shift on DRealSR (paired primary)                                   |
| Throughput target?      | 1080p@60fps 또는 4K@30fps                                                         |
| Critical baseline?      | INT4 SESR + INT4 LoRA + AdaptSR-equivalent + FixyNN-equivalent                    |
| First pilot after dataset? | MCM vs INT4 hardware viability (area / energy / latency / SRAM), before PoT trainability |
| Fallback framing?       | If INT4 wins greenfield comparison, focus on already-deployed MCM SoC adaptation   |
| Main contribution?      | Constrained LoRA under MCM-frozen base, with compute dominance asymmetry analysis |

---

## Appendix C: Reference List (Categorized)

### C.1 Direct prior work — must-cite, must-differentiate

1. **MCM-SR** — Lee et al., "MCM-SR: Multiple Constant Multiplication-Based CNN Streaming Hardware Architecture for Super-Resolution," _IEEE Access_, 2024. [DOI: 10.1109/ACCESS.2024 / IEEE Xplore 10777852]
2. **FixyNN** — Whatmough et al., "FixyNN: Efficient Hardware for Mobile Computer Vision via Transfer Learning," arXiv:1902.11128, 2019. (Workshop version: NeurIPS On-Device ML 2018, arXiv:1812.01672)
3. **AdaptSR** — Korkmaz, Mehta, Timofte, "AdaptSR: Low-Rank Adaptation for Efficient and Scalable Real-World Super-Resolution," arXiv:2503.07748, 2025.

### C.2 Backbones

4. **SESR** — Bhardwaj et al., "Collapsible Linear Blocks for Super-Efficient Super Resolution," _MLSys 2022_. arXiv:2103.09404.
5. **QuickSRNet** — Berger et al., "QuickSRNet: Plain Single-Image Super-Resolution Architecture for Faster Inference on Mobile Platforms," _CVPR Workshops 2023_. arXiv:2303.04336.

### C.3 Multiplier-free networks (B7 baselines)

6. **DeepShift** — Elhoushi et al., "DeepShift: Towards Multiplication-Less Neural Networks," _CVPR Workshops 2021_. arXiv:1905.13298.
7. **AdderNet** — Chen et al., "AdderNet: Do We Really Need Multiplications in Deep Learning?" _CVPR 2020_.
8. **AdderNet (HW)** — Wang et al., "AdderNet and Its Minimalist Hardware Design for Energy-Efficient Artificial Intelligence," arXiv:2101.10015, 2021.
9. **ShiftAddNet** — You et al., "ShiftAddNet: A Hardware-Inspired Deep Network," _NeurIPS 2020_. arXiv:2010.12785.
10. **DenseShift** — "DenseShift: Towards Accurate and Efficient Low-Bit Power-of-Two Quantization," arXiv:2208.09708, 2022.
11. **ShiftCNN** — Gudovskiy & Rigazio, "ShiftCNN: Generalized Low-Precision Architecture for Inference of Convolutional Neural Networks," arXiv:1706.02393, 2017.
12. **ShiftAddNAS** — "ShiftAddNAS: Hardware-Inspired Search for More Accurate and Efficient Neural Networks," arXiv:2205.08119, 2022.

### C.4 Quantized LoRA (LLM domain, methodology reference)

13. **QLoRA** — Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," _NeurIPS 2023_.
14. **QA-LoRA** — Xu et al., "QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models," _ICLR 2024_. arXiv:2309.14717.
15. **LoRA** — Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," _ICLR 2022_.

### C.5 SR + LoRA related (very recent)

16. **DSCLoRA** — "Distillation-Supervised Convolutional Low-Rank Adaptation for Efficient Image Super-Resolution," arXiv:2504.11271, 2025.
17. **S3Diff** — "Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors," arXiv:2409.17058, 2024.

### C.6 Quantized SR on mobile NPUs (B2/B3 baselines)

18. **Mobile AI 2021 Challenge** — Ignatov et al., "Real-Time Quantized Image Super-Resolution on Mobile NPUs," arXiv:2105.07825, 2021.
19. **Mobile AI 2022 Challenge** — Ignatov et al., "Efficient and Accurate Quantized Image Super-Resolution on Mobile NPUs," arXiv:2211.05910, 2022.
20. **NAWQ-SR** — Xu et al., "NAWQ-SR: A Hybrid-Precision NPU Engine for Efficient On-Device Super-Resolution," arXiv:2212.09501, 2022.

### C.7 Datasets

21. **DIV2K** — Agustsson & Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study," _CVPR Workshops 2017_.
22. **RealSR** — Cai et al., "Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model," _ICCV 2019_. arXiv:1904.00523.
23. **DRealSR** — Wei et al., "Component Divide-and-Conquer for Real-World Image Super-Resolution," _ECCV 2020_. arXiv:2008.01928.
24. **RealSR-RAW** — "Unveiling Hidden Details: A RAW Data-Enhanced Paradigm for Real-World Super-Resolution," arXiv:2411.10798, 2024.
