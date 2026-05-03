# RecurISP: Iteration-Conditioned Low-Rank Adaptation for Recurrent Neural Image Signal Processing

> **Version 5.0 — Related Work integrated (MCM-SR / FixyNN / ROMA 계열 prior art 반영)**

> **Archive status.** 이 문서는 **archived ambition / future-work seed** 이다. 현재 active paper plan 은 `paper_plan_v8_final.md` v8.1 이며, v5 의 Looped Transformer / DyT-HDR / early-exit / iteration-conditioned LoRA 는 이번 SR paper 의 scope 에 포함하지 않는다.

> **One-line definition.** RecurISP는 **MCM-realizable frozen base** 위에 **iteration-conditioned LoRA side-path**를 결합하여, mobile NPU의 hardware immutability 제약 아래에서도 sensor·ISO·조도 조건에 따라 적응 가능한 multiplier-efficient neural ISP 구조이다.

> **Primary claim.** Iteration-conditioned low-rank updates를 통한 **expressive capacity expansion under weight sharing**.
> **Secondary claim (optional).** Per-iteration LoRA가 자연스러운 **role specialization** (denoise → texture → color → tone)을 유도할 수 있음.
> **Hardware claim.** Frozen base는 MCM/shift-add로 합성 가능하고, adaptation은 quantization-constrained LoRA side-path로 분리하여 **"efficient + adaptable under hardware freeze"** 를 달성.

---

## Archive Note — Relation to v8.1

v5 는 "좋은 아이디어를 모두 넣은" ISP-oriented ambitious version 이고, v8.1 은 그중 출판 가능성이 높은 핵심 축만 남긴 SR-oriented executable plan 이다.

| Item                         | v5 status                    | v8.1 decision                                      |
| ---------------------------- | ---------------------------- | -------------------------------------------------- |
| Iteration-conditioned LoRA    | Primary claim                | 이번 paper 에서는 제외, ISP 후속 연구 seed 로 보존 |
| DyT-HDR                       | Core block                   | SR paper 에서는 제외, RAW/HDR ISP paper 후보       |
| Adaptive early-exit           | Compute-scaling contribution | 이번 paper 에서는 제외                             |
| Cycle-level HW comparison     | Main hardware ambition       | v8.1 에서는 throughput / cost validation 중심      |
| MCM-frozen base + constrained LoRA | Shared core idea             | v8.1 의 active thesis 로 승격                      |

이 문서는 삭제하지 않는다. 다만 implementation / pilot / submission 기준 문서로는 사용하지 않고, v8.1 이후 ISP 또는 hardware-extended paper 를 설계할 때 참고한다.

---

## 0. Executive Summary — Two-axis Positioning

RecurISP는 두 축의 prior work 사이에 **유일하게 비어있는 사분면**을 차지한다.

|                                        | **Static (no adaptation)**                | **Adaptive (post-deployment)**             |
| -------------------------------------- | ----------------------------------------- | ------------------------------------------ |
| **Multiplier-rich**                    | PyNet, MicroISP, AWNet, LiteISP, RMFA-Net | LoRA on full model (CDM-QTA, ROMA for LLM) |
| **Multiplier-free / frozen-weight HW** | **MCM-SR**, FixyNN, ShiftAddNet, AdderNet | **🎯 RecurISP (this work)**                |

기존 multiplier-free SR/ISP 가속기는 weight가 합성되어 deploy 후 변경 불가능하다. 반면 LoRA hardware accelerator들(ROMA, CDM-QTA, PRIMAL)은 LLM/diffusion 도메인이며 multiplier-rich base를 전제한다. **RecurISP는 multiplier-free base + adaptable LoRA side-path**라는 사분면을 ISP 도메인에서 채운다.

---

## 1. Motivation & Positioning

### 1.1 Hardware immutability 문제

Mobile SoC의 ISP 가속기는 다음 셋 중 하나로 구현된다.

1. **Programmable NPU** — multiplier 풍부, weight on-chip/DRAM, 자유롭게 update 가능. 그러나 area·energy 비용 큼.
2. **Streaming hardwired CNN** (예: MCM-SR) — multiplier를 shift-add로 합성, weight가 hardware에 박힘. Energy 효율 우수하나 **deploy 후 weight 변경 불가**.
3. **Hybrid (FixyNN-style)** — 일부 layer 고정 + 일부 programmable.

기존 multiplier-free SR 가속기 (MCM-SR류)는 (2)에 해당하며, 새로운 sensor/ISO/scene이 들어오면 re-synthesis 또는 re-tape-out이 필요하다.

> **MCM-SR solves efficiency, but not adaptability.**

### 1.2 RecurISP의 접근

세 design axis로 동시에 해결한다.

| 요구                       | 해결 도구                                           | 메커니즘                                                              |
| -------------------------- | --------------------------------------------------- | --------------------------------------------------------------------- |
| Image quality              | Looped Transformer refinement                       | shared block을 반복 적용해 점진적 보정                                |
| Multi-profile adaptation   | **Iteration-conditioned LoRA side-path**            | base는 frozen, condition은 low-rank update                            |
| HW efficiency under freeze | DyT-HDR + adaptive early-exit + MCM-realizable base | reduction-free normalization, condition-aware compute, shift-add base |

핵심 insight: **base의 weight가 변하지 않는다는 제약은 quality-adaptive ISP의 death sentence가 아니다.** Adaptation은 base path 밖으로 빼면 된다. 이게 LoRA의 자연스러운 hardware story다.

### 1.3 가장 가까운 prior work와의 관계

#### MCM-SR (Lim et al., IEEE 2024) — 가장 중요한 hardware prior

- **공통**: SR 도메인, streaming datapath, line buffer 기반, multiplier-free
- **차이**: MCM-SR은 **static**, 본 연구는 **frozen base + adaptable side-path**
- **활용**: MCM-SR을 base path의 hardware realization 후보로 사용. Cycle-level 비교에서 baseline.

#### FixyNN (Whatmough et al., MLSys 2019) — 가장 중요한 conceptual prior

- **공통**: "freeze the heavy part, adapt the light part" 패턴
- **차이**: FixyNN은 classification, 본 연구는 ISP regression. FixyNN의 back-end는 full programmable, 본 연구의 side-path는 **constrained low-rank**.
- **활용**: 패턴의 ISP 도메인 일반화로 framing.

#### ROMA (2025) — LLM 도메인의 직접 analog

- **공통**: frozen base를 비싼 메모리 구조에, LoRA를 빠른/유연한 메모리 구조에 — adapt path 분리
- **차이**: ROMA는 LLM autoregressive decoding, 본 연구는 streaming ISP
- **활용**: "domain analog" 인용으로 reviewer에게 친숙함 부여.

---

## 2. Architecture

### 2.1 전체 파이프라인 (Hardware-aware view)

```
RAW Bayer / Linear RGB
   │
   ▼
ISP Stem  (black-level, DPC, demosaic-aware conv, shallow denoise)
   │
   ▼
CNN Local Feature Encoder ◄─── MCM-realizable (frozen, shift-add)
   │
   ▼
AdaLoop-DyT Refinement × T_max
   ├──── Shared DyT Block ◄────── MCM-realizable (frozen, shift-add)
   └──── LoRA_t side-path ◄────── programmable (INT4/PoT/ternary, multiplier-rich)
   │
   ▼
ISP Reconstruction Head ◄──── MCM-realizable (frozen, shift-add)
   │
   ▼
sRGB / YUV
```

**Hardware partitioning rule:**

- Base path (Encoder + SharedDyTBlock + Head): frozen weights → **MCM/shift-add 합성 가능**
- Side path (LoRA_t per iteration): programmable, low-rank, condition-driven → **small multiplier engine 또는 INT4 MAC array**

### 2.2 AdaLoop-DyT Refinement Block

```
F₀ = Encoder(x)
for t = 1 ... T_max:
    c_t = ConditionEncoder(sensor_id, ISO, gain, exposure, CCT, t)
    Fₜ = Fₜ₋₁
        + SharedDyTBlock(Fₜ₋₁)        # frozen base, MCM-realizable
        + LoRA_t(Fₜ₋₁, c_t)           # adaptable side-path
    if EarlyExit(Fₜ, c_t):
        break
return F_t
```

### 2.3 SharedDyTBlock 내부

```
Input F
   ▼
DyT-HDR
   ▼
Local-Window Attention + Channel Attention
   ▼
LoRA_t(·, c_t)               ◄── iteration-specific side-path
   ▼
Residual Add
   ▼
DyT-HDR
   ▼
Depthwise-Conv FFN
   ▼
LoRA_t(·, c_t)               ◄── iteration-specific side-path
   ▼
Residual Add
```

ISP-적합 attention 선택:

- **Local window attention** — texture, edge
- **Strip attention** — row/column sensor artifact (column FPN 등)
- **Channel attention** — color/white balance

본 논문은 **local window + channel attention 결합**으로 commit, strip attention은 ablation.

---

## 3. Core Components

### 3.1 Iteration-Conditioned LoRA (Side-Path)

#### 3.1.1 기본 형태

```
W_eff,t = W_shared + A_t(c) B_t(c)        ← base + side-path
y = x · W_shared + (x A_t) B_t           ← side-path은 multiplier-rich
```

또는 lighter variant:

```
W_eff,t = W_shared + s_t(c) · A_t B_t
```

#### 3.1.2 Quantization-constrained side-path (NEW: 하드웨어 contribution)

Side-path는 base보다 작은 quantization budget을 가질 수 있다:

| Side-path quantization | Multiplier 비용 | 표현력                      |
| ---------------------- | --------------- | --------------------------- |
| FP16                   | high            | 최고                        |
| INT8                   | medium          | 우수                        |
| **INT4**               | **low**         | 본 연구 main target         |
| **PoT (Power-of-Two)** | **shift only**  | aggressive, multiplier-free |
| Ternary {-1,0,+1}      | add only        | 가장 aggressive             |

> **Design space contribution**: We systematically explore how much the side-path can be constrained while preserving image fidelity.

#### 3.1.3 Gradient collapse 방지 — Orthogonality regularization

```
L_ortho = Σ_{i≠j} ||Â_iᵀ Â_j||_F² ,   Â_i = A_i / ||A_i||_col
```

#### 3.1.4 Residual magnitude balancing

```
L_balance = Var( ||LoRA_t(x)||₂  across t )
```

#### 3.1.5 Subspace 설계 — 한쪽만 commit

| 옵션                                           | 형태                                             | 특성                               |
| ---------------------------------------------- | ------------------------------------------------ | ---------------------------------- |
| **A. Independent A_t, B_t + ortho reg** (main) | `W_eff,t = W_shared + A_t B_t`                   | 표현력↑, params `T·2·r·d`          |
| B. Block-diagonal sharing (variant)            | `A_shared ∈ ℝ^(d × Tr)`, t는 [tr:(t+1)r] columns | params `2·r·d`, subspace 자연 분리 |

> ⚠️ Ortho reg와 rank sharing은 동시 적용하지 않는다 (충돌).

#### 3.1.6 Parameter cost

- LoRA 추가 params: `T_max × 2 × r × d`
- T=4, r=8, d=256 → ~16K params
- **Rank sweep 필수**: r ∈ {4, 8, 16, 32}

### 3.2 DyT-HDR

```
DyT-HDR(x) = x + λ · tanh(α · x)
```

- Linear path `x`: dynamic range preservation
- Tanh path `λ tanh(αx)`: outlier suppression
- Per-channel learnable α, λ

원형 DyT [Zhu et al., CVPR 2025]는 normalization 대체로 제안되었으나, ISP RAW/HDR wide dynamic range에서는 saturation 위험이 있어 residual linear path를 추가했다.

#### Optional clamp variant

```
DyT-HDR-clamp(x) = x + λ · tanh(α · clamp(x, -β, β))
```

### 3.3 Adaptive Early-Exit

#### 3.3.1 Energy criterion

```
energy_t = ||F_t - F_{t-1}||₂ / ||F_{t-1}||₂
```

또는 learned MLP predictor.

#### 3.3.2 Training — Expected loss over exits

```
L = Σ_t p_t · L_t ,   p_t = softmax(-energy_t / τ_train)
```

**Schedule:**

- Warmup (epoch 0 ~ E_warm): uniform p_t = 1/T_max
- Annealing: τ_train 점진 감소
- Per-iteration reconstruction loss L_t

#### 3.3.3 Inference threshold

- Validation Pareto sweep
- Condition별 threshold table

#### 3.3.4 Hardware honesty

> Dynamic iteration은 평균 latency 감소시키지만 control divergence 도입. Mobile ISP는 batch=1이라 영향 제한적.

### 3.4 Condition Encoder

| Input     | Encoding                                                                                     |
| --------- | -------------------------------------------------------------------------------------------- |
| sensor_id | embedding lookup                                                                             |
| ISO       | log normalized                                                                               |
| gain      | linear normalized                                                                            |
| exposure  | log normalized                                                                               |
| **CCT**   | **mired (1e6/CCT) normalized** (sinusoidal NOT used — ordinal continuous, mired에서 더 균등) |
| t         | learnable embedding                                                                          |

```
c_t = MLP([sensor_emb, logISO, gain, log_exposure, mired_CCT, t_emb])
```

### 3.5 LoRA Runtime Modes

| Mode                                        | 형태                                             | 용도                                                           |
| ------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------- |
| **Offline-fused**                           | profile별 `W_eff = W_shared + A_t B_t` 사전 합성 | 그러나 **base가 frozen이면 fuse 불가**. Side-path만 별도 저장. |
| **Online side-path** (default for RecurISP) | `y = xW_shared + xA_t B_t` runtime 분리 계산     | Frozen base의 본질적 동작 방식                                 |

> **본 연구의 stance:** Base가 hardware-frozen이므로 LoRA는 **반드시 online side-path**로 동작. 이는 base와 side-path를 별도 hardware engine으로 매핑하는 것이 자연스럽다는 의미. (ROMA의 ROM-vs-SRAM 분리와 동일한 원리.)

---

## 4. Hardware Architecture (NEW: 강화)

### 4.1 Datapath template (Figure 1 plan)

```
           ┌─────────────────────────────────────┐
           │  Base Path (frozen, MCM-style)      │
RAW ──►────┤  - shift-add CONV/DyT engine        ├──► +
           │  - line buffer, no DRAM weight read │     │
           └─────────────────────────────────────┘     │
                                                       │
           ┌─────────────────────────────────────┐     │
           │  Side Path (programmable)           │     │
           │  - INT4/PoT MAC array               ├──► +
           │  - LoRA_t weights from SRAM         │     │
           │  - condition c_t selects t          │     │
           └─────────────────────────────────────┘     │
                                                       ▼
                                                   refined F_t
```

### 4.2 Scheduling

- **Base path**: streaming, line-buffer based (MCM-SR 호환)
- **Side path**: per-iteration weight load from SRAM (small footprint due to low rank)
- **Synchronization**: side-path output가 base-path output과 element-wise add 되는 시점

### 4.3 Cycle-level comparison plan

| Baseline            | Configuration                                       |
| ------------------- | --------------------------------------------------- |
| MCM-SR static       | RecurISP base only, no LoRA, no early-exit          |
| Programmable NPU    | Full multiplier MAC array, 모든 weight programmable |
| **RecurISP (ours)** | MCM base + INT4 side-path + early-exit              |

**Reporting**: latency / energy / area / SRAM access / DRAM bandwidth across SR scaling factor (×2, ×4) and ISP resolution (FHD, 4K).

---

## 5. Contributions (확정)

1. **Iteration-conditioned low-rank adaptation for recurrent ISP** — shared backbone + per-iteration LoRA + ortho regularization으로 weight sharing 하에서 표현력 확장
2. **DyT-HDR** — RAW/HDR을 위한 residual linear path를 갖는 reduction-free recurrent feature stabilizer
3. **Condition-adaptive early-exit** — expected-loss training과 energy-based gating으로 condition-aware compute scaling
4. **🆕 Constraint-aware LoRA side-path under MCM-frozen base** — frozen multiplier-free base + INT4/PoT side-path 분리로 hardware immutability 하에서의 adaptation 달성. MCM-SR 등 streaming SR 가속기에 직접 적용 가능한 architectural pattern.
5. **🆕 Hardware-aware deployment analysis** — cycle-level comparison vs MCM-SR static / programmable NPU 보고

---

## 6. Experimental Design

### 6.1 Baseline Ladder

| ID     | 구성                                        | 검증 목표                      |
| ------ | ------------------------------------------- | ------------------------------ |
| B0     | CNN ISP baseline (SESR/QuickSRNet)          | Lower bound                    |
| B1     | CNN + non-shared Transformer refinement     | Looped 비교                    |
| B2     | CNN + shared looped Transformer (LayerNorm) | Weight sharing                 |
| B3     | B2 + timestep embedding                     | Timestep awareness             |
| B4     | B3 + shared LoRA                            | LoRA 일반 효과                 |
| **B5** | B3 + **per-iteration LoRA**                 | **🎯 contribution 1**          |
| B6     | B5 + **DyT-HDR**                            | 🎯 contribution 2              |
| B7     | B6 + **adaptive early-exit**                | 🎯 contribution 3              |
| **B8** | B7 + **INT4 side-path on frozen base**      | 🎯 contribution 4              |
| **B9** | B7 + **PoT (shift-only) side-path**         | 🎯 contribution 4 (aggressive) |

**핵심 비교쌍:**

- B4 vs B5 — per-iteration LoRA 효과 (논문의 척추)
- B5 vs B5+ortho-reg — orthogonality 효과
- B6 vs (GN/IN/Scale-only)-residual — DyT-HDR 진짜 기여
- B7 vs B7-fixed-T — early-exit Pareto
- B7 vs B8 vs B9 — side-path quantization 영향
- **B8/B9 vs MCM-SR** — adaptable vs static 직접 비교

### 6.2 Ablation 항목

#### Per-iteration LoRA

- Rank: r ∈ {4, 8, 16, 32}
- T_max: ∈ {2, 3, 4, 6}
- Regularizer: none / ortho / balance / both
- Subspace 설계: independent + ortho vs block-diagonal

#### DyT-HDR (공정 비교)

모든 후보에 동일한 residual linear path 부여:

| 비교               | 형태             |
| ------------------ | ---------------- |
| DyT-HDR            | `x + λ·tanh(αx)` |
| GN-residual        | `x + λ·GN(x)`    |
| IN-residual        | `x + λ·IN(x)`    |
| Scale-only         | `x + λ·γ⊙x`      |
| LayerNorm (legacy) | `LN(x)`          |

**Stratified evaluation**: low-light (ISO≥3200), HDR scene, high ISO (>6400), skin tone, daylight low-ISO.

#### Adaptive early-exit

- Fixed T vs adaptive
- Energy criterion (norm) vs learned MLP
- τ(c) sweep → quality-latency Pareto

#### Side-path quantization (NEW)

- FP16 / INT8 / INT4 / PoT / Ternary
- Quality drop vs hardware cost trade-off
- Per-condition stratified (어떤 condition에서 quantization 민감한지)

#### OOD sensor generalization

- **Leave-one-sensor-out cross-validation**
- Backbone freeze → LoRA fine-tune
- 비교: full retrain / LoRA fine-tune / LoRA cold-start
- 메시지: "새 센서는 base 재합성 없이 LoRA만 fine-tune"

### 6.3 Datasets

| Dataset                          | 용도                                     |
| -------------------------------- | ---------------------------------------- |
| Zurich RAW2RGB [Ignatov+2020]    | Public RAW→sRGB benchmark                |
| Fujifilm UltraISP [Ignatov+2022] | Mobile AI Challenge dataset              |
| MIT-Adobe FiveK                  | Tone/style                               |
| SIDD                             | Real noise                               |
| **S.LSI 내부 multi-sensor**      | OOD generalization, production-realistic |

### 6.4 Evaluation Metrics

#### Image quality

- PSNR, SSIM, LPIPS
- ΔE (CIEDE2000)
- ITA (Individual Typology Angle) — skin tone
- Color constancy error (Auer angular)

#### ISP-specific

- Low-light noise residual (flat region std)
- HDR highlight recovery (clipped region SSIM)
- Demosaic artifact (zipper edge)
- Banding score

#### Computational / Hardware

- Parameter count (base / side-path 분리 보고)
- MACs (multiplier-required vs shift-add)
- Activation memory (peak, per-iteration)
- SRAM traffic
- Average iteration count
- **🆕 Side-path bit-width vs quality Pareto**
- **🆕 Cycle-level latency / energy / area vs MCM-SR baseline**

### 6.5 Visualization & Interpretability

#### 본문 (quantitative)

- **Frequency-domain analysis**: 각 iteration ΔF_t의 PSD
- **Channel sensitivity**: ∂output/∂LoRA_t per R/G/B
- **LoRA subspace overlap**: ||Â_iᵀ Â_j|| heatmap (T×T)

#### Supplementary (qualitative)

- ΔF_t 시각화
- Failure cases

### 6.6 Hardware Evaluation

| 옵션                                                    | 내용                      | Venue 적합성                 |
| ------------------------------------------------------- | ------------------------- | ---------------------------- |
| A. Cycle-level simulator (Py-V 등)                      | DMA/SRAM/MAC latency 모델 | CVPR/ICCV                    |
| **B. 실측** (S.LSI Custom SoC NPU 또는 ref. mobile NPU) | Actual silicon            | **ISSCC/JSSC/HotChips 분기** |

**Calibration (옵션 A)**: 표준 conv/MatMul layer를 simulator vs 실측 NPU correlation 보고.

**Reporting**:

- Latency (ms/frame)
- Energy (mJ/frame)
- Area (mm² synthesized)
- SRAM access (MB)
- DRAM BW (MB/s)
- vs MCM-SR / Vanilla Transformer ISP / Programmable NPU baseline

### 6.7 Training Memory 정직 보고

T_max unrolled BPTT → activation memory × T_max. 처리:

- Gradient checkpointing (memory ~O(√T))
- Truncated BPTT (옵션)
- Framing: "Inference 효율 ↔ training one-time cost"

---

## 7. Risk Management & Plan B

| Risk                                                 | 대응                                                                       |
| ---------------------------------------------------- | -------------------------------------------------------------------------- |
| Per-iteration LoRA의 specialization이 명확히 안 보임 | Primary claim은 capacity expansion이므로 영향 없음. Secondary는 톤 조정.   |
| DyT-HDR이 GN-residual 대비 미미                      | Stratified eval(low-light/HDR)에서만 우위 → "RAW/HDR-specialized" 포지셔닝 |
| Adaptive early-exit quality drop                     | Pareto curve로 framing → "user-tunable speed-quality knob"                 |
| OOD sensor LoRA fine-tune 격차                       | Rank 증가 또는 partial unfreeze 옵션. 같은 sensor family 내 결과 강조.     |
| Side-path INT4/PoT에서 quality 큰 drop               | INT8 fallback option. "quantization spectrum" framing.                     |
| HW 측정 calibration 부족                             | Reference public NPU 대체. Simulator + correlation 정직 처리.              |

**Framing 원칙:** 본문에 "Plan B" 명시하지 않음. 결과 적응형 톤.

---

## 8. Venue Strategy

| Venue              | Fit                      | 셀링 포인트                                       |
| ------------------ | ------------------------ | ------------------------------------------------- |
| CVPR / ICCV / ECCV | ★★★                      | Image quality + LoRA novelty + capacity expansion |
| WACV               | ★★                       | Mobile ISP application                            |
| **ISSCC / JSSC**   | **★ → ★★★ (HW 실측 시)** | MCM-base + LoRA side-path co-design               |
| HotChips / DAC     | ★★ (HW 실측 시)          | System deployment                                 |

> **추천 sequence:** CVPR/ICCV submission → 동일 연구의 hardware-extended version을 ISSCC/JSSC 분기 (S.LSI 환경 실측 가능 시).

---

## 9. Implementation Notes

### 9.1 Codebase

- PyTorch 2.x, FP16 mixed precision
- Per-iteration LoRA: `nn.ParameterList`
- Side-path quantization: PTQ (post-training) + QAT (quantization-aware training) 모두 실험
- DyT-HDR: custom `nn.Module` with learnable α, λ (per-channel)

### 9.2 Training schedule

- Warmup 5 epochs (uniform exit prob)
- Total 100 epochs Zurich RAW2RGB (preliminary), longer on full
- AdamW, cosine schedule, base lr 2e-4

### 9.3 Reproducibility

- Ablation 3 seed 평균
- HW 측정 N=100 frame, 5% trim, mean ± std

---

## 10. Final Abstract Draft

> We present **RecurISP**, a recurrent neural ISP architecture designed for mobile platforms where the heavy computation must remain hardware-frozen. Building on insights from MCM-based SR streaming accelerators [MCM-SR; Lim et al. 2024] and fixed-weight feature extractors [FixyNN; Whatmough et al. 2019], RecurISP separates a frozen, MCM-realizable base from a small, adaptable LoRA side-path. The base is a shared looped block stabilized by **DyT-HDR**, a residual nonlinear normalization-free unit suited to RAW/HDR wide dynamic range. The side-path consists of **iteration-conditioned LoRA** updates, regularized to occupy distinct subspaces across iterations, and is permitted to operate at aggressively quantized precision (INT4 or power-of-two). An **expected-loss-trained early-exit** mechanism scales compute to scene difficulty. Across public benchmarks and a multi-sensor evaluation protocol, RecurISP achieves competitive image quality with significantly reduced multiplier-rich computation and supports **out-of-distribution sensor generalization via LoRA-only fine-tuning** of a frozen base. We further report cycle-level latency and energy comparisons against static MCM-SR baselines and programmable NPUs.

---

## 11. Action Items (priority order)

1. **B4 vs B5 정량 비교 + rank sweep** — 논문의 척추
2. **B7 vs B8 vs B9 (side-path quantization)** — hardware contribution 4 검증
3. **OOD sensor + LoRA fine-tune 실험** — leave-one-sensor-out
4. **DyT-HDR vs (GN/IN)-residual stratified comparison** — reviewer 방어
5. **Cycle-level comparison vs MCM-SR baseline** — venue 분기 결정
6. **Frequency-domain analysis pipeline** — secondary claim 검증
7. **Ortho regularizer ablation** — capacity expansion 보강
8. **Early-exit Pareto sweep** — quality-latency trade-off

---

## 12. Related Work (with full citations)

### 12.1 Multiplier-free / Hardware-frozen Accelerators (RecurISP의 직접 prior)

- **MCM-SR** — H. Lim et al., "MCM-SR: Multiple Constant Multiplication-Based CNN Streaming Hardware Architecture for Super-Resolution," _IEEE Transactions on VLSI Systems_, 2024. ([IEEE Xplore 10777852](https://ieeexplore.ieee.org/document/10777852/))
- **FixyNN** — P. Whatmough, C. Zhou, P. Hansen, M. Mattina, "FixyNN: Efficient Hardware for Mobile Computer Vision via Transfer Learning," _MLSys 2019_. arXiv:1902.11128
- **HDSuper** — L. Chang et al., "HDSuper: Algorithm-Hardware Co-design for Light-weight High-quality Super-Resolution Accelerator," 2023.
- **ShiftAddNet** — H. You et al., "ShiftAddNet: A Hardware-Inspired Deep Network," _NeurIPS 2020_.
- **AdderNet** — H. Chen et al., "AdderNet: Do We Really Need Multiplications in Deep Learning?," _CVPR 2020_.
- **DeepShift** — M. Elhoushi et al., "DeepShift: Towards Multiplication-Less Neural Networks," _CVPR Workshops 2021_.
- **NASA** — Y. Fu et al., "NASA: Neural Architecture Search and Acceleration for Hardware Inspired Hybrid Networks," _ICCAD 2022_. arXiv:2210.13361
- **ShiftAddNAS** — H. You et al., "ShiftAddNAS: Hardware-Inspired Search for More Accurate and Efficient Neural Networks," _ICML 2022_. arXiv:2205.08119
- **ShiftAddAug** — Y. Guo et al., "ShiftAddAug: Augment Multiplication-Free Tiny Neural Network with Hybrid Computation," 2024. arXiv:2407.02881
- **PoT-quantization** — H. Tann et al., "Hardware-Software Codesign of Accurate, Multiplier-free Deep Neural Networks," _DAC 2017_. arXiv:1705.04288

### 12.2 LoRA Hardware Accelerators (직접적 LoRA HW prior)

- **ROMA** — K. Kong et al., "ROMA: a Read-Only-Memory-based Accelerator for QLoRA-based On-Device LLM," 2025. arXiv:2503.12988 — _frozen ROM base + SRAM LoRA, RecurISP의 LLM-domain analog_
- **CDM-QTA** — "CDM-QTA: Quantized Training Acceleration for Efficient LoRA Fine-Tuning of Diffusion Model," 2025. arXiv:2504.07998
- **PRIMAL** — "PRIMAL: Processing-In-Memory Based Low-Rank Adaptation for LLM Inference Accelerator," 2025. arXiv:2601.13628
- **TrainDeeploy** — "TrainDeeploy: Hardware-Accelerated Parameter-Efficient Fine-Tuning of Small Transformer Models at the Extreme Edge," 2025. arXiv:2603.09511
- **AHWA-LoRA** — "Efficient transformer adaptation for analog in-memory computing via low-rank adapters," 2024. arXiv:2411.17367

### 12.3 LoRA & Parameter-Efficient Fine-Tuning

- **LoRA** — E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," _ICLR 2022_. arXiv:2106.09685

### 12.4 Looped / Weight-Shared Transformers

- **Universal Transformer** — M. Dehghani et al., "Universal Transformers," _ICLR 2019_. arXiv:1807.03819
- **ELT** — "ELT: Elastic Looped Transformers for Visual Generation," 2025.
- **SpiralFormer** — Yu et al., 2025/2026 — early loops global, later loops local refinement.
- **Loop Neural Networks** — "Loop Neural Networks for Parameter Sharing," 2024. arXiv:2409.14199

### 12.5 Normalization-Free Transformers

- **DyT** — J. Zhu, X. Chen, K. He, Y. LeCun, Z. Liu, "Transformers without Normalization," _CVPR 2025_, pp. 14901-14911. arXiv:2503.10622
- **Stronger Norm-Free** — "Stronger Normalization-Free Transformers," 2025. arXiv:2512.10938

### 12.6 Adaptive Computation / Early Exit

- **BranchyNet** — S. Teerapittayanon, B. McDanel, H.-T. Kung, "BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks," _ICPR 2016_. arXiv:1709.01686
- **MSDNet** — G. Huang et al., "Multi-Scale Dense Networks for Resource Efficient Image Classification," _ICLR 2018_.
- **Adaptive Neural Networks** — "Adaptive Neural Networks for Efficient Inference," 2017. arXiv:1702.07811

### 12.7 Neural ISP (image quality baseline 비교군)

- **PyNet** — A. Ignatov et al., "Replacing Mobile Camera ISP with a Single Deep Learning Model," _CVPR Workshops 2020_.
- **PyNet-V2 Mobile** — A. Ignatov et al., "PyNET-V2 Mobile: Efficient On-Device Photo Processing With Neural Networks," 2022. arXiv:2211.06263
- **MicroISP** — A. Ignatov et al., "MicroISP: Processing 32MP Photos on Mobile Devices with Deep Learning," 2022. arXiv:2211.06770
- **AWNet** — L. Dai et al., "AWNet: Attentive Wavelet Network for Image ISP," 2020.
- **LiteISP** — Zhang et al., 2021.
- **MW-ISPNet** — A. Ignatov et al., 2020.
- **CSANet (AI-ISP accelerator)** — "AI-ISP Accelerator with RISC-V ISA Extension for Image Signal Processing," _IEEE 2024_.
- **RMFA-Net** — "RMFA-Net: A Neural ISP for Real RAW to RGB Image Reconstruction," 2024. arXiv:2406.11469
- **Simple ISP w/ Global Context** — "Simple Image Signal Processing using Global Context Guidance," 2024. arXiv:2404.11569

### 12.8 Datasets / Benchmarks

- **Zurich RAW2RGB** — Ignatov, Van Gool, Timofte, 2020.
- **Fujifilm UltraISP / MAI 2022** — A. Ignatov et al., "Learned Smartphone ISP on Mobile GPUs with Deep Learning, Mobile AI & AIM 2022 Challenge: Report," 2022. arXiv:2211.03885
- **SIDD** — Abdelhamed et al., "A High-Quality Denoising Dataset for Smartphone Cameras," _CVPR 2018_.
- **MIT-Adobe FiveK** — Bychkovsky et al., "Learning Photographic Global Tonal Adjustment with a Database of Input/Output Image Pairs," _CVPR 2011_.

---

_문서 버전: Archived v5.0 future-work seed (Related Work integrated, MCM-SR / FixyNN / ROMA-class prior art positioning 반영)_
_작성 환경: Samsung S.LSI Custom SoC team, platform definition 연계_
