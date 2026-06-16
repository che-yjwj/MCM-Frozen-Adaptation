# v8.2 Merge Inserts — order C → B → A

> v8.1 본문에 **드롭인**할 수 있게 작성. 각 블록은 “교체/추가 위치”를 명시.
> 문서 스타일(사전 등록 threshold, 표)에 맞춤. 서지 미상은 `(서지 확인 필요)` 표기.

-----

# (C) residual-field dynamics를 C2 핵심 기여로 승격 + Pilot 4 재설계

## C-0. 왜 이게 지적 핵심인가 (한 문단, §0 또는 §3.1 근처에 삽입)

> MCM 합성된 base가 계산하는 것은 학습된 float weight `W`가 아니라, shift-add 그래프가 실현
> 가능한 값으로 절단·반올림된 `W̃`이다. 따라서 배포된 base는 `f_W̃(x) = f_W(x) + ε_synth(x)`를
> 계산하며, **`ε_synth`는 합성 시점에 고정된, 입력 의존적·구조화된 결정론적 오차장(field)**이다
> (균일 양자화 noise가 아니라 공유 부분식·절단 규칙에 의해 채널 상관·저랭크로 구조화됨).
> MCM-frozen base 위의 parallel LoRA는 두 가지를 **동시에** 흡수해야 한다: (i) 목표인 sensor-domain
> shift, (ii) 의도치 않은 세금인 `ε_synth`. 본 논문의 과학적 핵심 질문은
> **“구조화된 `ε_synth` × 곱셈기-제약(PoT/Ternary) 보정기 × 병합 불가(non-mergeable)의 삼중 조합이,
> QLoRA/AdaptSR가 가진 (균일 잔차 × FP 보정기 × 병합 가능)에는 없는 용량 병목·학습 병리를 만드는가”**이다.

**QLoRA와의 분리(중요)**: QLoRA도 LoRA가 base 양자화 잔차를 암묵 보정한다. 신규성은 “잔차 보정”
자체가 아니라 **(a) 잔차가 균일 양자화가 아니라 MCM 그래프 구조에 의해 저랭크/채널상관으로
구조화**되고, **(b) 보정기 자신이 곱셈기-free(PoT/Ternary)로 제약**되며, **(c) 재합성 없인 병합이
원천 불가**라는 세 제약이 겹친다는 점이다.

## C-1. C2 contribution 문구 교체 (§8, Contribution 2 → 아래로 대체)

> **2. Constrained-LoRA over a hardware-frozen residual field (핵심 기여).** MCM 합성 잔차
> `ε_synth`가 고정 오차장이 되는 setting에서, 곱셈기-제약 저랭크 보정기가 *sensor 적응 용량*과
> *합성 잔차 보정*을 어떻게 분배·경쟁하는지를 정량화한다. 구체적으로 (i) `ε_synth`가 소비하는
> 보정기 용량의 비율(“residual tax”), (ii) PoT/Ternary 제약이 이 잔차 보정과 상호작용하여
> 추가 손실을 유발하는지, (iii) 병합 불가 parallel 구조가 mergeable LoRA(AdaptSR/QLoRA) 대비
> 최적화 dynamics를 바꾸는지를 보인다.

## C-2. Pilot 4 전면 재설계 (§6.6 → 아래로 대체)

### Pilot 4 (재설계): Residual-Field Capacity Competition (3~5 days, **CRITICAL — C2의 생사**)

**핵심 측정 자산**: float base와 MCM base를 둘 다 보유하므로 `ε_synth`는 **오프라인에서 직접
계산 가능**하다. 이것이 이 pilot을 결정 가능하게 만든다.

```
정의 (calibration 분포 위에서):
  r_synth(x) = f_W (LR)        − f_W̃ (LR)     # 합성 오차를 되돌리는 데 필요한 보정장
  r_task (x) = HR_target        − f_W (LR)     # clean float base 기준 sensor 적응 잔차
  이상적 LoRA(MCM base) ≈ r_synth + r_task     # 둘 다 떠안아야 함
```

**측정 1 — Capacity accounting (residual tax):**
학습된 `LoRA(x)`를 `r_synth` 부분공간과 `r_task` 부분공간에 사영.
`tax = ‖proj_{r_synth} LoRA‖ / ‖LoRA‖` = 합성 잔차 보정에 소비된 용량 비율.

**측정 2 — 결정적 ablation (이게 없으면 C2 폐기):**

|Setting|base                              |의미       |
|-------|----------------------------------|---------|
|S_float|float weight frozen (= ε_synth 없음)|잔차세 0 기준선|
|S_mcm  |MCM-frozen (ε_synth 존재)           |본 설정     |
|S_int8 |INT8 weight frozen (균일 양자화 잔차)    |QLoRA류 대조|

같은 LoRA·rank·data·target에서 S_float vs S_mcm 차이 = **합성 잔차세를 순수 분리**.
S_mcm vs S_int8 차이 = **“구조화 잔차 vs 균일 잔차”** 효과 분리.
→ S_float ≈ S_mcm 이면 residual-field claim은 죽음(정직하게 demote). 다르면 C2 승격.

**측정 3 — Rank tax 곡선:**
adaptation gain G(r) vs rank `r ∈ {2,4,8,16}`를 S_float·S_mcm 각각에 대해.
같은 gain 도달에 필요한 rank 수평 격차 = `ε_synth`가 잡아먹은 유효 용량.

**측정 4 — 정밀도×잔차 상호작용(이중 제약):**
FP→PoT, FP→Ternary retention을 S_float vs S_mcm에서 각각.
`drop_PoT(S_mcm) − drop_PoT(S_float) > 0` 이면 “구조화 잔차가 저정밀 보정기를 특히 괴롭힌다”는
새 발견(= QLoRA에 없는 dynamics).

**측정 5 — `ε_synth` 구조 특성화:**
공간 균일/채널 상관/고주파(절단) 여부 → 저랭크 보정기가 왜 흡수 가능/불가능한지 설명.
(채널상관·저랭크면 LoRA가 싸게 흡수; 고주파 공간잡음이면 흡수 실패.)

**Pre-registered 승격/강등 threshold (시작 전 고정):**

|결과                                                                          |결정                                                                                        |
|----------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
|S_mcm이 S_float·S_int8과 **유의하게 다름** (tax ≥ ~15%, 또는 rank tax ≥ 2×, 또는 측정4 양수)|C2를 **main 기여로 승격**, RQ7을 본문 핵심 분석으로                                                      |
|차이 작음 (tax < ~5%, 곡선 거의 겹침)                                                 |C2 **demote** → 논문은 hardware-contract trade-off study로 유지, dynamics는 1문단 negative result로만|
|중간                                                                          |trade-off framing 유지하되 측정1·4를 supporting으로 보고                                             |


> 주: threshold 수치(15%/5%/2×)는 placeholder. Pilot 4 착수 *전*에 작은 dry-run으로 noise floor를
> 잡아 확정할 것(측정 자체가 confirmation bias 안 타게).

-----

# (B) SR 삽입 위치 확정(sRGB) + 동기·데이터셋 재정렬

## B-0. 결정 (ratify 필요 — 뒤집으려면 RAW 분기로)

**결정: sRGB-domain SR로 확정.** 근거: SESR/QuickSRNet·DRealSR/RealSR·“demosaic out-of-scope(§12)“가
모두 sRGB 전제라, RAW로 가면 backbone·dataset·scope 세 곳이 동시에 깨진다. 대신 **동기를 재정의**해
sRGB에서도 sensor 적응이 정당하도록 만든다(아래 B-1). RAW 분기는 B-3에 future-work로 보존.

## B-1. 동기 재작성 (§2.3 / §2.4 교체 — “ISP가 이미 처리한다” 반론 봉쇄)

> **§2.3 (재작성) Why per-camera adaptation, even after the ISP.** 모바일/로봇 파이프라인에서 ISP는
> black level, lens shading, white balance, per-illuminant CCM, exposure 등 **전역·통계적 sensor
> 정규화**를 SR 이전에 수행한다. 따라서 본 연구는 sensor의 color/noise 통계 보정을 SR이 다시 한다고
> 주장하지 않는다. SR backbone이 암묵적으로 가정하는 것은 그보다 **카메라 고유의 *공간 열화 연산자***
> — 광학 PSF, anti-aliasing 필터, demosaic 보간 artifact, in-camera sharpening/denoise/compression —
> 이며, 이는 ISP가 정규화하지 못하고 **카메라마다 다르다**(RealSR/DRealSR의 inter-camera PSNR gap이
> 곧 이 연산자 차이의 증거). MCM 합성은 이 열화 연산자의 역(inverse)을 **하나의 기준 카메라에 대해**
> 실리콘에 동결시킨다. 새 카메라의 열화 연산자는 재합성 없이 바꿀 수 없으므로, parallel LoRA가
> 이 **연산자 잔차**를 보정하는 유일한 채널이 된다.

> **§2.4 (보강) Anticipated attack: “ISP가 처리하지 않나?”** → 위 논거를 명시 답변으로:
> 통계적 sensor 정규화(ISP) ≠ 공간 열화 연산자 역변환(SR). 후자는 per-camera이며 frozen base가 추적 불가.

**추가 권장 baseline(값싼 대안 봉쇄, Tier-3 지적 반영)**: frozen base *앞단*의 **per-sensor affine/color
전처리**(1×1 global correction)를 B-pre로 추가. “왜 mid-conv LoRA여야 하나(전처리로 안 되나)“에 대한
실험적 답. mid-conv LoRA가 B-pre를 이기면 “열화는 전처리로 흡수 불가한 공간 구조”라는 동기를 강화.

## B-2. 데이터셋 재정렬 (§1.3 / §6.1 / §7.5 / §9.2 일괄)

동기를 “per-camera 열화 연산자”로 좁혔으므로 **DRealSR/RealSR(sRGB 멀티카메라 쌍)이 정당한 primary**가
된다(서로 다른 카메라 열화를 그대로 페어링). 다만 타깃(모바일/로봇) vs 데이터(DSLR) external validity
격차는 명시 처리:

|항목              |v8.1               |v8.2 (변경)                                                                                       |
|----------------|-------------------|------------------------------------------------------------------------------------------------|
|Primary         |DRealSR (5 DSLR)   |유지 — **단, “controlled multi-camera 열화” 설정으로 positioning**                                       |
|일반화 점검          |RealSR-RAW = Plan B|**smartphone subset(RealSR-RAW의 sRGB-render 또는 모바일 쌍)을 generalization check로 승격**               |
|한계 명시(§9.2 L2 옆)|—                  |**L2′: DSLR을 proxy로 사용. 기전(per-camera 열화 연산자 shift)은 카메라 무관하므로 현상 검증엔 유효하나, 모바일 절대 수치는 다를 수 있음**|


> 핵심: “기전은 카메라-agnostic → DSLR은 *현상*의 valid proxy”라는 한 줄을 §9.2에 박으면,
> DSLR 데이터로 모바일 주장하는 약점이 “현상 vs 절대수치”로 분리되어 방어된다.

## B-3. RAW 분기 보존 (§12.1 Archived ideas 표에 1행 추가)

|v-future idea                                    |v8.2 decision            |future-work use                                            |
|-------------------------------------------------|-------------------------|-----------------------------------------------------------|
|RAW-domain joint demosaic+SR with MCM-frozen base|본 논문 제외(scope·dataset 충돌)|RAW에선 sensor 의존성 최대 → 동기 가장 강함. 별도 RAW-ISP 논문에서 LoRA 채널 재검증|

-----

# (A) ReMCM를 related work + baseline로 통합

## A-1. Related work 단락 (§1.1에 **MCM-SR 바로 다음**으로 삽입)

> **Reconfigurable MCM (ReMCM / RCCM / ReMB)** [Tummeltshammer–Hartley–Dempster, IEEE TCAD 2007;
> Kumm et al., pipelined RCM/RPAG; OSR; RCCM-for-DL]. ReMCM은 곱셈기 없이(가산·시프트·MUX) **단일
> 입력을 여러 상수 집합 중 하나와 런타임 선택 곱셈**하는 회로다. 표면적으로 이것은 “MCM base는
> frozen”이라는 본 논문의 전제를 정면으로 위협한다 — base를 ReMCM으로 합성하면 weight를 런타임에
> 바꿀 수 있기 때문이다. **그러나 ReMCM의 가변성은 닫혀 있다(closed-set)**: 선택 가능한 계수 집합은
> *합성(tape-out) 시점에 열거된 유한 config*에 한정되며, MUX/면적 비용이 config 수와 weight 다양성에
> 따라 증가해 config가 많아지면 MCM의 면적 이점이 잠식된다. 따라서 ReMCM은 **사전 열거된 소수
> 카메라 간 전환**에는 유효하지만, **tape-out 이후 등장하는 미지(open-set) 카메라의 열화 연산자**에는
> 새 config를 합성할 수 없어 무력하다. 본 논문의 LoRA 채널은 정확히 이 open-set·post-deployment
> 영역을 대상으로 하며, ReMCM은 closed-set 대안으로서 **본 논문의 가장 날카로운 baseline**이 된다.
> (참고: 동일 ReMCM 원리를 ISP 색계수에 *기여*로 쓰는 보완 연구가 가능하나, 본 논문에서는 *foil*로 쓴다.)

→ 이 단락은 thesis를 죽이지 않고 **정밀하게 좁힌다**: “frozen이라서 LoRA”가 아니라
**“closed-set 재구성으로 못 잡는 open-set 적응이라서 LoRA”**.

## A-2. §1.2 positioning 표에 행 추가

|              |Hardware efficiency|Adaptation                               |Adapter multiplier budget   |
|--------------|-------------------|-----------------------------------------|----------------------------|
|… (기존 행)      |                   |                                         |                            |
|**ReMCM base**|✅ multiplier-free  |⚠️ **closed-set만** (pre-enumerated config)|n/a (base MUX 전환)           |
|**Ours**      |✅ MCM base         |✅ **open-set** (post-deployment)         |✅ PoT/INT4/Ternary, explicit|

## A-3. Baseline 추가 (§7.1 표에 행 추가)

|#     |Baseline                                  |Purpose                            |Reference              |
|------|------------------------------------------|-----------------------------------|-----------------------|
|**B9**|**ReMCM base (K-config) + 런타임 config 선택** |**closed-set 적응 대안 — 가장 날카로운 foil**|TCAD 2007 / Kumm et al.|
|B-pre |frozen base 앞단 per-sensor affine/color 전처리|“전처리로 안 되나” 대안(§B-1)               |—                      |

**B9 실험 설계(공정·결정적):**

- DRealSR의 N개 카메라 중 **N−1개를 ReMCM config로 사전 합성**, 나머지 1개를 held-out.
- held-out 카메라에서: ReMCM은 nearest config 선택(열화), Ours(LoRA)는 작은 adapter 학습(회복).
- **두 축으로 보고**:
1. **Open-set 회복**: held-out PSNR/SSIM/LPIPS — Ours가 이겨야 함(핵심 주장).
1. **Closed-set·비용 스케일링**: 지원 sensor 수 K에 따른 (면적/저장) 증가 — ReMCM은 K config baking,
   Ours는 K개 작은 LoRA dispatch. K 증가 시 Ours의 incremental cost 우위.
- **정직한 양보**: 소수(K작음)·닫힌 sensor 집합에선 ReMCM이 이길 수 있음 → 그 영역을 명시하고,
  본 논문 주장은 **open-set + incremental scaling**으로 한정(문서 기존 톤과 일치).

## A-4. RQ 추가 (§5)

|RQ     |Question                                                                                                               |
|-------|-----------------------------------------------------------------------------------------------------------------------|
|**RQ8**|Closed-set ReMCM(K-config) 대비, open-set held-out 카메라에서 parallel LoRA의 적응 우위는 얼마이며, 지원 sensor 수 K에 따라 (면적/저장) 교차점은 어디인가?|

## A-5. Risk register 갱신 (§10)

|Risk                              |Severity          |Mitigation                                                                              |
|----------------------------------|------------------|----------------------------------------------------------------------------------------|
|Reviewer: “ReMCM으로 base 재구성하면 되잖아”|**High→Mitigated**|§1.1 ReMCM 단락 + B9 baseline. closed-set/open-set 구분과 K-scaling으로 정량 방어. held-out 실험이 핵심.|

## A-6. 참고문헌 추가 (§C.1 또는 §C.3 — 서지 확인 필요)

- **ReMCM/TMCM** — Tummeltshammer, Hartley, Dempster, “Time-Multiplexed Multiple Constant
  Multiplication,” *IEEE TCAD*, 2007. `(서지 확인 필요)`
- **Pipelined RCM (RPAG)** — Kumm et al., “Pipelined Reconfigurable Multiplication with Constants
  on FPGAs,” *FPL*-계열. `(서지 확인 필요)`
- **OSR** — “Optimal Shift Reassignment in Reconfigurable Constant Multiplication Circuits.” `(서지 확인 필요)`
- **RCCM-for-DL** — “Time-Multiplexed Multiple-Constant Multiplication” RCCM(딥러닝/FPGA 맥락 최신). `(서지 확인 필요)`

-----

## 적용 순서 체크리스트

1. **C** 먼저(논문 지적 핵심) — Pilot 4 재설계가 일정상 가장 먼저 결판나야 C2 승격/강등이 정해짐.
1. **B** 동기/데이터셋 — sRGB 확정 ratify 후 §2.3/2.4/6.1/7.5/9.2 일괄 수정.
1. **A** ReMCM — §1.1/1.2/5/7.1/10/C 추가. held-out(RQ8) 실험을 main 실험 배치에 편입.
