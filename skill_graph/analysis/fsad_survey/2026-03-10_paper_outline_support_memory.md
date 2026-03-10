# FSAD Paper Outline (Refined) — Support Utilization as the Bottleneck in Foundation-Model-based Few-Shot AD

> 작성일: 2026-03-10  
> 목적: 서베이 기반 문제의식을 논문 제출 가능한 수준의 서론/방법/실험 아웃라인으로 구체화

---

## 0. Quick Review (현재 초안의 강점과 보강점)

### 0.1 강점
- 문제의식이 명확하다: "강한 encoder 이후의 병목은 support utilization".
- 최신 baseline 축을 잘 잡았다: VisionAD, FoundAD, D2-4FAD.
- 제안 아이디어가 단순 모듈 추가가 아니라 memory organization 원리로 정리되어 있다.

### 0.2 보강이 필요한 지점
- claim-evidence 연결이 문단 서술 위주여서, 반증 가능한 형태로 보강이 필요하다.
- "representation보다 support가 병목"이라는 핵심 주장의 성립 조건/예외 조건이 명시되지 않았다.
- 방법 섹션이 개념 설명 중심이라 재현 가능한 수식/알고리즘 단계가 필요하다.
- 실험 설계가 baseline 나열 수준이며, 무엇을 증명/반증하는지 test matrix가 부족하다.

---

## 1. Field Trend Reframe (논문 서론용)

### 1.1 초기 FSAD의 중심 질문
- 초기 few-shot AD는 "적은 정상 샘플로 정상 분포를 어떻게 추정할 것인가"가 중심이었다.
- 대표 축:
  - Meta-learning: Metaformer, HTD, MetaUAS, MetaCAN
  - Memory bank/retrieval: PatchCore, GraphCore, Optimized PatchCore
  - Registration/correspondence: RegAD, CARL, FOCT
  - Reconstruction/generation: FastRecon, One-to-Normal, SeaS, POUTA

### 1.2 최근 변화: foundation feature + inference rule 중심
- foundation model 기반에서 성능 상한이 크게 상승했다.
- VisionAD(검색 기반), FoundAD(사영 기반), D2-4FAD(가중 기반)는 서로 다르지만 공통적으로 "support를 어떻게 쓰는가"를 주요 설계 축으로 둔다.

### 1.3 보수적 정리(과단정 방지)
- "representation이 더 이상 중요하지 않다"가 아니라,
  - "강한 foundation feature를 전제로 할 때, 추가 성능 차이를 만드는 1차 설계 축이 support utilization일 가능성이 크다"로 서술한다.
- 논문 간 프로토콜 차이(shot/split/seed/backbone)가 커 직접 수치 비교는 제한적임을 명시한다.

---

## 2. Core Claim and Falsifiability

### 2.1 Main Claim
In foundation-model-based FSAD, performance bottleneck shifts from feature extraction quality to support memory organization and utilization.

### 2.2 Falsifiable Sub-claims
- C1. 같은 backbone에서 memory organization만 바꿔도 유의미한 성능 차이가 발생한다.
- C2. dominant-only compression은 false positive를 줄이지만 rare normal 누락으로 recall이 깨질 수 있다.
- C3. dominant + residual 분리 구조는 C2의 trade-off를 완화한다.
- C4. 개선 폭은 저-shot(1/2-shot)에서 더 크게 나타난다.

### 2.3 Boundary (성립/비성립 조건)
- 성립 가능 조건:
  - support 수가 매우 적고 intra-class variation이 높은 카테고리
  - anomaly가 미세하여 nearest-neighbor 단일 거리로 혼동이 잦은 경우
- 약화/실패 가능 조건:
  - support가 충분히 많은 고-shot 영역
  - 카테고리 변동이 작아 flat memory도 충분한 경우
  - backbone 자체가 도메인 mismatch로 약한 경우(이 경우 representation bottleneck 재부상)

---

## 3. Positioning vs Prior Work (압축 버전)

### 3.1 PatchCore / GraphCore
- 공통점: support patch를 retrieval 대상으로 사용.
- 차이점: 본 방향은 retrieval 규칙보다 memory의 내부 구조화(반복/희소 분리)를 1차 설계 대상으로 둔다.

### 3.2 VisionAD / Search-centric methods
- 공통점: support coverage 문제를 핵심으로 본다.
- 차이점: VisionAD는 augmentation 기반 양적 확장, 본 방향은 구조적 재조직(organization)으로 품질 개선.

### 3.3 FoundAD
- 공통점: 강한 foundation feature 전제.
- 차이점: FoundAD는 projection으로 support를 implicit 사용, 본 방향은 support 내부 관계를 explicit하게 모델링.

### 3.4 D2-4FAD
- 공통점: support를 query-conditional하게 다룬다.
- 차이점: D2-4FAD는 support 단위 가중합, 본 방향은 support 내부 공통성/예외성 구조를 먼저 정의하고 이후 매칭.

### 3.5 CIF (Commonality in Few)
- 공통점: support 구조성에 주목.
- 차이점: CIF는 commonality 강화 중심, 본 방향은 commonality와 rare-normal preservation의 균형을 핵심 문제로 둔다.

---

## 4. Method Blueprint (구현 가능한 수준)

### 4.1 Notation
- support 이미지 집합: S = {I_1, ..., I_k}, k ∈ {1,2,4,8}
- backbone feature extractor: φ (frozen foundation model, e.g., DINOv2-Reg ViT-L/14)
- support patch feature set: X = {x_n}_{n=1..N}, x_n ∈ R^d (d=1024 for ViT-L, N≈196k for 14×14 grid)
- query patch feature: q_t ∈ R^d
- multi-layer feature: layers [4,6,8,...,L] 결합 (VisionAD/FoundAD에서 검증된 중간~후반 layer 활용)

### 4.2 Memory Decomposition

**목표**: X를 dominant memory M_d와 residual memory M_r로 분해

**M_d**: 반복 빈도와 응집도가 높은 정상 패턴. Gaussian component로 파라미터화 → Mahalanobis scoring 가능.
**M_r**: M_d로 잘 설명되지 않지만 정상으로 유지해야 하는 희소 패턴. Raw patch 보존 → NN scoring.

#### 4.2.1 Decomposition 방법 선택 근거

| 방법 | 장점 | 단점 | Few-shot 적합성 |
|------|------|------|----------------|
| K-means | 단순, 빠름 | K 사전 결정 필요, 구형 가정 | △ (K 선택 불안정) |
| **DPMM (truncated)** | K 자동 결정, weight spectrum 제공 | 구현 복잡 | **◎** (MICCAI 2025에서 DINOv2에 검증) |
| RPCA (low-rank+sparse) | 이론적 정당성, clustering과 독립 | 해석이 다름 | ○ (cross-validation용) |

- **Primary**: Truncated DPMM with diagonal covariance (Ref: MICCAI 2025, DINOv2 patch features에 적용 사례)
  - 초기 K_max=50 (few-shot scale), pruning → 실제 5-15 component 생존
  - component weight w_c가 자연적으로 dominant/rare spectrum을 제공
- **Fallback/Ablation**: K-means (K = sqrt(N/2), 경험적) + frequency threshold
- **Cross-validation**: RPCA로 low-rank(dominant) + sparse(residual) 분해 → DPMM 결과와 비교

#### 4.2.2 Dominant/Residual 분류 기준

DPMM component c = (μ_c, Σ_c, w_c) 에서:
- **Dominant 조건**: w_c ≥ τ_freq (e.g., τ_freq = 1/(2K_surviving))
- **Residual**: w_c < τ_freq인 component의 멤버 patch들을 raw로 보존
- τ_freq는 "uniform 대비 절반 이하 빈도"라는 해석 가능한 기준

### 4.3 Residual Memory Filtering (rare-normal vs noise 구분)

**핵심 문제**: residual에 포함된 patch가 "rare-but-valid normal"인지 "noise/artifact"인지 구분

#### Multi-criteria filtering (3개 독립 기준의 conjunction):

1. **Global consistency (Conformal p-value)**
   - 각 residual patch r에 대해, 전체 support X에서의 NCM(Non-Conformity Measure) score 계산
   - p_r = |{x ∈ X : NCM(x) ≥ NCM(r)}| / |X|
   - **보존 조건**: p_r > ε (e.g., ε = 0.02) → 전체 support 분포에서 극단적이지 않음
   - Ref: Conformal Anomaly Detection (Laxhammar), distribution-free guarantee

2. **Local consistency (LOF)**
   - LOF(r) = local reachability density ratio with k=min(20, N/10) neighbors
   - **보존 조건**: LOF(r) < τ_lof (e.g., τ_lof = 2.0) → 지역적으로 고립되지 않음
   - 주의: 1-shot에서는 local structure가 약함 → 2-shot 이상에서 활성화
   - Ref: Breunig et al. (2000)

3. **Coverage necessity**
   - d_dominant(r) = min_c d_Maha(r, μ_c, Σ_c) for dominant components
   - **보존 조건**: d_dominant(r) > δ → M_d로 이미 커버되는 patch는 residual에 불필요
   - δ = median(d_dominant(x) for x in M_d members) + MAD (robust threshold)

#### 결과: M_r = {r : p_r > ε AND LOF(r) < τ_lof AND d_dominant(r) > δ}

→ "globally not extreme, locally not isolated, not already covered by dominants"

### 4.4 Dual-Path Anomaly Scoring

#### 4.4.1 Dominant path score (distributional)

Mahalanobis distance with Ledoit-Wolf shrinkage covariance:

```
Σ̂_c = (1-γ)·Σ_sample_c + γ·(tr(Σ_sample_c)/d)·I    (Ledoit-Wolf shrinkage)
s_d(q_t) = min_c { (q_t - μ_c)^T Σ̂_c^{-1} (q_t - μ_c) }
```

- Ledoit-Wolf shrinkage: few-shot에서 sample covariance가 rank-deficient → identity 방향 수축으로 안정화
- scikit-learn에 구현 있음 (LedoitWolf estimator)
- Ref: PaDiM (Defard et al., ICPR 2020) — position-wise Gaussian with regularized covariance

#### 4.4.2 Residual path score (retrieval)

```
s_r(q_t) = min_{r ∈ M_r} ||q_t - r||_2
```

- Residual은 parametric modeling 불가 (sample 부족) → 단순 NN distance 유지

#### 4.4.3 Score combination: Product gating

```
a_t = s_d(q_t) · g(s_r(q_t))
where g(s_r) = 1 - exp(-s_r / σ_r)
σ_r = median(s_r(x) for x ∈ X)    (auto-calibration)
```

**해석**:
- s_r이 작으면 (query가 residual memory에 가까우면) → g ≈ 0 → anomaly score 억제 (rare normal로 판단)
- s_r이 크면 (residual에서도 먼) → g ≈ 1 → s_d가 그대로 anomaly score
- σ_r은 support 내부에서 자동 보정 → hyperparameter-free

**대안 (ablation용)**: Energy-based unified score
```
a_t = -T · log( Σ_c w_c · exp(-d_Maha(q_t, c)/T) + Σ_r w_r · exp(-||q_t - r||²/T) )
```
- T: temperature (T→0 = hard NN, T→∞ = mean distance)
- Mixture density 해석: dominant과 residual을 하나의 mixture로 통합
- Ref: Liu et al. (NeurIPS 2020) Energy-based OOD detection

#### 4.4.4 Score calibration
- Per-category z-score normalization: z_t = (a_t - μ_support) / σ_support
- μ_support, σ_support는 support patch들의 self-score로 추정
- Modified z-score (median + MAD) for robustness
- Ref: UniAD (You et al., NeurIPS 2022)

### 4.5 Image-Level Aggregation

```
S_image = (1/P) · Σ_{t ∈ top-P} a_t
```
- P = top-K% patches (e.g., K=1% for MVTec, tunable)
- 대안: quantile pooling (e.g., 99th percentile)

### 4.6 Pixel-Level Localization

- Patch-level score map → bilinear interpolation to original resolution
- Optional Gaussian smoothing (σ_smooth = 4) for clean heatmap
- Ref: PatchCore, VisionAD의 standard practice

### 4.7 Design Principles (강화)
- **P1**: dominant는 distributional modeling(Mahalanobis), residual은 instance retrieval(NN). 두 경로의 수학적 성격이 다름 → 각각에 적합한 scoring.
- **P2**: residual filtering의 3 기준은 모두 support 내부에서 계산 가능 → 외부 데이터 불필요.
- **P3**: backbone 독립 plug-in 모듈. VisionAD의 augmented memory, FoundAD의 projected features 위에도 적용 가능.
- **P4**: DPMM의 component weight가 dominant/residual 분류의 유일한 pivot → 해석 가능하고 ablation 용이.

### 4.8 Computational Complexity
- Feature extraction: O(N·d) per image (backbone forward pass, frozen)
- DPMM fitting: O(N·K·d·I) where I=iterations (N≈200-1600, K≈50, d≈1024, I≈100) → < 1s per category on GPU
- Mahalanobis scoring: O(|M_d|·d²) per query patch (diagonal Σ → O(|M_d|·d))
- NN scoring: O(|M_r|·d) per query patch
- **Total inference**: backbone forward + scoring → comparable to PatchCore, faster than FoundAD (no projector forward)

### 4.9 Config Toggles (ablation-friendly)
```yaml
MEMORY:
  enable_pattern_split: true/false         # [ABLATION] 전체 on/off
  enable_residual: true/false              # [ABLATION] residual 경로 on/off
  decomposition_method: dpmm/kmeans/rpca   # [ARCH] 분해 방법
  dpmm:
    K_max: 50                              # [TUNE] 초기 최대 component 수
    diagonal_cov: true                     # [ARCH] 대각 공분산 사용
  residual_filter:
    conformal_eps: 0.02                    # [TUNE] conformal p-value threshold
    lof_tau: 2.0                           # [TUNE] LOF threshold
    min_shot_for_lof: 2                    # [ARCH] LOF 활성화 최소 shot 수
SCORE:
  dominant_method: mahalanobis             # [ARCH] mahalanobis/cosine/l2
  combination: product_gating/energy       # [ABLATION] 결합 방식
  covariance_shrinkage: ledoit_wolf        # [ARCH] 공분산 추정 방식
  calibration: modified_zscore             # [ARCH] 점수 보정 방식
  energy_temperature: 1.0                  # [TUNE] energy 방식 사용 시
AGGREGATE:
  method: top_p_mean                       # [ARCH]
  top_p_ratio: 0.01                        # [TUNE]
```

### 4.10 Technical References Summary
| 기술 | 역할 | 핵심 참고문헌 |
|------|------|-------------|
| Truncated DPMM | Memory decomposition | MICCAI 2025 (DINOv2 patch features) |
| Ledoit-Wolf shrinkage | Few-shot covariance estimation | Ledoit & Wolf (2004); PaDiM (ICPR 2020) |
| Mahalanobis distance | Dominant path scoring | Lee et al. (NeurIPS 2018) |
| Conformal p-value | Residual filtering (global) | Laxhammar (2014); Ishimtsev et al. (2017) |
| LOF | Residual filtering (local) | Breunig et al. (2000) |
| Product gating | Score combination | Novel formulation (dual-path specific) |
| Energy-based score | Ablation alternative | Liu et al. (NeurIPS 2020) |
| Modified z-score | Score calibration | Iglewicz & Hoaglin (1993); UniAD (NeurIPS 2022) |
| RPCA | Cross-validation decomposition | Candès et al. (2011) |

---

## 5. Experiment Plan (Claim-Evidence 매핑)

### 5.1 Internal Baselines (same backbone, ablation)
- **B0**: Raw flat memory + NN scoring (PatchCore-style)
- **B1**: Dominant-only memory + Mahalanobis scoring (no residual)
- **B2**: Dominant + Residual + product gating (proposed full)
- **B2-e**: Dominant + Residual + energy-based score (ablation: combination 방식 비교)
- **B2-nf**: Dominant + Residual without residual filtering (filtering 효과 분리)

### 5.2 External References (positioning용, 직접 재현 또는 보고 수치)
- **VisionAD**: training-free SOTA (DINOv2-Reg + augmentation + NN)
- **FoundAD**: manifold projection (ICLR 2026)
- **PatchCore**: standard baseline
- **GraphCore**: GNN-enhanced baseline
- 모든 external은 동일 backbone에서 재현하여 공정 비교 (가능한 경우)

### 5.3 Core Tests

**T1 (C1: memory organization matters)**
- 설계: B0 vs B1 vs B2, 동일 backbone(DINOv2-Reg ViT-L), 동일 augmentation, shot={1,2,4,8}
- MVTec-AD 15 categories + VisA 12 categories
- 기대: B2 > B1 > B0 (특히 I-AUROC, PRO에서)
- 보고: mean±std (3 seeds), category-wise delta 분포

**T2 (C2/C3: dominant-only의 rare normal 손실 + residual의 복원)**
- 설계: MVTec-AD 카테고리를 normal variation 기준으로 2군 분류
  - **High-variation**: screw, transistor, cable 등 (support 내 정상 patch 다양성 높음)
  - **Low-variation**: bottle, capsule 등 (support 내 균일)
- Normal variation proxy: support patch feature의 average pairwise cosine distance
- 비교: B1 vs B2에서 high-variation 카테고리의 FPR@95TPR 변화
- 기대: B1은 high-variation에서 FN 증가, B2가 이를 회복

**T3 (C4: low-shot에서 효과 증폭)**
- 설계: shot={1,2,4,8}에서 B0→B2 delta 추이
- 기대: 1-shot delta > 4-shot delta > 8-shot delta (support가 적을수록 구조화 이득 큼)
- 보고: delta vs shot curve (figure)

**T4 (Boundary: 실패 조건 탐색)**
- 설계: B2가 B0보다 열등한 카테고리 식별 및 원인 분석
- 예상 실패 조건:
  - 카테고리 variation이 극히 작아 flat memory로 충분한 경우
  - Support 자체에 noise/misalignment이 있는 경우
- 보고: category-wise scatter (x=normal variation, y=B2-B0 delta)

**T5 (Ablation: 개별 모듈 기여 분리)**
| 실험 | DPMM | Residual | Filtering | Mahalanobis | Product gating |
|------|------|----------|-----------|-------------|----------------|
| B0   | ✗    | ✗        | ✗         | ✗           | ✗              |
| B1   | ✓    | ✗        | ✗         | ✓           | ✗              |
| B2-nf| ✓    | ✓        | ✗         | ✓           | ✓              |
| B2   | ✓    | ✓        | ✓         | ✓           | ✓              |
| B2-e | ✓    | ✓        | ✓         | (energy)    | (energy)       |

**T6 (Cross-backbone robustness)**
- Backbone: DINOv2-Reg ViT-L, DINOv2 ViT-B, CLIP ViT-L, WideResNet-50
- 동일 method(B2) 적용, backbone별 성능 변동 관찰
- 목적: "backbone 바꾸면 효과 사라지지 않는다" 증명

**T7 (Decomposition method ablation)**
- DPMM vs K-means vs RPCA for memory decomposition
- 동일 scoring, 동일 backbone에서 분해 방법만 교체

### 5.4 Metrics and Reporting
- **Primary**: Image AUROC, Pixel AUROC, PRO
- **Secondary**: FPR@95TPR (rare-normal 분석용), AUPR
- **Statistical**: mean ± std over 3 seeds, paired t-test for significance
- **Category-wise**: delta 분포 (violin plot), worst-category 성능
- **Reproducibility**: config diff, seed, support image indices, log path

### 5.5 Evidence Table
| Claim | Test | Expected Signal | Failure Interpretation |
|---|---|---|---|
| C1 | T1 | B2 > B0 on mean AUROC/PRO | 구조화 이득 미약 → backbone이 여전히 1차 변수 |
| C2 | T2 | B1의 high-var FPR@95TPR 악화 | dominant-only가 rare normal 손실 유발 |
| C3 | T2 | B2가 B1 대비 high-var FN 회복 | residual 경로의 rare normal 복원 효과 |
| C4 | T3 | 1-shot delta > 4-shot delta | low-shot에서 구조화 효과 강화 |
| — | T5 | 각 모듈 제거 시 성능 하락 | one-factor ablation으로 기여 분리 |
| — | T6 | cross-backbone에서 B2>B0 유지 | backbone 독립적 principle |
| — | T7 | DPMM ≥ K-means > RPCA | 분해 방법 선택의 영향 |

### 5.6 Figures/Tables 계획 (논문용)
- **Table 1**: Main results — B0/B1/B2 vs external baselines (MVTec-AD, VisA, shot={1,2,4,8})
- **Table 2**: Ablation — T5 결과표
- **Table 3**: Cross-backbone — T6 결과표
- **Figure 1**: Method overview (pipeline diagram)
- **Figure 2**: Motivation — support 내 dominant/rare 패턴 시각화 (t-SNE + 대응 image patch)
- **Figure 3**: Shot vs delta curve (T3)
- **Figure 4**: Category-wise scatter (T4, x=variation, y=delta)
- **Figure 5**: Qualitative — anomaly heatmap 비교 (B0 vs B1 vs B2, rare-normal 영역 강조)

---

## 6. Risks, Counterarguments, and Controls

### 6.1 예상 반론과 구체적 대응

**R1: "그냥 clustering + memory bank 변형 아닌가?"**
- 대응: novelty는 clustering 알고리즘이 아니라 **dominant-residual 분리 원리 + 이중 경로 scoring**에 있음
- 증거: T5 ablation에서 분해 방법(DPMM/K-means/RPCA) 교체해도 dual-path scoring의 이득이 유지되면, 원리가 알고리즘 선택보다 중요함을 보임
- 서술: "We contribute a memory organization *principle*, not a specific clustering algorithm"

**R2: "backbone 바꾸면 효과 사라지는 것 아닌가?"**
- 대응: T6 (cross-backbone test)에서 DINOv2, CLIP, WRN-50 모두에서 B2>B0 유지를 보임
- 실패 시: backbone 의존성을 honestly 보고하고 boundary condition으로 명시

**R3: "rare normal 보존은 noise 보존과 구분이 어려움"**
- 대응: multi-criteria residual filtering (conformal + LOF + coverage necessity)으로 원리적 구분
- 추가 실험: **Contamination stress test** — support에 의도적으로 noisy patch 삽입 후 filtering 효과 측정
  - filtering 있을 때: noisy patch 제거됨 (conformal p-value < ε 또는 LOF > τ)
  - filtering 없을 때: noise가 residual에 잔류 → FP 증가
- B2-nf vs B2 비교로 filtering의 noise 억제 효과를 정량화

**R4: "VisionAD보다 성능이 낮으면?"**
- 대응 1: VisionAD + 본 method(plug-in) 조합으로 상호보완 가능성 제시
- 대응 2: 성능이 동등하더라도 **support organization principle의 실증**이 기여
- 대응 3: 특정 카테고리(high-variation)에서의 우위를 보이면 conditional advantage로 주장

**R5: "DPMM fitting이 불안정하지 않은가?"**
- 대응: 3 seeds × 3 random initialization으로 안정성 보고
- Fallback: K-means 기반 분해도 동시 보고 (T7)

---

## 7. Intro/Method Writing Skeleton (바로 본문 전개 가능)

### 7.1 Introduction 4-paragraph skeleton
1) 배경: foundation feature의 급격한 향상과 FSAD baseline 상승
2) 갭: support set이 여전히 flat/summary 중심으로 사용되어 구조 정보 손실
3) 통찰: support에는 redundancy와 rarity가 공존하며 둘을 분리 취급해야 함
4) 기여:
   - support bottleneck 가설 제시
   - pattern-aware dual memory 제안(dominant + residual)
   - low-shot에서의 trade-off 완화 실증

### 7.2 Method section skeleton
- Problem setup -> Memory decomposition -> Dual-path scoring -> Complexity -> Ablation knobs

### 7.3 Main contribution line (draft)
We show that, under strong foundation features, reorganizing limited support into a dominant-residual memory yields more reliable few-shot anomaly scoring than flat support storage, especially in low-shot regimes with high normal variation.

---

## 8. Safer Wording Snippets (표현 가드레일)

- Avoid: "representation learning은 끝났다"
- Use: "in the strong-foundation regime, support utilization becomes a first-order design factor"

- Avoid: "Vision-only가 항상 VLM보다 우월하다"
- Use: "recent vision-only methods report competitive or superior performance under their reported protocols"

- Avoid: "training-free가 학습 기반보다 무조건 낫다"
- Use: "training-free baselines now set a high reference point, raising the burden of evidence for learned methods"

---

## 9. Next Execution Checklist (실행 전환용)

- [ ] B0 (PatchCore-style flat memory + NN) 최소 재현 구현
- [ ] DINOv2-Reg ViT-L feature extraction pipeline 구현
- [ ] DPMM fitting 모듈 구현 (truncated, diagonal covariance)
- [ ] Mahalanobis scoring + Ledoit-Wolf shrinkage 구현
- [ ] Residual filtering (conformal + LOF + coverage) 구현
- [ ] Product gating score combination 구현
- [ ] MVTec-AD 1-shot에서 B0 vs B1 vs B2 최초 비교 실행
- [ ] 카테고리별 normal variation proxy 계산 (avg pairwise cosine distance)
- [ ] T5 ablation table 최초 작성
- [ ] 반증 시나리오(C1~C4 실패 패턴) 관찰 및 기록

---

## 10. Internal Evidence Anchors (문장-근거 연결)

- E1 (Trend): training-free/vision-only baseline 상승
  - 근거: `2026-03-10_comprehensive_survey.md`의 VisionAD/FoundAD 정리와 trend 요약 문단
- E2 (Support bottleneck hypothesis): support organization 원리 부족
  - 근거: `2026-03-10_research_gaps.md` 2.1 절(지원 조직 원리 부족, VisionAD/FoundAD/D2-4FAD 비교)
- E3 (Coverage vs structure): augmentation은 coverage 확장이나 memory 구조화와는 별개
  - 근거: `papers/search_is_all_you_need.md`의 augmentation/search-centric 요약
- E4 (Implicit vs explicit support use): projection 기반은 support 내부 관계를 직접 모델링하지 않음
  - 근거: `papers/foundation_visual_encoders.md` + 갭 노트의 해석 문단
- E5 (Query-conditional weighting limits): support relevance weighting은 있으나 support 간 구조 모델링은 별개 이슈
  - 근거: `papers/dual_distillation.md`의 learn-to-weight 요약

---

## 11. Writing QA Checklist (제출용 품질 게이트)

- Q1. 헤더-클레임 정합성: 각 주요 섹션 제목이 검증 가능한 주장 형태인가?
- Q2. Claim-Evidence Traceability: C1~C4 각각에 대응하는 실험(T1~T4)과 표/그림이 매핑되는가?
- Q3. 과장 방지: "always", "state-of-the-art" 같은 단정 표현 없이 protocol 조건을 명시했는가?
- Q4. 재현성 최소 요건: seed/split/config diff/log path를 결과와 함께 기록하는가?
- Q5. 한계 공개: 실패/약화 조건(Boundary)을 본문에서 명시하고 해석했는가?

---

## 관련 노트

### 서베이 및 갭 분석
- [FSAD 종합 서베이](./2026-03-10_comprehensive_survey.md)
- [FSAD Research Gaps](./2026-03-10_research_gaps.md)

### 핵심 비교 논문
- [PatchCore](./papers/patchcore.md) — memory bank baseline
- [GraphCore](./papers/graphcore.md) — GNN-enhanced memory
- [VisionAD / Search is All You Need](./papers/search_is_all_you_need.md) — training-free SOTA
- [FoundAD](./papers/foundation_visual_encoders.md) — manifold projection
- [D2-4FAD](./papers/dual_distillation.md) — query-conditional support calibration
- [CIF](./papers/commonality_in_few.md) — hypergraph commonality
- [UniVAD](./papers/univad.md) — training-free unified AD
- [UniADC](./papers/uniadc.md) — detection + classification 통합

### 방법론 구체화 리서치
- [Memory Decomposition 기술 조사](./2026-03-10_decomposition_techniques_research.md)
- [Anomaly Scoring 수식 조사](./2026-03-10_anomaly_scoring_formulations.md)
- [Support Enrichment 기술 조사](./2026-03-10_support_enrichment_techniques.md)
- [Rare-Normal vs Anomaly 경계 조사](./2026-03-10_rare_normal_vs_anomaly_literature.md)
