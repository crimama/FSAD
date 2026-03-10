# Support Feature Decomposition Techniques for FSAD: Technical Research

> 작성일: 2026-03-10
> 목적: "dominant normal patterns" vs "rare-but-valid normal residuals" 분해를 위한 기존 기법 조사
> 관련 문서: paper_outline_support_memory.md 4.2-4.3절의 구현 선택지 구체화

---

## 1. Feature Clustering for Anomaly Detection

### 1.1 K-Means

**핵심 아이디어**: N개 patch feature를 K개 centroid로 압축. 각 centroid가 하나의 "dominant normal pattern".

**Few-shot 적합성**:
- 1-shot(~196 patches for 14x14 grid) ~ 8-shot(~1568 patches): K-means 작동 가능 범위
- K 선택이 핵심 문제. Elbow/silhouette는 소표본에서 불안정
- **실용적 가이드**: K = sqrt(N/2) 또는 K = 5~20 범위에서 grid search

**장점**: 구현 단순, 빠름 (sklearn으로 수 ms), 해석 용이
**단점**: 구형(spherical) 클러스터 가정, 초기화 민감, outlier에 취약
**추천 변형**: K-Means++ 초기화 필수. Mini-batch 불필요 (N이 충분히 작음)

**Anomaly detection 적용 사례**:
- PatchCore 계열: 직접적 clustering은 아니지만 coreset이 유사한 역할
- GraphCore (ICLR 2023): VIIF + memory bank에서 redundancy 제거에 활용

### 1.2 Gaussian Mixture Model (GMM)

**핵심 아이디어**: K개 Gaussian component의 혼합으로 모델링. 각 component가 하나의 normal pattern mode.

**Few-shot 적합성**:
- **핵심 문제**: d차원 full covariance에 O(d^2) 파라미터 → 768-d에서 ~295K params/component
- 1-shot 196 patches로 full-cov GMM 학습은 수학적으로 불가능 (rank-deficient)
- **해결책**:
  - Diagonal covariance: O(d) params → 768 params/component, 실용적
  - Tied covariance: 모든 component가 같은 공분산 공유
  - PCA 차원 축소 후 GMM (e.g., 768→64 then GMM)
  - Factor analysis 변형: low-rank + diagonal

**K 선택**:
- BIC/AIC: 소표본 고차원에서 신뢰성 낮음
- **Bayesian 대안 (DPMM)**: 아래 1.4절 참조 - 자동으로 K 결정
- 실용적 범위: K = 3~10 (few-shot에서)

**AD 적용 논문**:
- **PaDiM** (ICPR 2021): 각 spatial position마다 multivariate Gaussian 1개 → position-wise modeling. Few-shot 확장 시 공분산 추정이 불안정해지는 한계.
- **HGAD** (ECCV 2024): Hierarchical GMM + Normalizing Flow. Inter-class GMM으로 multi-class 분리, intra-class mixed centers로 세분화. 단, unified AD 세팅이라 few-shot과는 다른 맥락.
- **MGAD** (2024): Multi-layer Gaussian discriminant + GMM으로 pixel-wise density estimation. MVTec 98.8% AUROC. Full-shot 세팅.

**계산 비용**: sklearn GMM으로 수십 ms (diagonal cov, K≤10, N≤2000)

### 1.3 DBSCAN / Density-Based Clustering

**핵심 아이디어**: 밀도 기반으로 클러스터 형성. Low-density points를 자동으로 noise/outlier로 분류.

**Few-shot 적합성**:
- eps, min_samples 하이퍼파라미터에 매우 민감
- 고차원(768-d)에서 distance 분포가 집중(concentration of measure) → eps 설정 극히 어려움
- **치명적 문제**: 소표본에서 대부분의 점이 noise로 분류될 위험

**장점**: K 사전 지정 불필요, 비구형 클러스터 발견 가능, outlier 자동 분리
**단점**: 고차원 + 소표본 조합에서 실질적으로 작동하기 어려움

**판단**: Few-shot FSAD에는 **비추천**. 768-d에서 density estimation 자체가 curse of dimensionality에 걸림.

### 1.4 Dirichlet Process Mixture Model (DPMM) — 가장 유망

**핵심 아이디어**: Bayesian nonparametric GMM. Component 수를 데이터가 자동 결정.

**핵심 논문: Anomaly Detection by Clustering DINO (MICCAI 2025)**:
- DINOv2 patch embedding에 truncated DPMM (Gaussian components) 적용
- 초기 K=500, 학습 후 120~150개 component 생존 (자동 pruning)
- Diagonal covariance 사용 → 고차원에서도 안정적
- Batched EM: batch 12 images, discount factor 0.2, 40 epochs
- **Scoring**: cosine similarity to nearest surviving component mean
  - s(y) = max_{k: π_k > 1e-6} [cos(y, μ_k)]
- **성능**: BraTS 96.21% AUROC (AnomalyDINO full-shot 97.71% 대비 -1.5pp)
- **속도**: 35-58ms/sample (AnomalyDINO 422-984ms 대비 10-20x 빠름)
- **메모리**: 1.5-2.0 GB (AnomalyDINO 6.9-38.1 GB 대비 극적 절감)

**Few-shot 적용 시 고려사항**:
- 원 논문은 medical imaging full-shot 세팅
- Few-shot(~200-1600 patches)에서는 truncation을 K=50~100으로 낮추면 충분할 것
- Bayesian prior가 overfitting을 자연스럽게 억제 → few-shot에 유리
- Component weight가 자연스럽게 "dominant" (높은 π_k) vs "rare" (낮은 π_k) 분리 제공

**본 연구와의 직접적 연결**:
- π_k가 큰 component → dominant memory M_d의 centroid
- π_k가 작지만 생존한 component → rare-but-valid pattern의 후보
- **이것이 dominant/residual 분해의 가장 원리적인(principled) 방법**

### 1.5 Spectral Clustering

**핵심 아이디어**: Affinity matrix의 eigenvector로 비선형 클러스터 구조 발견.

**Few-shot 적합성**:
- N×N affinity matrix 구축 필요 → N=1568 (8-shot)에서 O(N^2) = ~2.5M
- Eigendecomposition O(N^3) → 수 초 소요 가능
- K 사전 지정 필요 (eigengap heuristic 가능하나 소표본에서 불안정)

**장점**: 비구형 클러스터, manifold 구조 포착
**단점**: K-means 대비 느림, K 선택 여전히 필요, few-shot 규모에서 과잉

**판단**: Few-shot에서는 K-means나 DPMM 대비 이점이 약하고 비용만 증가.

---

## 2. Coreset Selection vs Full Memory

### 2.1 PatchCore Greedy Coreset (Baseline)

**방법**: Greedy furthest-point sampling으로 memory bank 크기를 1-10%로 축소.
- 핵심: coverage 최대화 (minimax facility location)
- 시간 복잡도: O(N × |coreset|)

**Few-shot 한계**:
- Full-shot(~수만 patches)에서 설계된 방법
- Few-shot(~200-1600 patches)에서는 압축 자체가 불필요하거나 오히려 정보 손실
- 1-shot에서 10% coreset = ~20 patches → 극단적 정보 손실

### 2.2 GraphCore VIIF (ICLR 2023)

**방법**: Visual Isometric Invariant Feature로 rotation-invariant 표현 → memory 중복 제거
- MVTec 1-shot: +5.8% AUC, 4-shot: +3.4% AUC
- MPDD 1-shot: +25.5% AUC
- Feature-level invariance로 effective memory 크기를 줄이면서도 coverage 유지

**본 연구와의 관계**: Feature transformation으로 redundancy 줄이는 것과, structure-aware decomposition은 직교적(orthogonal) 접근. 병합 가능.

### 2.3 Few-Shot에서의 권장 전략

**핵심 통찰: Few-shot에서는 coreset "selection"보다 coreset "organization"이 더 중요하다.**

- N이 이미 작으므로 greedy coreset의 compression 이점이 희박
- 대신 N개 patch를 **구조화** (dominant/residual 분류)하는 것이 정보 활용도를 높임
- 구체적 대안:
  1. **Full memory + structured index**: 모든 patch 보존, 검색 시 dominant/residual 경로 분리
  2. **Prototype compression**: dominant cluster의 centroid만 저장 + residual은 원본 보존
  3. **DPMM compression**: component mean이 자연스러운 coreset 역할 (120-150개 prototype)

---

## 3. GMM for Normal Distribution Modeling (상세)

### 3.1 Position-wise Gaussian (PaDiM 스타일)

- 각 spatial position (i,j)마다 독립 Gaussian N(μ_{ij}, Σ_{ij})
- 장점: position alignment가 자연스러움, Mahalanobis distance로 scoring
- Few-shot 한계: position당 k개 샘플로 d-dim Gaussian → k=4, d=768이면 공분산 rank ≤ 3
- **대응**: PCA로 d=100~200 축소 후 적용 (PaDiM 원 논문도 random dimension selection 사용)

### 3.2 Global GMM (Position-agnostic)

- 모든 patch를 하나의 pool에 넣고 GMM fitting
- 장점: 더 많은 샘플 확보 (N = k × H × W), position invariance
- 단점: spatial context 손실
- Few-shot 권장 component 수: K = 5~15 (diagonal cov 기준)

### 3.3 Hybrid: Clustered Positions

- 먼저 position을 의미적으로 clustering (e.g., object vs background)
- 각 position cluster 내에서 Gaussian fitting
- AnomalyDINO의 foreground masking이 이 방향의 단순 버전

### 3.4 Few-Shot Regime에서의 실용적 가이드라인

| Setting | Patches (N) | Max GMM Components (diagonal) | 권장 방법 |
|---------|-------------|-------------------------------|-----------|
| 1-shot | ~196 | 3-5 | Single Gaussian or K=3 GMM |
| 2-shot | ~392 | 5-8 | K=5 GMM or DPMM(trunc=30) |
| 4-shot | ~784 | 8-15 | DPMM(trunc=50) 권장 |
| 8-shot | ~1568 | 10-20 | DPMM(trunc=100) 권장 |

*(14×14 DINOv2 patch grid 가정, 448×448 입력)*

---

## 4. Prototype Networks and Support Organization

### 4.1 Classic Prototypical Networks (Snell et al., 2017)

**방법**: 각 class의 support features를 평균(mean)하여 single prototype 생성.
- Query와 prototype 간 Euclidean distance로 분류

**FSAD 적용 시 한계**:
- Normal class 하나뿐 → single prototype이 intra-class variation을 압축 소멸
- Mean prototype은 dominant pattern만 반영, rare variation은 소실

### 4.2 Multi-Prototype Extensions

**방법**: 하나의 class에 여러 prototype 할당
- Clustering-based: K-means centroid를 prototype으로 사용
- Attention-based: query-conditional prototype selection

**FSAD 관련 논문**:
- **D2-4FAD**: support image별 가중치를 query-conditional하게 학습 → support-level prototype weighting
  - 한계: support 내부의 patch-level 구조는 모델링하지 않음
- **CIF (Commonality in Few, 2025)**: Hypergraph-enhanced memory로 support 간 공통성(commonality) 강조
  - 한계: commonality 강화는 dominant 패턴 중심 → rare normal 보존 미고려

### 4.3 Matching Networks 스타일

**방법**: Prototype 없이 support set 전체와 soft attention matching
- FSAD에서는 PatchCore-style nearest neighbor가 이 방향의 가장 단순한 형태
- AnomalyDINO: patch-level NN matching, foreground masking으로 support 정제

### 4.4 Support Organization의 미탐색 영역 (= 본 연구의 위치)

| 방법 | Support 구조화 수준 | Dominant/Rare 분리 |
|------|---------------------|-------------------|
| PatchCore | Flat (coreset) | X |
| AnomalyDINO | Foreground mask만 | X |
| VisionAD | Augmentation으로 양적 확장 | X |
| D2-4FAD | Support image-level weighting | X |
| CIF | Commonality 강조 | 부분적 (dominant 편향) |
| **Proposed** | **Patch-level clustering + dual memory** | **O** |

---

## 5. Density-Based Outlier Detection Within Normal Features

### 5.1 문제 정의

Support set 내부에서:
- **Dominant**: 반복적으로 나타나는 정상 패턴 (e.g., 균일한 텍스처, 반복 구조)
- **Rare-but-valid**: 드물지만 정상인 패턴 (e.g., 라벨, 나사 구멍, 제품 가장자리)
- **Noise**: 추출 아티팩트, 배경 잡음

목표: Rare-but-valid를 noise와 구분하여 보존

### 5.2 Local Outlier Factor (LOF)

**방법**: 각 점의 local density를 k-nearest neighbors 대비 비교
- LOF > 1: 주변보다 sparse → potential outlier
- LOF ≈ 1: 주변과 비슷한 density → inlier

**Few-shot 적용**:
- k=10~20 설정 시 N=196(1-shot)에서도 작동
- LOF가 높은 patch = rare pattern 후보
- **문제**: rare-but-valid와 noise를 추가 기준 없이 구분 불가

### 5.3 Cluster-Based Residual Detection (본 연구 방향과 직결)

**제안 프레임워크**:
1. K-means/DPMM으로 dominant clusters 형성
2. 각 patch의 nearest cluster center까지 거리 계산
3. 거리가 threshold 초과인 patch를 residual pool에 배정
4. Residual pool 내에서 추가 필터링:
   - Mutual kNN: residual끼리도 가까운 이웃이 있으면 → rare-but-valid
   - Isolated points (kNN distance 극히 큼) → noise 후보로 제거

**Threshold 설정**:
- Median Absolute Deviation (MAD) 기반: robust to outliers
- distance > median + 2*MAD → residual 후보
- 또는: cluster 내 거리 분포의 95th percentile

### 5.4 Robust PCA 접근

**방법**: Feature matrix X ∈ R^{N×d}를 low-rank L + sparse S로 분해
- L: dominant normal subspace (low-rank → 반복 패턴)
- S: sparse deviations (rare patterns + noise)

**FSAD 적용 가능성**:
- X = L + S 분해 후:
  - L의 column space → dominant memory의 basis
  - S의 nonzero rows → residual candidates
- RPCA는 convex optimization으로 풀 수 있어 안정적
- Few-shot에서 N < d일 경우에도 잘 작동 (underdetermined regime에서 sparse recovery 유효)

**계산 비용**: Augmented Lagrange Multiplier로 O(N × d × rank(L)) → ms 단위

**주의**: RPCA의 "sparse"는 element-wise sparsity → patch-wise rarity와 정확히 대응하지는 않음. Row-sparse RPCA 변형이 더 적합할 수 있음.

### 5.5 Dictionary Learning

**핵심 논문: Patchwise Sparse Dictionary Learning (IEEE 2022)**:
- Pretrained CNN activation map에서 dictionary 학습
- 각 patch를 dictionary atoms의 sparse combination으로 표현
- Reconstruction error가 높은 patch = anomaly

**Dominant/Residual 분리와의 연결**:
- Dictionary atoms = dominant normal patterns
- 잘 복원되는 patch → dominant에 속함
- 복원 잘 안 되지만 정상인 patch → residual (atoms 조합으로 표현 안 되는 valid variation)

**Few-shot 한계**:
- Dictionary 크기 << N이어야 의미 → 1-shot(N=196)에서 dictionary 50개 atoms은 과잉
- 4-shot 이상에서 실용적

---

## 6. 종합 비교 및 권장

### 6.1 Method Comparison Matrix

| Method | Auto K | High-dim OK | Few-shot OK | Dominant/Rare 분리 | 계산 비용 | 구현 복잡도 |
|--------|--------|-------------|-------------|-------------------|----------|------------|
| K-Means | X | △ | O | 간접적 (크기 기반) | 매우 낮음 | 낮음 |
| GMM (diag) | X | O | △ (K 선택 문제) | O (weight 기반) | 낮음 | 낮음 |
| DBSCAN | O | X | X | O (noise label) | 낮음 | 낮음 |
| **DPMM** | **O** | **O (diag)** | **O** | **O (weight 자동)** | **중간** | **중간** |
| Spectral | X | △ | △ | 간접적 | 높음 | 중간 |
| RPCA | N/A | O | O | O (L/S 분리) | 중간 | 중간 |
| Dict. Learning | X | O | △ (4+ shot) | O (recon error) | 높음 | 높음 |
| LOF | N/A | △ | O | 간접적 (score 기반) | 낮음 | 낮음 |

### 6.2 Recommended Approach for Paper

**Primary: DPMM with Diagonal Covariance (MICCAI 2025 방식 적응)**

이유:
1. Component 수 자동 결정 → K 선택 문제 제거
2. Component weight π_k가 자연스럽게 dominant(높은 π_k) vs rare(낮은 π_k but surviving) 분리 제공
3. Diagonal covariance로 고차원(768-d)에서도 안정적
4. Bayesian prior가 few-shot overfitting 억제
5. 기존 FSAD에서 DPMM을 사용한 논문이 없음 → novelty 확보
6. Component mean이 prototype 역할 → 해석 가능
7. MICCAI 2025 결과가 computational efficiency 우위를 입증

**Secondary/Ablation: K-Means + MAD-based residual detection**

이유:
1. 가장 단순한 baseline으로 ablation에 적합
2. "DPMM의 자동 K 결정이 정말 필요한가?" 질문에 답변 가능
3. 구현이 trivial하여 빠른 프로토타이핑 가능

**Optional Extension: RPCA for validation**

이유:
1. Completely different decomposition principle (subspace vs clustering)
2. "Dominant/residual 분리의 이점이 특정 clustering 방법에 의존하는가?" 검증
3. Few-shot underdetermined regime에서도 이론적으로 잘 작동

### 6.3 구현 우선순위

```
Phase 1 (MVP): K-Means + frequency/distance-based dominant/residual split
Phase 2 (Main): DPMM diagonal → weight-based dominant/residual
Phase 3 (Ablation): RPCA L+S decomposition for cross-validation
```

### 6.4 Novelty Argument 정리

기존 FSAD 방법들은 support memory를 flat structure로 취급하거나(PatchCore, AnomalyDINO), augmentation으로 양적 확장(VisionAD)하거나, projection으로 우회(FoundAD)한다. **Support 내부를 dominant/residual로 구조화하고, 이를 dual-path scoring에 반영하는 것은 탐색되지 않은 방향**이다.

DPMM을 채택할 경우 추가 novelty:
- Component weight의 연속적 스펙트럼에서 dominant/rare threshold를 학습 가능하게 설계
- Medical DPMM (MICCAI 2025)은 full-shot + single distance scoring → few-shot + dual-path scoring으로 확장

---

## 7. Open Questions for Implementation

1. **PCA 전처리 필요 여부**: DINOv2 768-d에 직접 DPMM vs PCA→128-d 후 DPMM
   - MICCAI 2025는 direct fitting (L2-norm 후 diagonal cov)
   - PCA가 rare variation을 소거할 위험 vs 안정성 향상

2. **Dominant/residual threshold**: π_k의 어떤 값을 기준으로 분리?
   - Adaptive: top-p% weight cumsum이 90%가 될 때까지 dominant, 나머지 residual
   - Fixed: π_k > 1/K_effective면 dominant

3. **Residual scoring 방식**: nearest residual patch vs residual component mean
   - Residual이 소수이므로 nearest patch가 더 안정적일 수 있음

4. **Position 정보 활용**: position-agnostic DPMM vs position-aware variant
   - Position 무시: texture 카테고리에 적합
   - Position 조건부: object 카테고리에 적합 (spatial regularity)

---

## Sources

- [AnomalyDINO (WACV 2025)](https://arxiv.org/abs/2405.14529) — DINOv2 few-shot patch NN
- [Anomaly Detection by Clustering DINO (MICCAI 2025)](https://arxiv.org/html/2509.19997) — DPMM on DINOv2 embeddings
- [HGAD (ECCV 2024)](https://arxiv.org/abs/2403.13349) — Hierarchical GMM normalizing flow
- [Few-shot Online AD](https://arxiv.org/html/2403.18201v1) — K-NG network + Mahalanobis
- [PaDiM (ICPR 2021)](https://arxiv.org/abs/2011.08785) — Position-wise multivariate Gaussian
- [GraphCore (ICLR 2023)](https://arxiv.org/abs/2301.12082) — VIIF for memory compression
- [RegAD (ECCV 2022)](https://ar5iv.labs.arxiv.org/html/2207.07361) — Registration-based FSAD
- [PromptAD (CVPR 2024)](https://arxiv.org/abs/2404.05231) — Prompt learning for FSAD
- [Patchwise Sparse Dictionary Learning (IEEE 2022)](https://ieeexplore.ieee.org/document/9956215/)
- [Dictionary Learning for AD](https://arxiv.org/abs/2201.03869) — Uniform sparse representations
- [PatchCore](https://arxiv.org/abs/2106.08265) — Greedy coreset memory bank
- [Optimizing PatchCore for Few-shot](https://arxiv.org/abs/2307.10792)
- [CIF: Commonality in Few (2025)](https://arxiv.org/html/2511.05966) — Hypergraph-enhanced memory

---

## 관련 노트

- [Paper Outline (Support Memory)](./2026-03-10_paper_outline_support_memory.md) — 방법론 초안의 4.2-4.3절
- [Research Gaps](./2026-03-10_research_gaps.md) — 2.1절 "support 조직 원리 부족"
- [Comprehensive Survey](./2026-03-10_comprehensive_survey.md)
