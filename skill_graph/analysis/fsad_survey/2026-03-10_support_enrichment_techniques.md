# Beyond Augmentation: Techniques for Enriching Few-Shot Support Representations

> 작성일: 2026-03-10
> 목적: image-level augmentation을 넘어서 support set의 feature representation을 enrichment하는 기법 서베이
> 관련: paper outline의 "dominant + residual memory decomposition"과 complementary한 기법 탐색

---

## 1. Feature Augmentation in Embedding Space

Image-level augmentation (rotation, flip, color jitter)은 pixel 공간에서 다양성을 만들지만, embedding 공간에서 직접 feature를 생성/변형하는 방법들이 few-shot learning에서 활발히 연구되었다.

### 1.1 Feature Hallucination

**핵심 아이디어**: few-shot support에서 추출한 feature vector를 기반으로 가상의(hallucinated) feature를 생성하여 support set을 확장.

| Paper | 방법 | 핵심 기법 | Sample Efficiency |
|-------|------|-----------|-------------------|
| **Tensor Feature Hallucination** (Lazarou et al., WACV 2022) | Tensor feature 생성 | Vector가 아닌 tensor(spatial structure 보존) feature를 생성하여 공간 정보 유지 | 1-shot/5-shot classification에서 검증 |
| **Feature Hallucination via VI** (Luo et al., WACV 2021) | Variational Inference 기반 | 각 category에 대해 latent space에서 generating space를 정의, cosine classifier 기반 | 1-shot에서 특히 유효 |
| **Feature Hallucination via MAP** (KBS 2021) | MAP estimation | Base set이 Gaussian이라 가정, novel category의 분포를 1개 example에서도 MAP으로 추정 | 1-shot에서도 동작 |

**AD 적용 가능성**: 높음. support patch feature (~200-1600개 patch)에서 hallucinated patch feature를 생성하면 memory bank coverage를 확장할 수 있다. 다만, AD에서는 class boundary가 아니라 정상 분포의 coverage가 목표이므로, hallucination이 정상 분포 내부를 채우는 방향이어야 한다.

**구현 복잡도**: 중간. 학습 가능한 hallucinator network 필요 (경량 MLP 수준).

### 1.2 Distribution Calibration (DC)

**핵심 논문**: **Free Lunch for Few-Shot Learning: Distribution Calibration** (Yang et al., ICLR 2021 Oral)

- 핵심 아이디어: few-shot class의 feature 분포를 base class 통계(mean, variance)를 이용해 calibrate.
- 각 feature dimension이 Gaussian을 따른다고 가정하고, 유사한 base class의 통계를 빌려와서 novel class의 분산을 보정.
- 보정된 분포에서 샘플링하여 support set을 확장.
- **추가 파라미터 없이** pretrained feature extractor 위에 바로 적용 가능.
- miniImageNet에서 ~5% 개선.

**AD 적용 가능성**: 매우 높음 + 구현 간단. AD에서는 base class가 없지만, **같은 support 내의 다른 spatial position의 통계**나 **다른 category의 정상 통계**를 빌려오는 변형이 가능하다. Memory decomposition에서 dominant component의 통계를 residual component calibration에 활용하는 것과 자연스럽게 연결.

**구현 복잡도**: 낮음. 추가 네트워크 불필요, 통계 연산만.

### 1.3 Manifold Mixup

**핵심 논문**: **Manifold Mixup** (Verma et al., ICML 2019)

- Hidden layer에서 두 샘플의 feature를 convex combination하여 augmentation.
- 결과: smoother decision boundary, fewer directions of variance.
- **AlignMixup** (Venkataramanan et al., 2021): geometrically aligned feature interpolation → style transfer 효과.

**AD 적용 가능성**: 중간. 정상 샘플 간 mixup은 정상 분포 내부를 채우는 효과가 있지만, 정상 boundary 밖으로 나가면 false negative 유발 위험. Support patch 간 같은 spatial position끼리 mixup하면 risk를 줄일 수 있다.

**구현 복잡도**: 낮음. Feature extraction 후 단순 연산.

---

## 2. Support Set Refinement (Pruning, Reweighting, Reorganization)

### 2.1 Coreset Selection (PatchCore 계열)

**핵심 논문**: **Towards Total Recall in Industrial Anomaly Detection (PatchCore)** (Roth et al., CVPR 2022)

- k-center-greedy coreset subsampling으로 memory bank에서 대표적인 patch subset 선택.
- Johnson-Lindenstrauss lemma 기반 random projection으로 차원 축소 후 greedy selection.
- PatchCore-1% (메모리 99% 감소)에서도 pixel AUROC 98.0% 유지.
- 핵심 insight: **redundant feature 제거**가 성능을 거의 떨어뜨리지 않으면서 효율성을 크게 높인다.

**AD 적용 확인됨**: PatchCore 자체가 AD 방법. Few-shot에서는 coreset이 작아서 subsampling 효과가 제한적이지만, "어떤 patch를 보존할 것인가"의 원리는 여전히 유효.

**우리 방법과의 관계**: Dominant/residual decomposition은 coreset의 상위 개념으로 볼 수 있다. Coreset은 "대표성 기준 선별", 우리는 "frequency 기반 구조화"로 차별화.

### 2.2 SoftPatch: Noise-aware Reweighting

**핵심 논문**: **SoftPatch: Unsupervised Anomaly Detection with Noisy Data** (Xi et al., 2024)

- Training data에 noise(약한 anomaly)가 섞여있을 때, 각 patch feature에 confidence weight를 부여.
- Memory bank의 각 entry에 soft weight를 적용하여 noisy patch의 영향을 줄임.

**AD 적용 확인됨**: 직접 AD 방법. Few-shot에서는 noise보다 coverage 부족이 더 큰 문제이지만, reweighting 원리는 dominant/residual 간 가중치 조절에 응용 가능.

### 2.3 Prototype Rectification

**핵심 논문**: **Prototype Rectification for Few-Shot Learning** (Liu et al., ECCV 2020)

- Intra-class bias: 소수 샘플이 class 중심을 편향되게 추정.
- Cross-class bias: 다른 class 정보가 prototype에 침투.
- 해결: label propagation으로 intra-class bias 완화 + feature shifting으로 cross-class bias 완화.
- miniImageNet 1-shot: 70.31%, 5-shot: 81.89%.

**AD 적용 가능성**: 높음. Few-shot AD에서 support가 정상 분포의 중심을 편향되게 추정하는 것은 동일한 문제. Label propagation 대신, patch간 유사도 기반 propagation으로 변형 가능.

### 2.4 BaseTransformers

**핵심 논문**: **BaseTransformers: Attention over Base Data-Points for One Shot Learning** (Maniparambil et al., 2022)

- Support instance의 representation을 base dataset의 잘 학습된 feature representation을 참조하여 개선.
- Base dataset에서 support에 가장 가까운 feature를 attention으로 가져옴.
- 1-shot classification에서 SOTA.

**AD 적용 가능성**: 중간-높음. AD에서 "base dataset"은 존재하지 않지만, 다른 category의 정상 패턴이나, auxiliary normal dataset의 feature를 참조하는 변형이 가능. Multi-class FSAD setting에서 다른 class의 정상 feature를 빌려오는 것과 유사.

---

## 3. Graph-based Feature Propagation

### 3.1 GraphCore (ICLR 2023)

**핵심 논문**: **Pushing the Limits of Few-shot Anomaly Detection in Industry Vision: GraphCore** (Xie et al., ICLR 2023)

- Patch 간 관계를 graph로 모델링하여 rotation-invariant feature (VIIF) 생성.
- Graph representation을 통해 patch 간 geometric relationship을 capture.
- Memory bank 크기를 크게 줄이면서 성능 향상.
- MVTec 1-shot: +5.8%, MPDD 1-shot: +25.5% (vs PatchCore).

**핵심 contribution**: Feature 자체가 아니라 feature 간의 **관계**(graph structure)를 저장함으로써, 같은 수의 support에서 더 많은 정보를 추출.

**우리 방법과의 관계**: GraphCore의 graph structure + 우리의 dominant/residual decomposition은 orthogonal. Graph로 관계를 모델링하면서 동시에 frequency 기반으로 구조화하는 것이 가능.

### 3.2 UniVAD: Graph-Enhanced Component Modeling (GECM)

**핵심 논문**: **UniVAD** (Gu et al., 2024)

- Component-Aware Patch Matching (CAPM)과 Graph-Enhanced Component Modeling (GECM) 결합.
- Image를 semantic component로 분할 후, component 간 관계를 graph로 모델링.
- 9개 dataset에서 domain-specific 모델을 능가.

**AD 적용 확인됨**: 직접 FSAD 방법. Component 단위 graph modeling이 patch 단위보다 semantic하게 동작.

### 3.3 Few-shot Segmentation의 GNN 활용

Few-shot segmentation에서 support-query 간 관계를 GNN으로 propagation하는 연구가 다수 존재:
- Support patch와 query patch를 node로, 유사도를 edge로 구성.
- Message passing을 통해 support 정보가 query로 전파.
- AD에서는 query가 anomaly일 수 있으므로, propagation이 정상→이상 방향으로만 일어나도록 제어 필요.

---

## 4. Attention-based Support Aggregation

### 4.1 Cross-Attention in Few-Shot Segmentation

**핵심 논문**: **Self-Calibrated Cross Attention Network (SCCAN)** (Xu et al., 2023)

- Query와 support feature 간 cross-attention으로 pixel-level matching.
- Patch alignment module: query patch를 가장 유사한 support patch에 정렬.
- Query BG features가 support FG와 잘못 매칭되는 문제를 해결.
- COCO-20^i 5-shot: +5.6% mIoU vs previous SOTA.

**AD 적용 가능성**: 높음. AD에서 query patch가 정상이면 support에서 matching이 잘 되고, anomaly이면 matching이 안 되는 것을 attention score로 직접 anomaly score화할 수 있다.

### 4.2 AENet: Ambiguity Elimination

**핵심 논문**: **AENet** (Xu et al., 2024)

- Cross-attention 기반 FSS에서, FG feature에 BG가 혼재되는 ambiguity 문제를 해결.
- Discriminative query FG region을 mining하여 ambiguous feature를 rectify.
- Plug-in으로 기존 cross-attention FSS에 적용 가능.

### 4.3 InCTRL: In-Context Residual Learning

**핵심 논문**: **InCTRL** (Zhu et al., CVPR 2024)

- Few-shot normal images를 "in-context sample prompts"로 사용.
- Query와 support 간의 **residual**을 학습하여 anomaly 판별.
- Generalist model: 다양한 domain에 걸쳐 일반화.

**AD 적용 확인됨**: 직접 FSAD 방법. Residual learning이 우리의 residual memory와 conceptually 유사하지만, InCTRL은 query-support 간 residual, 우리는 support 내부의 dominant-residual 분해로 차별화.

### 4.4 DictAS: Dictionary Lookup for FSAS

**핵심 논문**: **DictAS** (Qu et al., 2025)

- Normal reference image features로 dictionary를 구성.
- Sparse lookup으로 query feature를 dictionary에서 retrieve.
- Retrieve 실패 = anomaly로 판별.
- Contrastive Query Constraint로 anomalous feature가 dictionary에서 retrieve되기 어렵게 만듦.

**AD 적용 확인됨**: 직접 FSAD 방법. Dictionary 구성 방식이 memory bank과 유사하지만, sparse lookup이라는 retrieval 방식이 차별점.

---

## 5. Distribution Estimation with Few Samples

### 5.1 PaDiM: Multivariate Gaussian per Patch

**핵심 논문**: **PaDiM** (Defard et al., ICPR 2021)

- 각 spatial position에서 multi-layer feature의 multivariate Gaussian distribution을 추정.
- Mahalanobis distance로 anomaly score 계산.
- PCA로 feature dimension 축소 (100-200차원).
- Full-shot (수백장) 기준으로 설계되어, few-shot에서는 covariance 추정이 불안정.

**Few-shot 문제**: k-shot에서 patch당 k개 sample로 수백 차원의 covariance를 추정하는 것은 수학적으로 ill-posed (n < p 문제).

### 5.2 Shrinkage Estimators

**Ledoit-Wolf Shrinkage** (Ledoit & Wolf, JMVA 2004):
- Sample covariance를 identity matrix 방향으로 shrink: `Σ_shrunk = (1-α)Σ_sample + αI`
- α를 data-driven으로 최적 선택하는 closed-form formula 제공.
- **장점**: 항상 positive definite, well-conditioned, n < p에서도 안정적.
- scikit-learn에 구현되어 있어 즉시 사용 가능.
- **AD에서의 활용**: PaDiM의 few-shot 변형에서 covariance 추정의 핵심 문제를 해결.

**Oracle Approximating Shrinkage (OAS)** (Chen et al., 2010):
- Ledoit-Wolf의 개선 버전, 더 나은 shrinkage coefficient 추정.

**Robust Shrinkage for High Dimensions** (Chen et al., IEEE TSP 2011):
- Tyler의 robust covariance estimator + shrinkage 결합.
- Elliptical distribution family에서 distribution-free.
- n < p에서도 동작하는 robust estimator.

### 5.3 Minimum Covariance Determinant (MCD)

- Robust covariance 추정의 고전적 방법.
- Outlier에 강건하지만, **few-shot에서는 n이 너무 작아서 적용 어려움** (MCD는 일반적으로 n > 5p를 권장).
- AD training에는 outlier가 없으므로(정상만), robustness보다는 regularization이 더 중요.

### 5.4 Distribution Estimation 전략 비교 (Few-Shot AD 관점)

| 방법 | n < p 대응 | 정밀도 | 구현 복잡도 | Few-shot AD 적합성 |
|------|-----------|--------|------------|-------------------|
| Sample Covariance | X (singular) | - | 매우 낮음 | 사용 불가 (k < d) |
| PCA + Sample Cov | O (d를 줄임) | 중간 | 낮음 | PaDiM 방식 |
| Ledoit-Wolf Shrinkage | O | 높음 | 낮음 | **매우 적합** |
| OAS | O | 높음 | 낮음 | 매우 적합 |
| Robust Shrinkage | O | 높음 | 중간 | 적합 (outlier 걱정 시) |
| KDE (non-parametric) | O | 상황의존 | 중간 | k가 매우 작으면 불안정 |
| Normalizing Flow | O | 높음 | 높음 | 학습 필요, few-shot에서 overfitting 위험 |

---

## 6. Synthesis: Complementarity with Memory Decomposition

### 6.1 우리의 Dominant + Residual Decomposition과의 조합 가능성

| Technique | Complementarity | How to Combine |
|-----------|----------------|----------------|
| **Distribution Calibration** | **매우 높음** | Dominant component의 통계로 residual component의 분포를 calibrate → residual의 coverage 확장 |
| **Feature Hallucination** | 높음 | Residual subspace에서 hallucinated features를 생성하여 rare normal pattern coverage 확대 |
| **Shrinkage Estimation** | **매우 높음** | Decomposition 후 각 subspace에서 Ledoit-Wolf로 안정적 distribution estimation |
| **Coreset Selection** | 중간 | Dominant component 내에서 coreset으로 further 압축, residual은 전체 보존 |
| **Graph Propagation** | 높음 | Patch 간 graph로 dominant/residual membership을 propagate, boundary patch 처리 개선 |
| **Cross-Attention** | 높음 | Query-support matching 시, dominant과 residual에 서로 다른 attention weight 적용 |

### 6.2 권장 조합 (Novelty + Feasibility 기준)

**1순위: Memory Decomposition + Distribution Calibration + Shrinkage**
- Novelty: dominant의 풍부한 통계로 residual의 sparse 분포를 calibrate하는 것은 기존에 없는 접근.
- Feasibility: 추가 학습 불필요, 통계 연산만으로 구현.
- 학술적 의미: "frequent pattern의 통계적 풍요가 rare pattern 추정을 지원한다"는 원리 제시.

**2순위: Memory Decomposition + Graph Structure**
- Novelty: patch 간 관계를 graph로 모델링하면서, graph의 hub node를 dominant, peripheral node를 residual로 매핑.
- Feasibility: GNN 학습이 필요할 수 있으나, graph construction은 training-free 가능.
- 학술적 의미: "feature space에서의 topology가 memory 구조를 결정한다"는 원리.

**3순위: Memory Decomposition + Feature Hallucination (Residual Subspace)**
- Novelty: residual subspace에서만 targeted hallucination.
- Feasibility: hallucinator 학습 필요.
- 학술적 의미: "어디를 augment할지 아는 것이 얼마나 augment할지보다 중요하다."

---

## 7. Key Papers Quick Reference

### Feature Augmentation
- Lazarou et al. "Tensor Feature Hallucination for Few-Shot Learning" WACV 2022
- Luo et al. "Few-Shot Learning via Feature Hallucination with Variational Inference" WACV 2021
- Yang et al. "Free Lunch for Few-Shot Learning: Distribution Calibration" ICLR 2021 Oral
- Verma et al. "Manifold Mixup: Better Representations by Interpolating Hidden States" ICML 2019

### Support Refinement
- Roth et al. "Towards Total Recall in Industrial Anomaly Detection (PatchCore)" CVPR 2022
- Liu et al. "Prototype Rectification for Few-Shot Learning" ECCV 2020
- Maniparambil et al. "BaseTransformers: Attention over Base Data-Points for One Shot Learning" 2022
- Xi et al. "SoftPatch: Unsupervised Anomaly Detection with Noisy Data" 2024

### Graph-based
- Xie et al. "Pushing the Limits of Few-shot Anomaly Detection: GraphCore" ICLR 2023
- Gu et al. "UniVAD: Training-free Unified Model for Few-shot Visual Anomaly Detection" 2024

### Attention-based
- Xu et al. "Self-Calibrated Cross Attention Network for Few-Shot Segmentation" 2023
- Zhu et al. "InCTRL: In-context Residual Learning for Few-shot AD" CVPR 2024
- Qu et al. "DictAS: Dictionary Lookup for Few-Shot Anomaly Segmentation" 2025

### Distribution Estimation
- Defard et al. "PaDiM: Patch Distribution Modeling" ICPR 2021
- Ledoit & Wolf "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices" JMVA 2004
- Chen et al. "Robust Shrinkage Estimation of High-dimensional Covariance Matrices" IEEE TSP 2011

### FSAD Methods (Support Utilization Focus)
- Huang et al. "RegAD: Registration-based Few-Shot Anomaly Detection" ECCV 2022 Oral
- Liao et al. "COFT-AD: Contrastive Fine-Tuning for Few-Shot AD" IEEE TIP 2024
- Wang et al. "VisionAD: Search is All You Need for Few-shot AD" 2025
- Gao "MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning" NeurIPS 2024

---

## 관련 노트
- [Paper Outline: Support Memory](./2026-03-10_paper_outline_support_memory.md)
- [Comprehensive Survey](./2026-03-10_comprehensive_survey.md)
- [Research Gaps](./2026-03-10_research_gaps.md)
