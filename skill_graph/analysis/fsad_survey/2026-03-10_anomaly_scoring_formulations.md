# Anomaly Scoring Formulations for Structured Memory (Dominant + Residual)

> 작성일: 2026-03-10
> 목적: 현 아웃라인의 ad-hoc scoring (Section 4.3-4)을 대체할 수 있는 principled scoring 방법론 조사

---

## 1. Mahalanobis Distance Scoring

### 1.1 Core Formulation (Lee et al., NeurIPS 2018)

**Reference**: "A Simple Unified Framework for Detecting OOD Samples and Adversarial Attacks" (Lee et al., NeurIPS 2018)

**Assumption**: Feature vectors follow class-conditional Gaussians with shared covariance:

```
p(f | y=c) = N(mu_c, Sigma)
```

**Mahalanobis distance score**:

```
d_Maha(x, mu_c) = (phi(x) - mu_c)^T  Sigma^{-1}  (phi(x) - mu_c)

s_Maha(x) = -min_c  d_Maha(x, mu_c)
```

**Multi-layer combination** (Lee et al.): weighted sum via logistic regression on validation set:

```
s(x) = sum_l  alpha_l * s_Maha^l(x)
```

where alpha_l are learned weights per layer l.

### 1.2 PaDiM Variant (Defard et al., ICPR 2020) -- Position-Conditional

**Key innovation**: Per-position Gaussian instead of per-class.

```
For each spatial position (i,j):
  N(mu_ij, Sigma_ij)

mu_ij = (1/N) sum_k x_ij^k
Sigma_ij = (1/(N-1)) sum_k (x_ij^k - mu_ij)(x_ij^k - mu_ij)^T + eps * I
```

**Anomaly map**:

```
M(i,j) = sqrt( (x_ij - mu_ij)^T  Sigma_ij^{-1}  (x_ij - mu_ij) )
```

**Multi-scale**: Concatenate features from layers {l1, l2, l3} before computing Gaussian. Random dimensionality reduction from d=448 to d=100 with only 0.4pp AUROC drop.

### 1.3 Mahalanobis++ (2025) -- Feature Normalization

```
phi_hat(x) = phi(x) / ||phi(x)||_2
```

Simple L2-normalization before Mahalanobis distance addresses heavy-tailed feature norms that violate Gaussian assumptions.

### 1.4 Applicability to Our Framework

**For dominant memory M_d** (clusters with mu_c, Sigma_c):

```
s_d(q_t) = min_c  (q_t - mu_c)^T  Sigma_c^{-1}  (q_t - mu_c)
```

This is strictly better than the current outline's `||q_t - mu_c|| / (sigma_c + eps)` because:
- Captures feature correlations (off-diagonal covariance)
- Accounts for anisotropic spread per cluster
- Reduces to scaled Euclidean when Sigma = sigma^2 * I

**Few-shot concern**: Sigma_c may be rank-deficient when cluster size < feature dim.
- Mitigation 1: Regularize with `Sigma_c + eps * I` (PaDiM approach)
- Mitigation 2: Shared covariance across clusters (Lee et al. approach) -- more stable with few samples
- Mitigation 3: Diagonal covariance approximation (reduces parameters from d^2 to d)
- Mitigation 4: Random projection to low-d before covariance estimation

**Computational overhead**: O(d^2) per query for full covariance, O(d) for diagonal. Covariance inversion is one-time at setup.

---

## 2. Multi-Scale / Multi-Granularity Scoring

### 2.1 Feature Pyramid Concatenation (PaDiM)

Concatenate features from multiple backbone layers before distribution modeling:

```
x_ij = [phi_l1(I)_ij ; phi_l2(I)_ij ; phi_l3(I)_ij]  in R^{d_1+d_2+d_3}
```

Lower layers: fine texture. Higher layers: semantic structure. Captures anomalies at different scales.

### 2.2 Feature Pyramid Matching (STPM, BMVC 2021)

**Reference**: Student-Teacher Feature Pyramid Matching (Wang et al., BMVC 2021)

Teacher (pretrained) and student (trained on normal) produce feature pyramids. Anomaly at each scale:

```
A_l(i,j) = ||F_teacher^l(i,j) - F_student^l(i,j)||^2
```

**Multi-scale aggregation**:

```
A(i,j) = sum_l  w_l * upsample(A_l, target_size)(i,j)
```

Typically w_l = 1 (equal weighting) or normalized by layer feature variance.

### 2.3 Multi-Scale Score Aggregation Strategies

| Strategy | Formula | Used by |
|---|---|---|
| Sum | S = sum_l S_l | STPM, CFLOW-AD |
| Product | S = prod_l S_l | Rarely used alone |
| Max | S = max_l S_l | Simple baseline |
| Weighted sum | S = sum_l w_l S_l | Lee et al. (learned w) |
| Concat + model | S = f([S_l1; ...; S_lK]) | Learned fusion |

### 2.4 Applicability to Our Framework

Our dominant and residual memories already operate at the patch level. Multi-scale enters through:
- **Option A**: Extract features at multiple layers, build separate dominant/residual memories per layer, aggregate scores across layers.
- **Option B**: Concatenate multi-layer features first (PaDiM-style), then build single structured memory on the concatenated space.
- **Option C**: Use different layers for dominant vs residual (e.g., higher layers for dominant patterns, lower layers for residual fine-grained patterns).

**Recommendation**: Option B is simplest and proven. Option C is novel but needs justification.

---

## 3. Dual-Path Score Combination

### 3.1 Current Outline (Ad-hoc)

```
a_t = alpha * s_d(t) + (1 - alpha) * max(0, s_d(t) - beta * s_r(t))
```

Problems: (1) Two arbitrary hyperparameters alpha, beta. (2) max(0, ...) clipping has no probabilistic interpretation. (3) Asymmetric treatment without clear justification.

### 3.2 Principled Alternatives

#### 3.2.1 Probabilistic Combination (Product of Likelihoods)

If dominant and residual paths model independent aspects of normality:

```
log p(q_t | normal) = log p_d(q_t) + log p_r(q_t)

s(q_t) = -log p(q_t | normal) = s_d(q_t) + s_r(q_t)
```

where s_d = -log p_d is the Mahalanobis-based score from dominant memory, and s_r = -log p_r is a distance-based score from residual memory.

**Interpretation**: A patch is anomalous if it is far from BOTH dominant AND residual patterns. This is the "conservative" combination -- only flags anomaly when neither path explains the query.

#### 3.2.2 Gated Minimum (OR-logic)

```
s(q_t) = min(s_d(q_t), s_r(q_t))
```

**Interpretation**: A patch is normal if EITHER path explains it well. This directly addresses the rare-normal preservation goal: even if s_d is high (far from dominant), low s_r (close to residual) suppresses the alarm.

**Equivalent formulation**:

```
s(q_t) = s_d(q_t) * sigmoid(gamma * (s_d(q_t) - s_r(q_t))) + s_r(q_t) * sigmoid(gamma * (s_r(q_t) - s_d(q_t)))
```

As gamma -> infinity, this becomes hard min. Finite gamma gives a soft version.

#### 3.2.3 Residual as Correction (Recommended)

**Motivation**: Dominant path is primary scorer. Residual path corrects false positives for rare normals.

```
s(q_t) = s_d(q_t) - lambda * max(0, s_d(q_t) - tau) * exp(-s_r(q_t) / sigma_r)
```

where:
- `s_d(q_t) - tau`: only correct when dominant score exceeds threshold tau
- `exp(-s_r(q_t) / sigma_r)`: correction strength decays as residual distance grows (only nearby residuals suppress)
- lambda: correction magnitude

**Simplified version** (1 hyperparameter):

```
s(q_t) = s_d(q_t) * (1 - exp(-s_r(q_t) / sigma_r))
```

When s_r is small (close to residual memory): score is suppressed toward 0.
When s_r is large (far from everything): score equals s_d.

This has a clean interpretation: **"anomalous if far from dominant AND far from residual."**

#### 3.2.4 Harmonic Mean

```
s(q_t) = 2 * s_d(q_t) * s_r(q_t) / (s_d(q_t) + s_r(q_t) + eps)
```

Low when either score is low. Naturally bounded. No hyperparameters beyond score normalization.

#### 3.2.5 Learned Combination (If Validation Data Available)

```
s(q_t) = MLP([s_d(q_t), s_r(q_t), s_d(q_t) - s_r(q_t), s_d(q_t) * s_r(q_t)])
```

Only viable if a small validation set with anomalies exists. Not few-shot friendly in the pure unsupervised setting.

### 3.3 Summary Table

| Method | Hyperparams | Interpretation | Few-shot? | Novelty |
|---|---|---|---|---|
| Weighted sum | alpha | Linear mix | Yes | Low |
| Product (log-sum) | 0 | Independent normality | Yes | Low |
| Min | 0 | OR-logic for normality | Yes | Low |
| Residual correction | sigma_r | Dominant primary, residual corrects FP | Yes | Medium |
| Product gating | 0 | AND-logic for anomaly | Yes | **High** |
| Harmonic mean | 0 | Bounded AND-like | Yes | Medium |
| Learned MLP | many | Data-driven | Needs val | Low |

**Recommendation**: The **product gating** formulation `s(q_t) = s_d(q_t) * (1 - exp(-s_r(q_t)/sigma_r))` is the strongest candidate:
- Single interpretable hyperparameter (sigma_r, or set to median residual distance)
- Clean story: "anomaly must be unexplained by both dominant AND residual memory"
- Naturally handles the rare-normal problem (low s_r -> low final score)
- No learned parameters needed

---

## 4. Likelihood-Based Scoring (Normalizing Flows)

### 4.1 CFLOW-AD (Gudovskiy et al., WACV 2022)

**Formulation**: Conditional normalizing flow per spatial position:

```
log p_Z(z, c, theta) = -(||u||^2 + D*log(2*pi)) / 2 + log|det J|

where u = g^{-1}(z, c, theta),  J = nabla_z g^{-1}
```

c encodes positional information. Multi-scale aggregation:

```
P_k = bilinear_upsample(exp(log_p_k))  in R^{H x W}
S = max(sum_k P_k) - sum_k P_k
```

### 4.2 FastFlow (Yu et al., 2021)

2D normalizing flow directly on feature maps (preserves spatial structure). Faster than CFLOW due to parallel computation.

### 4.3 DifferNet (Rudolph et al., 2021)

Image-level normalizing flow on global features. Exact log-likelihood for scoring.

### 4.4 Few-Shot Viability

**Problem**: Normalizing flows typically require hundreds/thousands of training samples to fit the bijective mapping parameters (coupling layers). In k-shot (k=1..8):
- Flow parameters are severely undertrained
- Risk of memorizing the few support samples
- Covariance-based methods (Mahalanobis) are statistically more efficient with small N

**Verdict**: Normalizing flows are NOT recommended for our few-shot setting. The Mahalanobis distance on structured memory is a more sample-efficient alternative that captures the same distributional information with far fewer parameters.

**Exception**: If using a pretrained/frozen flow (e.g., flow pretrained on ImageNet features, then evaluated on few-shot support), this could work as a general anomaly prior. But this changes the problem setup.

### 4.5 Computational Overhead

| Method | Training | Inference per image |
|---|---|---|
| Mahalanobis | O(N*d^2) one-time | O(K*d^2) per patch, K=num clusters |
| k-NN | O(1) | O(N*d) per patch |
| CFLOW-AD | O(epochs * N * flow_params) | O(H*W * flow_forward) |
| FastFlow | O(epochs * N * flow_params) | O(H*W * flow_forward) |

---

## 5. Energy-Based Scoring

### 5.1 Core Formulation (Liu et al., NeurIPS 2020)

```
E(x; f) = -T * log( sum_{i=1}^K exp(f_i(x) / T) )
```

where f_i(x) are logits, T is temperature. OOD score: higher energy = more anomalous.

**Connection to density**: `p(x) proportional to exp(-E(x))`, so energy is negative log-density up to normalization.

### 5.2 Adaptation to Our Setting

We don't have class logits, but we can define an energy analog over memory entries:

```
E(q_t) = -T * log( sum_{c in M_d} w_c * exp(-d(q_t, mu_c) / T) + sum_{r in M_r} w_r * exp(-d(q_t, r) / T) )
```

This is a **soft minimum distance** (LogSumExp of negative distances), which:
- Smoothly interpolates between min-distance (T -> 0) and mean-distance
- Naturally incorporates both dominant and residual memories in a single formula
- Weights w_c, w_r can encode prior frequency/importance

**This is arguably the most principled single-formula combination** of dominant and residual paths.

### 5.3 Temperature Selection

- T -> 0: Reduces to nearest-neighbor distance (hard assignment)
- T -> infinity: Average distance to all memory entries
- T moderate: Soft attention over memory, robust to single-point matching

Can set T via cross-validation or as median pairwise distance in support.

### 5.4 Few-Shot Viability

Excellent. No learned parameters. Only requires memory entries (mu_c from dominant, raw patches from residual) and a distance function. Temperature T is the single hyperparameter.

### 5.5 Computational Overhead

Same as nearest-neighbor: O(|M_d| + |M_r|) distance computations per query patch. Negligible additional cost over the distances already computed.

---

## 6. Score Calibration Across Categories

### 6.1 The Problem

Anomaly scores have different scales per category (e.g., "carpet" scores in [0, 50], "bottle" in [0, 200]). For unified thresholding or ranking across categories, calibration is needed.

### 6.2 Z-Score Normalization (Per-Category)

```
s_calibrated = (s - mu_normal) / sigma_normal
```

where mu_normal, sigma_normal are estimated from the support set scores (or a held-out normal validation set). After calibration, threshold t=3 means "3 sigma from normal."

**Few-shot issue**: With k=1..4 support images, mu_normal and sigma_normal have high variance. Mitigation: use patch-level statistics (hundreds of patches per image).

### 6.3 Percentile-Based Calibration

```
s_calibrated = percentile_rank(s, reference_distribution)
```

Map raw scores to [0, 1] based on the empirical CDF of normal scores. After calibration, s=0.99 means "higher than 99% of normal patches."

### 6.4 Modified Z-Score (Robust)

```
s_calibrated = 0.6745 * (s - median_normal) / MAD_normal
```

where MAD = median absolute deviation. More robust to outliers in the reference distribution.

### 6.5 UniAD Approach (You et al., NeurIPS 2022)

In multi-class unified models: derive per-class anomaly score distributions from training, calibrate test scores given the class. Key insight: inter-class distribution alignment via class-agnostic scoring.

### 6.6 Practical Recommendation for Our Setting

**Patch-level z-score normalization**:

```
For each category c:
  1. Compute anomaly scores for all support patches: {s_i}
  2. mu_c = mean({s_i}), sigma_c = std({s_i})
  3. For test patch: s_calibrated = (s - mu_c) / (sigma_c + eps)
```

This is simple, requires no additional data, and makes scores comparable across categories. With k=4 shot and ~100 patches per image, we have ~400 reference scores per category -- enough for stable statistics.

---

## 7. Concrete Proposal: Replacing Ad-Hoc Scoring in Section 4.3

### 7.1 Current (Ad-Hoc)

```
s_d(t) = min_c ||q_t - mu_c|| / (sigma_c + eps)
s_r(t) = min_{r in M_r} ||q_t - r||
a_t = alpha * s_d(t) + (1-alpha) * max(0, s_d(t) - beta * s_r(t))
```

### 7.2 Proposed Option A: Energy-Based Unified Score (Recommended)

```
s(q_t) = -T * log( sum_c w_c * exp(-d_Maha(q_t, mu_c, Sigma_c) / T)
                  + sum_r w_r * exp(-||q_t - r||^2 / T) )
```

**Properties**:
- Single formula combining dominant (Mahalanobis) and residual (Euclidean) paths
- Temperature T is the sole hyperparameter (set to median intra-support distance)
- w_c = cluster_size / N, w_r = 1/|M_r| (frequency-based priors)
- Reduces to nearest-neighbor as T -> 0
- Principled probabilistic interpretation (negative log mixture density)

**Novelty angle**: "Energy-based scoring over structured memory" -- combines the sample-efficiency of structured (clustered) representation with the principled scoring of energy-based models.

### 7.3 Proposed Option B: Mahalanobis Dominant + Product Gating

```
s_d(q_t) = min_c  (q_t - mu_c)^T  Sigma_c^{-1}  (q_t - mu_c)
s_r(q_t) = min_{r in M_r}  ||q_t - r||^2
s(q_t) = s_d(q_t) * (1 - exp(-s_r(q_t) / sigma_r))
```

where sigma_r = median({s_r(x_n) : x_n in support}).

**Properties**:
- Explicit dual-path with clear roles (dominant = primary, residual = FP correction)
- sigma_r is auto-calibrated from support statistics
- Clean story for the paper: "anomaly = unexplained by both dominant AND residual"
- Easy to ablate: remove exp term -> dominant-only; remove Mahalanobis -> flat memory

### 7.4 Comparison

| Aspect | Option A (Energy) | Option B (Product Gating) |
|---|---|---|
| Hyperparams | T | sigma_r (auto-set) |
| Interpretability | Mixture density | Dual-path with correction |
| Ablation clarity | Harder to disentangle paths | Easy on/off per path |
| Story for paper | "Structured energy scoring" | "Dominant-residual decomposition" |
| Novelty | Moderate (energy is known) | Higher (decomposition is our contribution) |
| Implementation | Single formula | Two scores + gating |

**Final recommendation**: **Option B** for the paper. The paper's novelty is the dominant-residual decomposition of support memory. Option B makes this decomposition explicit and ablatable in the scoring formula, which directly supports claims C2 and C3. Option A, while more elegant, merges the two paths and obscures the contribution.

Use Option A as an **additional ablation** to show that the structured memory helps regardless of how scores are combined.

---

## 8. Image-Level Aggregation (Patch -> Image Score)

### 8.1 Common Strategies

| Method | Formula | Robustness |
|---|---|---|
| Max | S_img = max_t s(q_t) | Sensitive to single outlier patch |
| Top-K mean | S_img = mean(top_k({s(q_t)})) | More robust, standard choice |
| Quantile | S_img = quantile_p({s(q_t)}) | Robust, parameter p |
| Generalized mean | S_img = (mean(s^p))^{1/p} | p=1: mean, p->inf: max |

**Standard practice**: Top-K mean with K = 1% of total patches (PatchCore convention).

---

## 관련 노트
- [Paper Outline](./2026-03-10_paper_outline_support_memory.md) -- Section 4.3 scoring to be replaced
- [Decomposition Techniques](./2026-03-10_decomposition_techniques_research.md)
- [Research Gaps](./2026-03-10_research_gaps.md)
- [PaDiM](./papers/) -- Mahalanobis scoring reference
- [Comprehensive Survey](./2026-03-10_comprehensive_survey.md)

## References
- Lee et al., "A Simple Unified Framework for Detecting OOD Samples and Adversarial Attacks", NeurIPS 2018
- Defard et al., "PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization", ICPR 2020
- Gudovskiy et al., "CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows", WACV 2022
- Wang et al., "Student-Teacher Feature Pyramid Matching for Anomaly Detection", BMVC 2021
- Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020
- You et al., "A Unified Model for Multi-class Anomaly Detection (UniAD)", NeurIPS 2022
- Roth et al., "Towards Total Recall in Industrial Anomaly Detection (PatchCore)", CVPR 2022
- Zavrtanik et al., "DSR: A Dual Subspace Re-Projection Network for Surface Anomaly Detection", ECCV 2022
- Mahalanobis++ (2025): Feature normalization for improved OOD detection
