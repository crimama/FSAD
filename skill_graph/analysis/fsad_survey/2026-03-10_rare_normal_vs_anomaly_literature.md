# Rare-Normal vs Anomaly Disambiguation: Literature Research

> Date: 2026-03-10
> Purpose: Principled methods to distinguish "rare but valid normal patterns" from "actual anomalies" in few-shot AD feature space. Feeds directly into residual memory design (Section 4.3 of paper outline).

---

## Problem Statement

In few-shot AD, the support set contains patches with varying frequency:
- **Dominant normals**: frequent, tightly clustered (e.g., uniform texture regions)
- **Rare normals**: infrequent but genuinely normal (e.g., edges, logos, transitions)
- **Anomalies**: genuinely defective patches

Both rare normals and anomalies are distant from the dominant cluster in feature space. A flat nearest-neighbor score conflates the two, causing:
- Rare normals scored as anomalous -> false positives
- Threshold raised to accommodate rare normals -> missed true anomalies

The residual memory (M_r) must preserve rare normals while not becoming a shelter for anomalies. We need a **principled criterion** for this separation.

---

## 1. Typicality vs Novelty (Information-Theoretic View)

### Key References
- **Nalisnick et al. (2019)** "Detecting Out-of-Distribution Inputs to Deep Generative Models Using Typicality" [arXiv:1906.02994](https://arxiv.org/abs/1906.02994)
- **FORTE (ICLR 2025)** "Finding Outliers with Representation Typicality Estimation" [arXiv:2410.01322](https://arxiv.org/abs/2410.01322)
- **Osada et al. (WACV 2023)** "OOD Detection with Reconstruction Error and Typicality-Based Penalty"
- **Typicality-Aware Learning (2024)** "Typicalness-Aware Learning for Failure Detection" [arXiv:2411.01981](https://arxiv.org/html/2411.01981)

### Core Principle
The **typical set** (from information theory) is the region containing most probability mass of a distribution. In high dimensions, high-likelihood regions and typical regions diverge: a sample can have high likelihood but be atypical (or vice versa). The key insight is:

- **Typicality != proximity to the mode.** A sample is "typical" if its log-likelihood falls within the expected range, not if it's closest to the center.
- Rare normals are **atypical but in-distribution** -- they sit outside the high-density core but within the typical set.
- True anomalies are **outside the typical set entirely** -- their statistics are inconsistent with the distribution.

### FORTE's Approach (Most Relevant)
FORTE (ICLR 2025) combines self-supervised representations (CLIP, DINOv2, ViT-MSN) with both parametric (GMM) and non-parametric (KDE, OCSVM) density estimators to estimate the typical set in representation space. Key improvements:
1. Uses self-supervised features (semantic, not pixel-level)
2. Incorporates manifold estimation for local topology
3. No additional training required

### Applicability to Patch-Level FSAD
- **High applicability.** Patch features from foundation models (DINOv2, etc.) are exactly the kind of representations FORTE targets.
- **Challenge:** The typical set estimation requires enough samples to characterize the distribution's entropy. In few-shot (k=1-4), the support set may be too small for reliable entropy estimation.
- **Adaptation idea:** Instead of estimating the full typical set, use a **relative typicality criterion**: compare the relationship between a residual patch and the dominant distribution to determine if it's "atypically normal" vs "out-of-distribution entirely." A patch that is distant from dominant centroids but whose local feature statistics (e.g., norm, angular distribution) are consistent with the support distribution is more likely rare-normal.
- **No additional training data required** -- works within the support set, but reliability improves with more samples.

### Implication for Residual Memory
The typicality framework provides theoretical grounding for why rare normals should be preserved: they are atypical (far from the mode) but still within the typical set. The residual memory M_r is essentially an explicit representation of the atypical-but-valid region.

---

## 2. Conformal Prediction for Anomaly Detection

### Key References
- **Laxhammar & Falkman (2014)** "Conformal Anomaly Detection" [Doctoral thesis, Halmstad University](https://www.diva-portal.org/smash/get/diva2:690997/FULLTEXT02.pdf)
- **Ishimtsev et al. (2017)** "Conformal k-NN Anomaly Detector for Univariate Data Streams" [PMLR](http://proceedings.mlr.press/v60/ishimtsev17a/ishimtsev17a.pdf)
- **Adaptive Conformal AD (2024)** [OpenReview](https://openreview.net/pdf/f683f9a6c7bdba56dd602884d3df0c16a8b0a309.pdf)
- **DANCE (2025)** "Doubly Adaptive Neighborhood Conformal Estimation" [arXiv:2602.20652](https://arxiv.org/html/2602.20652)

### Core Principle
Conformal prediction converts any scoring function into **calibrated p-values** with distribution-free guarantees:

1. Define a **nonconformity measure (NCM)**: any function measuring "strangeness" of a sample relative to a reference set (e.g., k-NN distance, Mahalanobis distance).
2. Compute NCM scores on a **calibration set** (here, the support set patches).
3. For a new query patch, its p-value = fraction of calibration scores >= query score.
4. If p-value < epsilon (significance level), flag as anomaly.

The critical property: **the p-value is calibrated** -- if the NCM is computed on exchangeable data, the false alarm rate is bounded by epsilon regardless of the underlying distribution.

### Relevance to Rare-Normal Problem
- Rare normals will have **low but non-negligible p-values** (e.g., 0.05-0.15). They are unusual but not extreme.
- True anomalies will have **near-zero p-values** (below any reasonable significance level).
- Conformal prediction provides an **adaptive threshold** that accounts for the actual distribution of strangeness scores in the support set, rather than a fixed distance threshold.

### Challenge: Small Calibration Set
- In few-shot AD, the calibration set (support patches) may be small. With N calibration points, the smallest achievable p-value is 1/(N+1). For 1-shot with ~196 patches (14x14 ViT grid), the resolution is ~1/197 = 0.005, which is reasonable.
- For coarser patch grids or very few shots, resolution degrades.
- **Mitigation:** Pool patches across spatial positions and (if available) across support images. Even 1-shot gives ~196 patches; 4-shot gives ~784.

### Applicability to Patch-Level FSAD
- **Moderate-to-high.** The framework is directly applicable to patch-level features.
- **Implementation:** Use k-NN distance to support patches as the NCM. Compute the empirical distribution of k-NN distances among support patches themselves. A query patch is anomalous if its k-NN distance exceeds the (1-epsilon) quantile of the support-internal distance distribution.
- **Key advantage:** No distributional assumptions; works with any feature space.
- **Key limitation:** Assumes exchangeability of support patches, which is only approximately true (spatial correlations exist).

### Implication for Residual Memory
Conformal prediction suggests a **principled threshold** for residual memory inclusion: patches with NCM p-values above a minimum threshold (e.g., > 1/(N+1)) are "conformally consistent" with the support and should be retained in M_r. This replaces ad-hoc distance thresholds with a statistically grounded criterion.

---

## 3. Local vs Global Outlier Scores

### Key References
- **Breunig et al. (2000)** "LOF: Identifying Density-Based Local Outliers" [ACM SIGMOD](https://dl.acm.org/doi/10.1145/335191.335388) -- foundational paper, 10k+ citations
- **k-NNN (WACV 2024W)** Nizan & Tal, "Nearest Neighbors of Neighbors for Anomaly Detection" [arXiv:2305.17695](https://arxiv.org/abs/2305.17695)
- **KFC** "Outlier Detection: How to Select k for k-NN-based Outlier Detectors" (Pattern Recognition Letters, 2023)

### Core Principle: Local Outlier Factor (LOF)
LOF computes the **local reachability density** of a point relative to its k-nearest neighbors. The key insight:

- **Global distance** treats all regions uniformly: a point 5 units from its nearest neighbor is always "far."
- **Local density ratio** normalizes by the local neighborhood: a point 5 units away in a sparse region (LOF ~ 1) is normal, but the same distance in a dense region (LOF >> 1) is anomalous.

For rare normals:
- Rare normal patches form **small but locally consistent clusters** (e.g., logo patches cluster together even though they're far from the dominant texture cluster).
- LOF score ~ 1 for these patches: they are not local outliers, just globally rare.
- True anomaly patches are **isolated even locally**: no consistent neighborhood structure.

### k-NNN: Second-Order Neighborhood Structure
k-NNN extends standard k-NN by considering not just a point's nearest neighbors, but the **neighbors of those neighbors**. The insight:
- If a patch's nearest neighbors also have each other as neighbors (high reciprocity / neighborhood overlap), the patch belongs to a coherent local structure (likely rare-normal).
- If a patch's nearest neighbors are themselves scattered with different neighbor sets, the patch is truly isolated (likely anomaly).

### Applicability to Patch-Level FSAD
- **High applicability, with caveats.**
- LOF can be computed purely within the support patch set. Rare normal patches (e.g., edge patches) will have LOF ~ 1 if they appear in multiple support images or form spatial groups.
- **Problem with 1-shot:** Only one image means logo/edge patches appear exactly once with no local cluster. LOF becomes unreliable.
- **Multi-shot advantage:** With k >= 2, the same spatial patterns recur across images, forming local clusters for rare normals. This is where LOF-based filtering becomes reliable.
- **k-NNN is directly plug-in compatible** with PatchCore-style memory banks -- replace the NN scoring operator without changing the rest.

### Implication for Residual Memory
LOF provides a **local density criterion** for residual memory quality control:
- Residual patches with LOF ~ 1 (locally consistent) -> keep in M_r (likely rare normal)
- Residual patches with LOF >> 1 (locally isolated) -> candidate for removal (noise or extraction artifact)

This is complementary to the conformal approach: conformal tests global consistency, LOF tests local consistency. Both passing = strong rare-normal candidate.

---

## 4. Support Coverage Analysis

### Key References
- **PatchCore (CVPR 2022)** Roth et al. -- **minimax facility location coreset** for memory bank compression [anomalib docs](https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/models/image/patchcore.html)
- **InCTRL (CVPR 2024)** Zhu & Pang, "Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts" [arXiv:2403.06495](https://arxiv.org/abs/2403.06495)
- **AFDM (2024)** "Few-shot AD with Adaptive Feature Transformation and Descriptor Construction" [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S100093612400219X) -- Sample-Conditioned Transformation
- **FastRecon (ICCV 2023)** Fang et al. -- few-shot feature reconstruction

### Core Principle: Coverage vs Compression
PatchCore's coreset sampling uses **greedy minimax facility location**: iteratively select the point that minimizes the maximum distance from any unselected point to the nearest selected point. This guarantees:
- Every original patch has a representative within distance delta in the coreset.
- **Coverage** = no patch is orphaned (too far from any coreset member).

However, this is **quantity-agnostic**: a rare normal patch and a dominant patch get equal representation rights. The coreset optimizes worst-case distance, not frequency-weighted importance.

### The Coverage Gap in Few-Shot
In few-shot, the coreset IS the support set (or close to it). The real coverage problem is:
- The support images may not contain all valid normal patterns (e.g., no logo visible in k=1).
- Even if present, rare patterns get 1-2 patches vs 100+ for dominant patterns.
- Standard coreset sampling may drop rare patches first (they're "close enough" to some dominant centroid in the compression metric, even though semantically distinct).

### InCTRL's Residual Approach
InCTRL (CVPR 2024) learns to detect anomalies by modeling the **residual between query and support**. Key insight: anomalies produce larger residuals than normals. This is directly related to our framework:
- Dominant normals -> small residual (well-represented)
- Rare normals -> moderate residual (partially represented)
- Anomalies -> large residual (not represented)

The gap between "moderate" and "large" residual is exactly the rare-normal vs anomaly boundary.

### Sample-Conditioned Transformation (AFDM)
AFDM introduces a module that transforms support features **conditioned on the query input** to generate additional normal patterns, improving coverage. This is complementary to our approach:
- AFDM: expand coverage by hallucinating new normals from support.
- Our approach: preserve coverage by explicitly retaining rare normals in M_r.

### Implication for Residual Memory
- Coverage analysis provides the **operational definition** of what M_r should contain: patches that are not well-covered by M_d (i.e., their nearest dominant centroid distance exceeds a coverage threshold).
- The minimax criterion gives a natural threshold: a patch needs residual representation if its distance to the nearest dominant centroid exceeds the average intra-dominant distance by some factor.
- **No additional training data required** -- purely support-internal computation.

---

## 5. Hard Normal Mining / Negative Mining

### Key References
- **Schroff et al. (CVPR 2015)** "FaceNet: A Unified Embedding for Face Recognition and Clustering" -- semi-hard triplet mining
- **ECCV 2020** "Hard Negative Examples Are Hard, but Useful" [ECCV paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590120.pdf)
- **Normality Learning via Multi-Scale Contrastive Learning (ACM MM 2023)** [ACM](https://dl.acm.org/doi/10.1145/3581783.3612064)
- **Prototype-Based Negative Mixing (2023)** for satellite telemetry AD [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10223175/)

### Core Principle
In metric learning, **semi-hard negatives** are the most informative training signals:
- **Easy negatives**: far from anchor, provide no gradient (already well-separated)
- **Hard negatives**: closest to anchor but from a different class, can destabilize training
- **Semi-hard negatives**: farther than the positive but within the margin -- stable, informative

For anomaly detection, the analogy is:
- **Easy normals** (dominant): trivially classified as normal
- **Hard normals** (rare): near the decision boundary, risk being classified as anomalous
- **Semi-hard anomalies**: actual anomalies that are close to normal but should be detected

### Hard Normal Mining in One-Class Setting
The challenge: in one-class AD (and especially few-shot), we have **no labeled anomalies**. We can't do traditional hard mining across classes. Instead:
- **Self-supervised hard normal mining**: identify support patches that are furthest from the support centroid but still normal. These are the "hard normals" that stress-test the boundary.
- **Normality learning (ACM MM 2023)**: uses multi-scale contrastive learning to learn a normality estimate, then selects reliable normal nodes to refine the model. The multi-scale aspect captures both local and global normal patterns.

### Prototype-Based Negative Mixing
Creates synthetic hard negatives by mixing prototype features, providing stronger contrastive signals. In our context, this could mean:
- Generate synthetic "near-anomaly" patches by perturbing rare normal patches.
- Train the scoring function to distinguish rare normals from these synthetic anomalies.
- **Requires training** (not purely inference-time), but could be a lightweight fine-tuning step.

### Applicability to Patch-Level FSAD
- **Moderate applicability.** Most hard mining techniques assume a training loop with labeled positive/negative pairs. In training-free FSAD, there's no such loop.
- **Adaptation for training-free setting:** Rather than mining for training, use the concept to **identify boundary patches** in the support set. Patches at the boundary of the normal distribution (furthest from centroid but still in M_r) are the "hard normals" that define the acceptance region.
- **If a lightweight adaptation step is allowed:** Use support-internal contrastive learning to sharpen the boundary between dominant and residual regions.

### Implication for Residual Memory
Hard normal mining provides the **operational strategy** for populating M_r with the most informative patches:
- Prioritize patches that are maximally distant from M_d but still pass the typicality/conformal/LOF filters.
- These "hard normals" are the patches most likely to be confused with anomalies and therefore most valuable in M_r.
- The margin concept from triplet loss suggests: M_r should extend the acceptance region by exactly enough to cover hard normals, not further.

---

## 6. Synthesis: A Principled Rare-Normal Criterion for Residual Memory

### Multi-Criteria Framework

No single method perfectly solves the rare-normal vs anomaly problem, but combining insights yields a principled framework:

```
For each support patch x_n not assigned to a dominant cluster:

1. Global consistency (Conformal):
   - Compute NCM score s(x_n) = distance to k-NN in support set
   - Compute p-value p(x_n) from empirical distribution of support-internal NCM scores
   - PASS if p(x_n) > epsilon_global  (not globally extreme)

2. Local consistency (LOF-inspired):
   - Compute local density ratio of x_n relative to its k neighbors
   - PASS if LOF(x_n) < tau_local  (locally consistent, not isolated)

3. Coverage necessity:
   - Compute distance to nearest dominant centroid: d(x_n, M_d)
   - INCLUDE in M_r if d(x_n, M_d) > delta_coverage  (not already covered by M_d)

Decision:
   - x_n in M_r  if  (global PASS) AND (local PASS) AND (coverage necessary)
   - x_n discarded  if  global FAIL or local FAIL
   - x_n absorbed into M_d  if coverage unnecessary
```

### Properties of This Framework
- **No additional training data**: all computations are within the support set.
- **Backbone-agnostic**: works with any feature extractor (pluggable).
- **Theoretically grounded**: conformal prediction provides statistical guarantees; LOF provides density-aware filtering; coverage analysis ensures no redundancy.
- **Shot-adaptive**: with more shots, all three criteria become more reliable (more calibration data, more local structure, better coverage estimation).
- **Config-controllable**: epsilon_global, tau_local, delta_coverage are tunable knobs for ablation.

### Connection to Paper Outline
This framework directly instantiates Section 4.3 step 3 ("Residual memory construction") with principled criteria replacing the current placeholder "residual pruning은 약하게 적용." It also addresses Risk 6.1.3 ("rare normal 보존은 noise 보존과 구분이 어려움") with a multi-criteria answer.

---

## 7. Practical Considerations

### Computational Cost
- LOF: O(N * k) where N = number of support patches, k = neighbors. For N~800 (4-shot), negligible.
- Conformal p-value: O(N) per query patch. Precomputable for support-internal use.
- Coverage distance: Already computed during dominant clustering.

### Failure Modes to Test
1. **1-shot edge case**: LOF unreliable with single image (no repeated rare patterns). Fall back to conformal-only.
2. **Contaminated support**: If a support image accidentally contains a subtle defect, the conformal and LOF criteria may accept it. Need contamination stress test.
3. **Over-aggressive residual pruning**: If criteria are too strict, M_r becomes empty and we revert to dominant-only (B1). Monitor M_r size as a diagnostic.

### Ablation Design
| Config | What it tests |
|--------|---------------|
| M_r disabled entirely | B1 vs B2 baseline |
| Conformal only (no LOF) | Value of local consistency |
| LOF only (no conformal) | Value of global calibration |
| Both criteria | Full framework |
| Varying epsilon_global | Sensitivity to global threshold |
| Varying tau_local | Sensitivity to local threshold |

---

## 관련 노트
- [Paper Outline (Support Memory)](./2026-03-10_paper_outline_support_memory.md) -- Section 4.3 residual memory
- [Research Gaps](./2026-03-10_research_gaps.md) -- Gap on support organization
- [PatchCore](./papers/patchcore.md) -- Coreset sampling baseline
- [Comprehensive Survey](./2026-03-10_comprehensive_survey.md)
