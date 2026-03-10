# Few-Shot Anomaly Detection (FSAD) 종합 서베이

> 작성일: 2026-03-10
> 대상 논문: 41편 (2021-2026)
> 목적: FSAD 연구 동향 파악, 방법론 유형 분류, 핵심 인사이트 추출

---

## 1. 연구 배경 및 문제 정의

**Few-Shot Anomaly Detection (FSAD)**: 카테고리당 소수(k=1~8)의 정상 이미지만으로 이상을 탐지하는 문제.

**왜 Few-shot인가?**
- 산업 결함 검출, 의료 영상 등 라벨링 비용이 극도로 높은 도메인
- 새로운 제품/카테고리가 빈번하게 추가되는 생산 환경
- Full-shot (수백 장) 수집이 현실적으로 어려운 경우

**핵심 Challenge:**
- 소수 정상 샘플로 "정상 분포"를 정확히 추정해야 함
- Category-specific vs Category-agnostic 트레이드오프
- Image-level detection + Pixel-level localization 동시 달성
- Texture anomaly vs Structural/Logical anomaly의 특성 차이

---

## 2. 전반적 연구 추세 (2021 → 2026)

### Phase 1: 초기 탐색 (2021)
- Meta-learning, Normalizing Flow, Self-supervised transformation 등 기존 ML 패러다임을 FSAD에 적용
- 대표: Metaformer, HTD, DifferNet
- 특징: 카테고리별 학습, 비교적 낮은 성능 (~80-85% AUROC)

### Phase 2: 강력한 Baseline 확립 (2022-2023)
- **PatchCore** (memory-bank), **RegAD** (registration)가 핵심 baseline으로 자리잡음
- GraphCore가 GNN으로 PatchCore를 few-shot에 최적화
- Feature reconstruction (FastRecon), Metric learning (FewSOME) 등 다양한 접근 등장
- 특징: Pretrained feature 활용이 핵심, ~85-95% AUROC

### Phase 3: VLM/Foundation Model 혁명 (2024)
- **CLIP, DINOv2** 등 foundation model이 FSAD의 backbone으로 급부상
- Prompt learning (PromptAD, AnoPLe, CLIP-FSAC++), In-context learning (InCTRL)
- Multimodal (text+vision) 활용, Generation 기반 augmentation
- 특징: Training-free 가능, ~93-97% AUROC

### Phase 4: 통합 및 심화 (2025-2026)
- **Training-free SOTA**: VisionAD(2504.11895)가 DINOv2-Reg + NN search만으로 1-shot MVTec 97.4%, VisA 94.8% 달성 → 학습 기반 방법을 압도
- **Vision-only paradigm**: FoundAD(ICLR 2026)와 VisionAD 모두 text prompt 없이 순수 visual feature로 SOTA → VLM 의존성에 도전
- **Manifold projection**: FoundAD가 latent space 사영으로 효율적 AD 수행 (97.8M params vs IIPAD 1.0B)
- **Medical domain 확장**: D2-4FAD(ICLR 2026)가 dual distillation + learn-to-weight로 의료 영상 FSAD
- **Detection + Classification 통합**: UniADC(2511.06644)가 detection과 defect category 분류를 동시에 해결
- 특징: training-free가 주류, vision-only > VLM, ~96-98%+ AUROC

```
성능 추이 (MVTec AD Image AUROC, ~4-shot 기준):
2021: ~80-85%  ███████████
2022: ~83-88%  █████████████
2023: ~88-95%  ████████████████
2024: ~93-97%  ███████████████████
2025: ~97-99%  ████████████████████████
```

---

## 3. 방법론 유형 분류 (Taxonomy)

### 3.1 Memory-Bank 기반
> 정상 feature를 저장하고 거리 기반으로 anomaly scoring

| 논문 | 연도 | 핵심 기법 | 특징 |
|------|------|----------|------|
| **PatchCore** | CVPR 2022 | Coreset memory + k-NN | Full-shot SOTA, few-shot baseline |
| **GraphCore** | ICLR 2023 | GNN-enhanced PatchCore | Few-shot에서 context 보완 |
| Opt. PatchCore | 2023 | PatchCore hyperopt | Few-shot 최적 설정 탐구 |

**핵심 원리:** `score(x) = max_i { min_j ||φ(x)_i - m_j||₂ }`
- φ: pretrained feature extractor, m_j: memory bank의 j번째 patch feature
- 각 test patch의 nearest normal patch까지의 거리 → 최대값이 image score

**장점:** 단순, 해석 가능, strong baseline
**한계:** Few-shot에서 memory coverage 부족, category-specific

**핵심 인사이트:**
- PatchCore의 mid-level feature (layer 2-3)가 texture + semantic 정보를 모두 포함
- Few-shot에서는 coreset subsampling이 오히려 해로울 수 있음 (이미 적은 데이터)
- GNN으로 patch 간 관계 모델링 시 few-shot 성능 대폭 향상 (GraphCore)

---

### 3.2 Registration 기반
> 정상 reference와 test image를 정렬(align)한 뒤 차이로 anomaly 탐지

| 논문 | 연도 | 핵심 기법 | 특징 |
|------|------|----------|------|
| **RegAD** | ECCV 2022 | STN registration + residual | Few-shot AD 핵심 baseline |
| CARL | 2024 | Category-agnostic registration learning | 단일 모델 registration |

**핵심 원리:** `anomaly_map = |F(warp(support)) - F(query)|`
- Spatial Transformer Network으로 support를 query에 정렬
- 정렬 후 feature 차이 = anomaly

**장점:** Few-shot에 자연스러운 프레임워크, cross-category transfer
**한계:** Texture anomaly에 약함, flexible object에서 정렬 실패

**핵심 인사이트:**
- "Anomaly detection은 본질적으로 registration 문제" (RegAD)

---

### 3.3 VLM/Prompt Learning 기반
> CLIP 등 Vision-Language Model을 prompt tuning으로 AD에 적용

| 논문 | 연도 | 핵심 기법 | 특징 |
|------|------|----------|------|
| **PromptAD** | CVPR 2024 | Normal-only prompt learning | Anomaly prompt를 normal에서 학습 |
| **AnoPLe** | 2024 | Bi-directional prompt learning | Text + Visual 양방향 |
| **InCTRL** | CVPR 2024 | In-context residual learning | Generalist model |
| AnomalyGPT | AAAI 2024 | LVLM + anomaly decoder | Detection + NL explanation |
| FADE | BMVC 2024 | VLM zero/few-shot engine | Zero-shot + Few-shot 통합 |
| SOWA | 2024 | Hierarchical window self-attn | CLIP adapter, locality 보존 |
| CLIP-FSAC++ | 2024 | Learnable anomaly descriptor | Anomaly를 text space에서 학습 |
| KAG-prompt | AAAI 2025 | Kernel-aware graph prompting | Cross-layer contextual reasoning |
| One-for-All | ICLR 2025 | Instance-induced V-L prompting | Universal multi-class AD |
| Fine-Grained V-L | 2025 | Multi-level semantic alignment | Patch-level localization 강화 |

**핵심 원리:** CLIP의 vision-language alignment를 활용하여 "normal"과 "anomaly"의 semantic 경계를 학습

**주요 설계 패턴:**
1. **Text prompt learning**: "a photo of a normal/damaged [X]" 류의 prompt를 학습
2. **Visual prompt learning**: 이미지 encoder에 learnable token 삽입
3. **Bi-directional**: text + visual 양방향 동시 학습 (AnoPLe)

**장점:** Zero-shot도 가능, semantic understanding, category-agnostic
**한계:** Pixel-level localization에서 granularity 문제, 학습 비용

**핵심 인사이트:**
- Normal-only prompt learning이 가능 — anomaly prompt는 normal의 "complement"로 학습 (PromptAD)
- CLIP의 image-level alignment는 pixel-level AD에 부적합 → fine-grained alignment 필요
- In-context learning 패러다임 (InCTRL): few-shot normals = LLM의 in-context examples

---

### 3.4 Foundation Model / Training-free 기반
> DINOv2 등 foundation model feature를 학습 없이 직접 활용

| 논문 | 연도 | 핵심 기법 | 특징 |
|------|------|----------|------|
| **AnomalyDINO** | 2024 | DINOv2 patch matching | DINOv2 feature의 AD 적합성 |
| UniVAD | CVPR 2025 | Contextual component clustering | Training-free unified AD |
| **VisionAD** | 2025 | DINOv2-Reg + NN search + dual aug | Training-free SOTA, pseudo multi-view |
| **FoundAD** | ICLR 2026 | Nonlinear manifold projection | Vision-only, 97.8M params |

**핵심 원리:** Foundation model의 pretrained feature가 이미 anomaly detection에 필요한 정보를 내재

**장점:** Training-free, 즉시 적용, category-agnostic
**한계:** Feature 선택/조합의 최적화 여전히 필요

**핵심 인사이트:**
- DINOv2의 self-supervised feature가 supervised feature보다 "normality"를 더 잘 표현
- "학습 없이도 충분히 좋은 FSAD가 가능하다" — 실용성 극대화
- Foundation model의 어떤 layer/scale이 AD에 최적인지가 핵심 연구 질문
- **Vision-only > VLM**: FoundAD, VisionAD 모두 text prompt 없이 CLIP 기반 방법을 상회
- **Augmentation이 핵심 기여**: VisionAD ablation에서 support augmentation이 +2.1pp로 단일 최대 기여
- **Manifold distance = anomaly score**: FoundAD가 embedding space 거리와 anomaly severity의 직접 상관을 실증

---

### 3.5 Meta-Learning 기반
> Episode-based training으로 "anomaly detection 방법 자체"를 학습

| 논문 | 연도 | 핵심 기법 | 특징 |
|------|------|----------|------|
| Metaformer | ICCV 2021 | Transformer + episode training | 초기 meta-learning AD |
| HTD | ICCV 2021 | Hierarchical transformation | Self-supervised meta-learning |
| FewSOME | CVPRW 2023 | Siamese contrastive | Metric learning + few-shot |
| **MetaUAS** | NeurIPS 2024 | One-prompt meta-learning | Universal anomaly segmentation |
| **D2-4FAD** | ICLR 2026 | Dual distillation + learn-to-weight | Medical FSAD, query-conditional support |
| MetaCAN | CIKM 2025 | AD meta-learning on LVLM | Cross-category and cross-domain generalization |

**핵심 원리:** 다양한 카테고리에서 "정상 vs 이상 구분법"을 meta-learn → unseen category에 few-shot 적응

**장점:** Category-agnostic, 일반화 능력 학습
**한계:** Episode 구성 복잡, 학습 비용 높음

**추가 인사이트 (D2-4FAD):**
- Self-distillation(support-specific 정상성)이 teacher-student(일반 정상성)보다 few-shot에서 기여가 큼
- Learn-to-weight로 query-conditional support calibration 구현 → 모든 support를 동등하게 쓰지 않음

---

### 3.6 Reconstruction / Generation 기반
> 정상 이미지의 재구성 또는 anomaly 이미지 생성

| 논문 | 연도 | 핵심 기법 | 특징 |
|------|------|----------|------|
| **FastRecon** | ICCV 2023 | Fast feature reconstruction | Reconstruction error = anomaly |
| **One-to-Normal** | NeurIPS 2024 | Anomaly personalization | Diffusion 기반 normal 복원 |
| Text-Guided VAE | CVPR 2024 | Text-conditioned generation | Language-guided anomaly aug |
| SeaS | ICCV 2025 | Separation-sharing diffusion tuning | Few-shot anomaly generation |
| POUTA | 2023 | Reconstructive representation reuse | Efficient anomaly localization |

**핵심 원리:**
- **Reconstruction**: test image를 normal로 복원 → 원본과의 차이 = anomaly
- **Generation**: anomaly image를 생성하여 data augmentation → supervised 학습

**장점:** Pixel-level localization에 강함, data augmentation 효과
**한계:** 생성 품질에 의존, few-shot에서 generative model 학습 어려움

**핵심 인사이트:**
- Feature space reconstruction이 image space보다 few-shot에 robust (FastRecon)

---

### 3.7 기타 특수 접근

| 논문 | 연도 | 핵심 기법 | 카테고리 |
|------|------|----------|----------|
| **DifferNet** | WACV 2021 | Normalizing Flow | Explicit likelihood |
| **COFT-AD** | TIP 2024 | Contrastive fine-tuning | Backbone adaptation |
| Crossmodal FM | CVPR 2024 | RGB+3D cross-modal | Multimodal consistency |
| Dual-path Freq | 2024 | Multi-frequency dual-path discriminator | Frequency-domain FSAD |
| FOCT | ACM MM 2024 | Foreground-aware conditional transport | Soft correspondence |
| Few-shot Online | 2024 | Streaming memory adaptation | Online deployment setting |
| Small Object Seg | 2024 | Non-resizing few-shot segmentation | Small-defect inspection |
| RFS Energy | 2023 | Random finite set energy | Local statistical modeling |
| UniADC | 2025 | Unified detection + classification | Diagnosis-aware FSAD |

---

## 4. 핵심 인사이트 종합

### 4.1 Pretrained Feature가 FSAD의 핵심
거의 모든 방법론이 pretrained backbone (ImageNet ResNet/WideResNet → DINOv2/CLIP)의 feature를 기반으로 함. 차이는 "추출된 feature를 어떻게 활용하느냐"에서 발생.

### 4.2 Few-shot의 근본 한계: Coverage 부족
- Memory-bank: feature 수 부족 → GNN, augmentation으로 보완
- Distribution modeling: 분포 추정 불안정 → robust estimation 필요
- Registration: reference 다양성 부족 → cross-category transfer
- Prompt learning: prompt의 일반화 한계 → instance-induced 동적 생성

### 4.3 Category-Agnostic이 대세
초기에는 카테고리별 모델을 학습했으나, 최신 연구는 대부분 **unified/universal model**을 지향:
- InCTRL, MetaUAS 등

### 4.4 Foundation Model의 등장이 게임 체인저
DINOv2, CLIP의 feature가 AD에 inherently 적합함이 밝혀지면서:
- Training-free 접근이 현실적 대안으로 부상
- "무엇을 학습할 것인가"에서 "어떻게 활용할 것인가"로 연구 질문 변화
- **Vision-only paradigm 확립**: FoundAD, VisionAD 모두 text prompt 없이 VLM 기반 방법 상회 → CLIP의 language branch가 FSAD에 필수가 아님
- **Baseline bar 급상승**: VisionAD 1-shot 97.4%(MVTec) → 이 수준을 넘지 못하면 방법론적 novelty만으로는 불충분

### 4.5 Anomaly Generation이 유망한 방향
- Few-shot의 data scarcity를 generation으로 우회 (One-to-Normal, Text-Guided VAE)
- Diffusion model의 few-shot fine-tuning 기법이 핵심 기술

### 4.6 Pixel-level Localization이 여전한 도전
- Image-level detection은 상당히 성숙
- Pixel-level에서는 CLIP의 granularity gap, registration의 alignment 정확도 등 해결 과제 존재
- SOWA의 hierarchical window attention 등이 해결 시도

---

## 5. 주요 성능 비교 (MVTec AD, 대략적 수치)

| 방법론 | 1-shot I-AUROC (MVTec) | 1-shot I-AUROC (VisA) | 특징 |
|--------|----------------------|---------------------|------|
| PatchCore | ~83.4% | ~79.9% | Simple baseline |
| RegAD | ~80-83% | — | Registration baseline |
| GraphCore | ~85-88% | — | GNN-enhanced |
| AnomalyGPT | ~94.1% | ~87.4% | + NL explanation |
| PromptAD | ~94.6% | ~86.9% | CLIP prompt |
| InCTRL | ~93-95% | — | Generalist |
| KAG-prompt | ~95.8% | ~91.6% | Kernel-aware graph |
| AnomalyDINO | ~95-97% | — | DINOv2 training-free |
| FoundAD | **96.1%** | — | Vision-only manifold projection |
| **VisionAD** | **97.4%** | **94.8%** | Training-free NN search SOTA |

*주: 논문마다 실험 세팅이 다르므로 직접 비교에 주의. VisionAD 수치는 mean±std 보고.*

---

## 6. 연구 기회 및 Open Problems

### 6.1 Under-explored 영역
1. **Logical/Structural anomaly**: 대부분 texture/appearance에 집중. 구조적 결함(부품 누락, 잘못된 조립)은 few-shot에서 더 어려움
2. **Frequency domain**: Dual-path Freq이 유일. Spatial과 complementary한 정보원
3. **Active few-shot selection**: 어떤 normal image를 reference로 선택해야 최적인지
4. **Online/Continual adaptation**: 실 환경의 streaming 데이터 활용

### 6.2 Novelty Potential이 높은 방향
1. **Foundation model feature의 deeper analysis**: 왜, 어떤 조건에서 작동/실패하는지
2. **Reconstruction + Matching의 상보적 결합**: 각각의 강점을 살린 hybrid
3. **Cross-category knowledge transfer**: 결함 패턴의 category-agnostic 특성 활용
4. **Few-shot에서의 distribution estimation 이론**: 소수 샘플로 robust한 분포 추정

### 6.3 실용적 Gap
- **Speed**: 실시간 inference 요구 (Training-free 방법이 유리)
- **Memory**: Edge device 배포 (경량화 필요)
- **Interpretability**: AnomalyGPT처럼 설명 가능한 AD
- **Multi-class unified**: 하나의 모델로 모든 카테고리 처리

---

## 7. 우리 연구에 대한 시사점

### Baseline 선정
- **PatchCore**: Memory-bank 기반 표준 baseline
- **RegAD**: Registration 기반 표준 baseline
- **PromptAD / InCTRL**: VLM 기반 최신 baseline
- **VisionAD**: Training-free SOTA (1-shot MVTec 97.4%) — **반드시 비교해야 하는 최강 baseline**
- **FoundAD**: Vision-only manifold projection (ICLR 2026) — efficient FSAD baseline

### Novelty 가능 방향
1. **Support utilization principle**: VisionAD/FoundAD는 support를 flat하게 사용 — pattern-aware memory organization이 여전히 미탐색
2. **Query-conditional support calibration**: D2-4FAD의 learn-to-weight가 medical에서 유효 → 산업 FSAD로 확장/심화 가능
3. **Augmentation을 넘는 support enrichment**: VisionAD는 augmentation이 핵심이지만 hand-crafted — learnable augmentation 또는 구조적 보강
4. **Training-free의 한계를 정확히 드러내기**: VisionAD가 실패하는 조건(카테고리, anomaly 유형)을 분석하면 method gap 식별 가능
5. **Detection + diagnosis 통합**: UniADC 방향이지만 normal-only setting에서의 해결은 미탐색

---

## 관련 노트
- [개별 논문 상세 노트](./papers/) — 각 논문의 상세 정리
- [_lessons.md](./_lessons.md) — 서베이에서 추출한 검증된 패턴

---

## 전체 논문 목록 (가나다/연도순)

### 2021
1. Learning unsupervised metaformer for anomaly detection [ICCV] — Meta-learning
2. A hierarchical transformation-discriminating generative model for few shot anomaly detection [ICCV] — HTD
3. Same same but different: Semi-supervised defect detection with normalizing flows [WACV] — DifferNet
4. Anomaly detection of defect using energy of point pattern features within RFS framework — Statistical

### 2022
5. Registration based few-shot anomaly detection [ECCV Oral] — **RegAD**
6. Towards total recall in industrial anomaly detection [CVPR] — **PatchCore**

### 2023
7. Pushing the limits of fewshot anomaly detection in industry vision: Graphcore [ICLR] — **GraphCore**
8. Optimizing PatchCore for Few/many-shot Anomaly Detection
9. FastRecon: Few-shot Industrial Anomaly Detection via Fast Feature Reconstruction [ICCV]
10. Produce Once, Utilize Twice for Anomaly Detection — POUTA
11. FewSOME: One-Class Few Shot Anomaly Detection with Siamese Networks [CVPRW]

### 2024
12. AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models [AAAI]
13. COFT-AD: COntrastive Fine-Tuning for Few-Shot Anomaly Detection [TIP]
14. Text-Guided Variational Image Generation for Industrial AD and Segmentation [CVPR]
15. Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping [CVPR]
16. PromptAD: Learning Prompts with only Normal Samples for Few-Shot AD [CVPR]
17. InCTRL: Toward Generalist AD via In-context Residual Learning [CVPR]
18. AnomalyDINO: Boosting Patch-based Few-shot AD with DINOv2
19. AnoPLe: Few-Shot AD via Bi-directional Prompt Learning
20. CLIP-FSAC++: Few-Shot Anomaly Classification with Anomaly Descriptor Based on CLIP
21. SOWA: Adapting Hierarchical Frozen Window Self-Attention to VLMs for Better AD
22. Dual-path Frequency Discriminators for Few-shot AD
23. Few-shot Online Anomaly Detection and Segmentation
24. Small Object Few-shot Segmentation for Vision-based Industrial Inspection
25. Few-Shot Anomaly Detection via Category-Agnostic Registration Learning — CARL
26. FADE: Few-shot/zero-shot Anomaly Detection Engine using Large VLM [BMVC]
27. FOCT: Few-shot Industrial AD with Foreground-aware Online Conditional Transport [ACM MM]
28. Learning to Detect Multi-class Anomalies with Just One Normal Image Prompt [ECCV]
29. UniVAD: A Training-free Unified Model for Few-shot Visual AD
30. MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning [NeurIPS]
31. One-to-Normal: Anomaly Personalization for Few-shot AD [NeurIPS]

### 2025-2026
32. KAG-prompt: Kernel-Aware Graph Prompt Learning for Few-Shot AD [AAAI 2025]
33. One-for-All Few-Shot AD via Instance-Induced Prompt Learning [ICLR 2025]
34. SeaS: Few-shot Industrial Anomaly Image Generation with Separation and Sharing [ICCV 2025]
35. VisionAD: Search is All You Need for Few-shot Anomaly Detection [2025] — **Training-free SOTA** ✅
36. MetaCAN: Improving Generalizability of FSAD with Meta-learning [CIKM 2025]
37. UniADC: A Unified Framework for Anomaly Detection and Classification [2025]
38. Commonality in Few: Few-Shot Multimodal AD via Hypergraph-Enhanced Memory [2025]
39. Towards Fine-Grained Vision-Language Alignment for Few-Shot AD [2025]
40. FoundAD: Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors [ICLR 2026] — **Vision-only manifold projection** ✅
41. D2-4FAD: Dual Distillation for Few-Shot Anomaly Detection [ICLR 2026] — **Medical FSAD, dual distillation** ✅
