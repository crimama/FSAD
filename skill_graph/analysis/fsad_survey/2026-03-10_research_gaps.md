# FSAD Research Gaps After Survey

> 작성일: 2026-03-10
> 목적: FSAD 서베이 결과를 방법론 축으로 재정리하고, 연구 가치가 있는 갭을 추출

---

## 1. 방법론 축 재정리

### 1.1 Retrieval / Correspondence 축
- 대표: PatchCore, GraphCore, RegAD, CARL, FOCT, Search is All You Need, UniVAD
- 공통점: support-query 사이의 대응 관계를 얼마나 잘 잡느냐가 핵심
- 현재 한계:
  - patch retrieval은 문맥을 잃기 쉽다
  - registration은 변형이 큰 구조적 anomaly나 texture anomaly에 약하다
  - training-free search는 빠르지만 hard negative 억제가 약하다

### 1.2 Prompt / Semantic Alignment 축
- 대표: PromptAD, AnoPLe, InCTRL, KAG-prompt, One-for-All, FineGrainedAD
- 공통점: anomaly를 semantic boundary 또는 prompt-induced comparison으로 푼다
- 현재 한계:
  - image-level semantics와 pixel-level anomaly granularity 사이 간극이 남아 있다
  - prompt가 강해질수록 localization이 아니라 classification 쪽으로 기운다
  - normal-only prompt 학습은 가능하지만 "어떤 abnormal semantics가 필요한가"는 여전히 불안정하다

### 1.3 Foundation / Training-free 축
- 대표: AnomalyDINO, UniVAD, VisionAD, FoundAD
- 공통점: backbone adaptation보다 feature reading/inference design에 집중
- **VisionAD**: DINOv2-Reg + dual augmentation + NN search로 1-shot MVTec 97.4% (현 SOTA)
- **FoundAD**: manifold projection으로 97.8M params의 효율적 SOTA
- 현재 한계:
  - 좋은 backbone을 찾는 것과 좋은 FSAD 원리를 찾는 것이 혼재돼 있다
  - strong encoder 성능이 novelty를 가리는 경우가 많다 (VisionAD ablation: backbone +3.8pp)
  - training-free라도 support organization과 matching rule이 여전히 hand-crafted이다
  - VisionAD의 augmentation 전략이 hand-crafted → learnable alternative 미탐색
  - FoundAD의 manifold projection은 support 자체를 우회 → support 구조 미활용

### 1.4 Generation / Reconstruction 축
- 대표: FastRecon, One-to-Normal, SeaS, POUTA, Text-Guided VAE
- 공통점: 부족한 support를 reconstruction 또는 generation으로 보완
- 현재 한계:
  - 생성 품질이 낮으면 detector 전체가 흔들린다
  - anomaly generation이 실제 anomaly distribution을 얼마나 잘 반영하는지 검증이 약하다
  - 생성 모듈이 붙는 순간 실용성이 떨어지기 쉽다

### 1.5 Meta / Generalist 축
- 대표: Metaformer, MetaUAS, MetaCAN, InCTRL, One-for-All, UniADC
- 공통점: 카테고리별 모델 대신 transferable FSAD capability를 학습
- 현재 한계:
  - generalization이 좋아질수록 support-specific sensitivity가 떨어질 수 있다
  - unified setting에서 category semantics는 강해지지만 defect localization은 흐려질 수 있다

---

## 2. 반복적으로 드러난 구조적 병목

### 2.1 Support가 적은 문제가 아니라, support를 조직하는 원리가 약하다
대부분의 최신 방법은 backbone 자체보다 support usage를 개선한다. 그러나 support 간 공통 구조, query-conditional relevance, support diversity를 함께 다루는 방법은 아직 부족하다.
- VisionAD: augmentation으로 양적 확장 → 구조적 조직 아님
- FoundAD: manifold projection으로 support를 implicit 활용 → support 내부 구조 미반영
- D2-4FAD: learn-to-weight로 개별 support 가중 → support 간 관계 미모델링

### 2.2 Image-level 향상은 많지만 pixel-level 원리는 아직 분산돼 있다
VLM 계열은 semantic understanding이 강하지만 local precision이 약하고, registration/retrieval 계열은 local 대응은 잘하지만 semantic robustness가 약하다. 두 축이 아직 통합되지 않았다.

### 2.3 Strong encoder에 의존한 성능 향상이 많아 방법론적 insight가 흐려진다
DINOv2, CLIP, 최근 foundation encoder는 성능을 강하게 끌어올리지만, 왜 특정 inference rule이 유효한지 설명이 약한 경우가 많다. 연구로서 의미를 가지려면 encoder 교체가 아니라 principle을 분리해 보여줘야 한다.
- VisionAD ablation에서 backbone 교체가 +3.8pp → 방법론 기여(augmentation +2.1pp)보다 큼
- FoundAD도 DINOv3 의존도가 매우 높음 (ablation에서 backbone에 따라 성능 변동 큼)

### 2.5 Vision-only paradigm이 VLM을 압도하면서, FSAD에서 text의 역할이 재정의 필요
FoundAD, VisionAD 모두 CLIP text branch 없이 VLM 기반 방법을 상회. 그러나 text가 정말 불필요한지, 아니면 현재 text 활용 방식이 suboptimal한 것인지 명확하지 않다.

### 2.4 대부분 offline benchmark 중심이다
Few-shot Online AD, UniADC 정도를 제외하면 실제 deployment 요소인 distribution shift, support refresh, defect taxonomy evolution을 거의 다루지 않는다.

---

## 3. 연구 가치가 높은 갭

### Gap A. Semantic-contextual correspondence의 통합 부재
- 현황:
  - retrieval/registration은 local correspondence에 강하다
  - VLM/prompt는 semantic prior에 강하다
- 문제:
  - 현재 방법들은 둘 중 하나만 강한 경우가 많다
- 학술적 의미:
  - "few-shot anomaly는 semantic mismatch인가, structural mismatch인가"라는 핵심 질문을 직접 다룰 수 있다
- 실험 가능한 가설:
  - support-query correspondence를 semantic prior로 gating하면 structural anomaly와 texture anomaly를 동시에 더 안정적으로 잡을 수 있다

### Gap B. Support set의 내부 구조를 explicit하게 모델링하는 방법 부족
- 현황:
  - memory bank, prototype, prompt token, support weighting이 따로 발전했다
- 문제:
  - support 간 공통성과 예외성을 동시에 모델링하는 unified principle이 없다
- 학술적 의미:
  - few-shot의 본질인 "적은 샘플에서 어떤 구조가 invariant인가"를 직접 시험할 수 있다
- 실험 가능한 가설:
  - support set을 hypergraph 또는 query-conditioned relation structure로 모델링하면 1-shot/2-shot에서 특히 이득이 커진다

### Gap C. Pixel-level localization principle이 아직 정리되지 않음
- 현황:
  - FineGrainedAD, SOWA, FOCT, RegAD가 각각 다른 방식으로 localization을 개선한다
- 문제:
  - 왜 어떤 방법이 localization에 유효한지 공통 원리가 없다
- 학술적 의미:
  - FSAD를 image-level classification의 확장으로 볼지, dense correspondence 문제로 볼지 구분하는 데 중요하다
- 실험 가능한 가설:
  - anomaly map은 feature distance보다 "locally consistent support-query disagreement"를 모델링할 때 더 안정적이다

### Gap D. Generalist FSAD와 category-specific sensitivity 사이 trade-off가 미정리
- 현황:
  - InCTRL, One-for-All, MetaCAN, UniVAD는 unified/generalist로 간다
- 문제:
  - generalist 능력이 높아질수록 특정 카테고리의 미세 defect sensitivity가 어떻게 변하는지 체계적 분석이 부족하다
- 학술적 의미:
  - universal model이 실제 few-shot setting의 최적 해인지 검증할 수 있다
- 실험 가능한 가설:
  - global shared model 위에 lightweight category-conditioned adapter 또는 support-conditioned calibration을 두면 두 마리 토끼를 잡을 수 있다

### Gap E. Offline few-shot benchmark에서 실제 운영 setting으로의 확장 부족
- 현황:
  - 대부분 static support, static taxonomy, static domain 가정
- 문제:
  - 실제 현장에서는 정상 패턴이 점진적으로 변하고 anomaly taxonomy도 고정돼 있지 않다
- 학술적 의미:
  - FSAD를 "few reference" 문제에서 "continual normality modeling" 문제로 확장할 수 있다
- 실험 가능한 가설:
  - support memory를 online하게 갱신하되 anomaly contamination을 억제하는 구조가 deployment robustness를 유의미하게 높인다

---

## 4. 이 저장소에 특히 맞는 유망 방향

### 방향 1. Support Structure Modeling
- 제안:
  - support 이미지를 독립 샘플로 보지 말고 relation structure로 모델링
- 이유:
  - 현재 코드베이스의 PatchCore/feature matching 계열 확장과 잘 맞는다
  - ablation-friendly하게 relation module on/off가 가능하다

### 방향 2. Semantic-Guided Local Matching
- 제안:
  - CLIP/DINO 계열 semantic prior로 local retrieval 또는 registration을 gating
- 이유:
  - foundation model 흐름을 따르면서도 단순 backbone swap이 아닌 방법론적 질문을 만들 수 있다
  - texture/object/logical anomaly 간 차이를 분석하기 좋다

### 방향 3. Query-Conditional Support Calibration
- 제안:
  - 모든 support를 동등하게 쓰지 않고 query relevance로 dynamic weighting
- 이유:
  - One-for-All, D24FAD, FOCT의 흐름을 industrial FSAD에 맞춰 단순화할 수 있다
  - 1-shot/2-shot에서 효과를 검증하기 쉽다

---

## 5. 추천 우선순위

### 1순위: Semantic-guided local matching
- 이유: novelty와 실험 가능성의 균형이 가장 좋다
- 최소 실험:
  - baseline: PatchCore 또는 existing retrieval baseline
  - variant: semantic gating on/off
  - split: texture vs object category, 1/2/4-shot

### 2순위: Query-conditional support calibration
- 이유: 구현 비용이 낮고 ablation이 명확하다
- 최소 실험:
  - uniform support aggregation vs learned/query-conditioned weighting

### 3순위: Support structure modeling
- 이유: 학술적 의미는 크지만 설계 자유도가 커서 실패 원인 분리가 어렵다

---

## 6. 피해야 할 방향

- 단순히 backbone을 DINO/CLIP/새 foundation encoder로 교체하는 것만으로는 연구 가치가 약하다
- diffusion 또는 VLM을 붙였다는 이유만으로 novelty를 주장하기 어렵다
- generalized/unified를 내세우더라도 support-specific sensitivity 분석이 없으면 insight가 빈약하다

---

## 관련 노트
- [종합 서베이](./2026-03-10_comprehensive_survey.md)
- [논문 아웃라인](./2026-03-10_paper_outline_support_memory.md)
- [PatchCore](./papers/patchcore.md)
- [GraphCore](./papers/graphcore.md)
- [PromptAD](./papers/promptad.md)
- [UniVAD](./papers/univad.md)
- [One-for-All](./papers/one_for_all.md)
- [VisionAD](./papers/search_is_all_you_need.md) — training-free SOTA, 최강 baseline
- [FoundAD](./papers/foundation_visual_encoders.md) — vision-only manifold projection
- [D2-4FAD](./papers/dual_distillation.md) — query-conditional support calibration
- [UniADC](./papers/uniadc.md) — detection + classification 통합
