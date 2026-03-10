# D2-4FAD — Dual Distillation for Few-Shot Anomaly Detection

> ICLR 2026 | arxiv: 2603.01713 | Xidian Univ, TU Munich
> Dual distillation (teacher-student + student self-distillation) | Learn-to-weight support

## 핵심 방법론
Frozen pretrained encoder(teacher)와 learnable decoder(student)를 사용하는 dual distillation 프레임워크. Teacher-student distillation(query에 대한 일반적 정상성 학습)과 student self-distillation(support-query 간 task-specific 정상성 학습)을 결합한다. Learn-to-weight 메커니즘으로 각 support 이미지의 query 관련성을 동적으로 가중한다.

## Architecture
- **Teacher**: Frozen pretrained encoder (WideResNet-50 기본). Multi-scale feature 추출
- **Student**: Learnable decoder. Multi-scale feature reconstruction
- **Learn-to-weight**: 1×1 conv (linear projection) → scaled dot-product softmax attention
  - query feature와 support feature 간 유사도로 support별 가중치 동적 결정
- **Training**: Episodic, normal-only. Task-agnostic 학습
- **Loss**: `L = λ·L_tsd + L_ssd_l2w` (λ=0.1 최적)
  - L_tsd: teacher-student cosine similarity loss (query)
  - L_ssd_l2w: weighted student self-distillation loss (support→query)

## 핵심 설계
- **Dual distillation**: teacher-student(일반 정상성) + self-distillation(support-specific 정상성) 분리
- **Query-conditional support weighting**: 모든 support를 동등하게 쓰지 않고 query relevance로 가중
- **Episodic training**: 학습 시 seen task에서 general FSAD capability를 학습 → unseen task에 전이
- **Architectural simplicity**: memory bank, normalizing flow, synthetic anomaly 없이 encoder-decoder만 사용

## 성능
**주의: MVTec-AD/VisA 미평가. Medical imaging 전용 벤치마크.**

| Dataset | 2-shot | 4-shot | 8-shot |
|---------|--------|--------|--------|
| HIS (histopathology) | 94.2 | 94.2 | 94.3 |
| LAG (glaucoma) | 94.7 | 96.2 | 97.3 |
| APTOS (retinopathy) | 100.0 | 100.0 | 100.0 |
| RSNA (chest X-ray) | 88.9 | 97.9 | 99.2 |
| Brain Tumor (MRI) | 95.5 | 95.3 | 95.5 |

- Image-level AUROC만 보고. Pixel-level AUROC, PRO 없음
- MVFA-AD, InCTRL, AnomalyGPT 등 18개 baseline 대비 우수

## Ablation 요약
- **Component**: SSD(self-distillation)이 단일 최강 (HIS 2-shot: 90.0 vs TSD-only 66.1). 세 모듈 결합이 최적
- **Backbone**: WideResNet-50 >> ResNet-50 >> Swin-B (Swin-B: 64.7 vs WRN-50: 96.7 on HIS 2-shot)
- **Pre-training data**: RadImageNet ≈ ImageNet (의료 도메인에서 약간 우위)
- **L2W variant**: Scaled dot-product > Gaussian, Embedded Gaussian, Concatenation
- **Support robustness**: Fixed vs random support 선택 시 분산 최소 → 안정적

## 핵심 인사이트
- **Self-distillation이 teacher-student보다 중요**: support-specific 정상성 학습이 일반 정상성보다 few-shot에서 기여가 큼
- **Learn-to-weight는 query-conditional support calibration**: 아웃라인의 "방향 3: Query-Conditional Support Calibration"과 직접 관련
- **Medical domain에서의 FSAD**: 산업 검사 외 의료 영상에서도 동일한 few-shot 패러다임 유효
- **WRN-50이 Swin보다 강함**: ViT/Swin이 항상 최적은 아님 (도메인 의존)

## Limitations
- MVTec-AD/VisA 미평가 → 산업 FSAD와 직접 비교 불가
- Pixel-level localization 미보고
- Episodic training에 다수 학습 task 필요

## 관련 노트
- [FoundAD](./foundation_visual_encoders.md) — 같은 ICLR 2026, support 활용 방식이 대조적
- [InCTRL](./inctrl.md) — 유사한 episodic/generalist 접근
- [RegAD](./regad.md) — registration 기반 baseline
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
