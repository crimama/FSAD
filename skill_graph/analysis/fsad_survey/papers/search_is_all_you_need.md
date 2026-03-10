# VisionAD — Search is All You Need for Few-shot Anomaly Detection

> arxiv: 2504.11895 | 2025 | Fudan Univ 등
> Training-free | DINOv2-Register NN search | Pseudo multi-view augmentation

## 핵심 방법론
**Training-free** nearest-neighbor search 프레임워크. DINOv2-Register ViT-L/14를 backbone으로 사용하여, prompt engineering이나 multi-modal 모델 없이 순수 vision feature의 NN search만으로 FSAD SOTA를 달성. Dual augmentation(support + query)과 multi-layer fusion이 핵심 기법.

## Architecture
- **Backbone**: DINOv2-Register ViT-Large/14 (frozen, no training)
- **Dual Augmentation**:
  - Support augmentation: rotation/flip으로 support 다양성 증가
  - **Pseudo Multi-View**: query와 support에 동일한 변환 적용 → 상보적 view 생성으로 single-view 맹점 보완
- **Multi-Layer Feature Integration**: ViT 중간 레이어 [4,6,8,...,19] 결합 → low-freq global + high-freq local 정보 동시 포착
- **Class-Aware Visual Memory Bank**: CLS token 유사도로 클래스 결정 → 해당 클래스 내 patch-level NN search

## 핵심 설계
- **Completely training-free**: fine-tuning, prompt learning, adapter 학습 없음
- **Vision-only**: CLIP/VLM의 language branch 불필요 → pure vision feature가 AD에 더 적합
- **Augmentation as enrichment**: few-shot support의 coverage 부족을 augmentation으로 보완
- **Search-centric FSAD**: few-shot bottleneck을 representation learning보다 memory coverage 문제로 봄

## 성능

### MVTec-AD
| Shot | I-AUROC | P-AUROC | PRO |
|------|---------|---------|-----|
| 1    | 97.4±0.4 | 96.2±0.2 | 92.5±0.3 |
| 4    | 98.6±0.1 | 96.9±0.1 | 93.7±0.1 |

### VisA
| Shot | I-AUROC | P-AUROC | PRO |
|------|---------|---------|-----|
| 1    | 94.8±1.0 | 97.6±0.3 | 91.6±0.8 |
| 4    | 95.7±0.3 | 98.0±0.0 | 92.5±0.2 |

### 기존 SOTA 대비 (1-shot I-AUROC)
- PatchCore: 83.4% / 79.9% (MVTec/VisA)
- WinCLIP: 93.1% / 83.8%
- PromptAD: 94.6% / 86.9%
- KAG-prompt: 95.8% / 91.6%
- **VisionAD: 97.4% / 94.8%** → 전 baseline 상회

## Ablation (MVTec I-AUROC, 누적)
| Component | AUROC | Δ |
|-----------|-------|---|
| Baseline (basic NN) | 89.0% | — |
| + DINOv2-Reg backbone | 92.8% | +3.8 |
| + Multi-layer fusion | 95.0% | +2.2 |
| + Support augmentation | 97.1% | +2.1 |
| + Pseudo multi-view query aug | 97.6% | +0.5 |

- **Support augmentation이 단일 최대 기여** (+2.1pp)
- Backbone 업그레이드도 큰 기여 (+3.8pp)

## 핵심 인사이트
- **"Vision-only is enough"**: VLM/prompt 기반 방법보다 순수 vision NN search가 우수
- **Pseudo Multi-View**: 동일 변환을 query-support 모두에 적용하는 단순하지만 효과적인 전략
- **매우 높은 baseline bar**: 이후 새 FSAD 방법은 이 수준을 넘거나, VisionAD가 본질적으로 못하는 것을 해결해야 함
- Foundation feature가 충분히 강할 때, FSAD 개선은 복잡한 adaptation보다 **retrieval design**에서 나올 수 있다

## Limitations
- DINOv2-Reg에 대한 강한 의존 → backbone 성능이 방법론적 novelty를 가림
- Training-free이므로 task-specific adaptation 불가
- Augmentation 전략이 hand-crafted
- NN search의 computational cost가 support/augmentation 수에 비례

## 관련 노트
- [FoundAD](./foundation_visual_encoders.md) — "vision-only" 철학 공유, 단 projection 학습
- [AnomalyDINO](./anomalydino.md) — DINOv2 기반 training-free
- [PatchCore](./patchcore.md) — NN search의 원형
- [UniVAD](./univad.md) — training-free unified AD
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
