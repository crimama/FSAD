# FoundAD — Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors

> ICLR 2026 | arxiv: 2510.01934 | TU Munich, MCML, MVTec Software
> Foundation encoder analysis | Nonlinear projection onto natural image manifold

## 핵심 방법론
Foundation visual encoder(DINOv3 등)의 embedding space에서 이상 이미지가 "자연 이미지 매니폴드"로부터 멀어진다는 관찰을 기반으로, 경량 nonlinear projector를 학습하여 이상 feature를 정상 매니폴드 위로 사영한다. 사영 전후 거리가 anomaly score. 텍스트 프롬프트 없이 **순수 visual embedding만** 사용하여 multi-class few-shot AD를 수행.

## Architecture
- **Encoder**: 2개의 동일한 frozen foundation encoder (Anomaly-Aware / Reference)
  - DINOv3를 기본 backbone으로 사용 (DINOv2, DINO, SigLIP, CLIP도 실험)
- **Manifold Projector**: 6-layer ViT self-attention block
  - residual connection: $x_{out} = \text{Attn}(x_{in}) + x_{in}$
  - trainable params만 학습 (encoder는 frozen)
- **Anomaly Synthesis**: CutPaste 기반, foreground mask로 영역 제한 (adaptive threshold binarization)
- **Setting**: multi-class few-shot (1/2/4/8-shot)

## 핵심 설계
- **Manifold Projection in Latent Space**: pixel-level reconstruction 대신 latent space에서 사영 → 효율적
- **Encoder analysis as method design**: foundation encoder 자체가 이미 FSAD 능력을 갖고 있다는 가설을 실증
- **Minimal adaptation**: prompt tuning이나 task-specific detector 없이 단순 projection만 학습
- **Training**: 정상 이미지만 사용 (unsupervised). L2 loss로 projected feature ↔ reference feature 정렬
  - synthesis probability σ=0.5, Adam lr=0.001

## 성능

### Multi-class few-shot (Table 1)
| Dataset  | Shot | I-AUROC | P-AUROC | PRO   |
|----------|------|---------|---------|-------|
| MVTec-AD | 1    | 96.1    | —       | 92.8  |
| MVTec-AD | 4    | 97.0    | —       | 94.1  |
| VisA     | 1    | —       | 99.7   | —     |
| VisA     | 4    | —       | 99.7   | —     |

- IIPAD 대비: MVTec 1-shot I-AUROC +1.9%, PRO +3.0%
- LogSAD 대비: VisA 1-shot P-AUROC +2.2%, PRO +9.8%

### Efficiency
- **Total params**: 97.8M (vs IIPAD 1.0B, LogSAD 1.3B → ~10× 작음)
- **Inference**: 128.7 ms/image (~7.8 fps)

## Ablation 요약
- **Backbone 선택** (Table 3): DINOv3 > DINOv2 > DINO > SigLIP > CLIP > WideResNet
  - 순수 visual pretrain이 VLM보다 pixel-level 성능 우수 (CLIP은 fine-grained spatial info 약함)
- **Layer 선택** (Figure 6): DINOv3 Layer 10 (중후반)이 최적 — semantic + spatial 균형
- **Projector 설계** (Table 4): ViT attention >> MLP (self-attention이 fine-grained 패턴 포착에 유리)

## 핵심 인사이트
- Foundation model 시대의 FSAD 연구 질문은 "새 detector를 얼마나 복잡하게 만들까"보다 **encoder embedding을 어떻게 읽어낼까**로 이동
- Visual-only foundation feature가 text 도움 없이도 strong FSAD signal을 담고 있음 → VLM prompt 의존성에 도전
- Anomaly severity가 embedding distance와 직접 상관 → manifold distance 자체가 anomaly score로 작동
- Multi-class 통합 모델로 category별 개별 모델 불필요

## Limitations / 열린 질문
- Failure case 존재 (논문 Figure 7) — 어떤 조건에서 실패하는지 상세 분석 필요
- Anomaly synthesis의 품질이 성능에 미치는 영향 (CutPaste의 한계)
- DINOv3 의존도가 높음 — backbone 변경 시 성능 변동 큼

## 관련 노트
- [AnomalyDINO](./anomalydino.md) — 같은 foundation encoder 활용, training-free 접근
- [UniVAD](./univad.md) — multi-class unified AD
- [Search is All You Need](./search_is_all_you_need.md)
- [PromptAD](./promptad.md) — text prompt 기반 (FoundAD가 도전하는 방향)
- [InCTRL](./inctrl.md) — in-context learning 기반 FSAD
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
