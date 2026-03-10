# UniADC — A Unified Framework for Anomaly Detection and Classification

> arxiv: 2511.06644 | 2025 | Beijing Univ of Posts and Telecommunications
> Unified AD + Classification | Training-free anomaly synthesis | Implicit-normal discriminator

## 핵심 방법론
이상 탐지와 이상 분류를 별개 태스크로 보지 않고 하나의 프레임워크로 통합한다. Training-free controllable inpainting(Stable Diffusion + BrushNet)으로 category-specific anomaly를 합성하고, **Implicit-Normal Discriminator(IND)**가 "정상 = 모든 이상 개념의 부재"로 모델링하여 detection + localization + classification을 동시에 수행한다.

## Architecture
- **Anomaly Synthesis Pipeline**:
  - GAP-Lib (Geometric Anomaly Prototype Library): 8종 mask × 3 size = 24 변형
    - Rectangle, Line, Polygon, Ellipse, Hollow Ellipse, Random Brush, Perlin Noise, Foreground Mask
  - Zero-shot: text-prompt guided inpainting + AlphaCLIP-based Category Consistency Selection(CCS)
  - Few-shot: real anomaly crop → paste → diffusion repaint + SSIM-based selection
  - **Training-free**: diffusion model fine-tuning 불필요
- **Implicit-Normal Discriminator (IND)**:
  - Backbone: CLIP 또는 DINOv2 → multi-scale feature extraction
  - Feature Fusion Network(FFN) → dense feature `f ∈ R^(H'×W'×C)`
  - Per-pixel anomaly score: `s_y(h,w) = σ(⟨f(h,w), g_y⟩ / ε)` (g_y = anomaly category embedding)
  - Detection map = mean of all category classification maps
  - **핵심**: explicit normal embedding 제거 → normal pixel imbalance 문제 해결
- **Loss**: Focal + Dice (detection) + λ·CE (classification, normal 영역 무시)

## 핵심 설계
- **Unified task definition**: "어디가 이상인가" + "무슨 이상인가"를 동시에 품
- **Implicit-normal**: normal을 직접 모델링하지 않고, anomaly 개념의 부재로 정의 → ~20% classification accuracy 향상
- **Category-controllable generation**: DRAEM/CutPaste와 달리 특정 defect category의 anomaly 생성 가능
- **Few-shot setting 차이**: K_n normal + K_a anomaly (labeled) → standard FSAD (normal-only)와 다른 가정

## 성능
**주의: MVTec-FS 사용 (standard MVTec-AD와 다름). VisA 미평가.**

| Setting | Method | I-AUROC | P-AUROC | PRO | Acc | mIoU |
|---------|--------|---------|---------|-----|-----|------|
| Zero-shot (K_n=2) | WinCLIP | 93.20 | 94.43 | 86.47 | 40.75 | 25.17 |
| Zero-shot (K_n=2) | UniADC(DINO) | **97.09** | **97.04** | **92.15** | **74.74** | **36.66** |
| Few-shot (K_n=2, K_a=1) | NAGL+ZipAF | 96.41 | 96.56 | 92.96 | 57.78 | 42.34 |
| Few-shot (K_n=2, K_a=1) | UniADC(DINO) | **98.56** | **98.90** | **92.48** | **86.85** | **51.49** |

- MTD, WFDD에서도 일관된 개선

## 핵심 인사이트
- 산업 현장에서는 anomaly score만으로 끝나지 않고 **defect type 분류**까지 연결돼야 한다
- FSAD 연구가 detection-only에서 **detection + diagnosis** 문제로 확장되는 흐름
- "Normal = absence of anomaly concepts" 모델링이 pixel imbalance 해결에 효과적
- Diffusion 기반 anomaly synthesis가 training-free로 가능 → data augmentation baseline으로 유용

## Limitations
- **Standard FSAD와 다른 setting**: K_a=1 (labeled anomaly 필요) → 순수 few-shot normal-only 방법과 직접 비교 불가
- MVTec-FS ≠ MVTec-AD → 기존 FSAD 벤치마크와 수치 비교 어려움
- VisA 미평가
- Anomaly category label이 필요 → 실용적 제약

## 관련 노트
- [One-for-All](./one_for_all.md) — unified multi-class AD
- [FineGrainedAD](./fine_grained_vl.md) — fine-grained localization
- [FoundAD](./foundation_visual_encoders.md) — vision-only 접근과 대조
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
