# RegAD — Registration Based Few-Shot Anomaly Detection

> ECCV 2022 Oral | Registration 기반 | Few-shot AD 핵심 baseline

## 핵심 방법론
Image registration을 AD의 핵심 프레임워크로 활용. Support image와 query image를 STN으로 정렬한 뒤, 정렬 후 차이(residual)가 anomaly.

## Architecture
- **2-stage pipeline:**
  1. Registration network 학습 (meta-train): 다양한 카테고리에서 이미지 정렬 학습
  2. Few-shot AD (meta-test): 새 카테고리의 support-query 정렬 → residual = anomaly
- **Feature Extractor**: Pretrained ResNet
- **Registration Module**: Spatial Transformer Network (STN)
- **Scoring**: Feature-level L2 distance after registration

## 핵심 수식
```
anomaly_map = |F(warp(support)) - F(query)|
L_reg = ||F(warped_support) - F(query)||²
```

## 성능
- MVTec AD (2-shot): Image AUROC ~83%, Pixel AUROC ~96%
- MVTec AD (4-shot): Image AUROC ~86%, Pixel AUROC ~97%
- MVTec AD (8-shot): Image AUROC ~88%

## 핵심 인사이트
- "Anomaly detection은 본질적으로 registration 문제"
- Cross-category generalization: registration 능력이 카테고리 간 transfer 가능
- Few-shot에 자연스러운 프레임워크 — support가 곧 reference
- **한계**: Texture anomaly에 약함, flexible object 정렬 실패

## 관련 노트
- [CARL](./carl.md) — Category-agnostic registration 확장
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
