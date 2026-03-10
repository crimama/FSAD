# PromptAD — Learning Prompts with only Normal Samples for Few-Shot AD

> CVPR 2024 | CLIP Prompt Learning | Normal-only 학습

## 핵심 방법론
CLIP 기반으로 normal과 anomaly prompt를 학습하되, normal 샘플만 사용. "Semantic concatenation" 전략으로 normal-anomaly prompt pair를 생성. Multi-scale visual feature로 pixel-level localization.

## Architecture
- **Base model**: CLIP (frozen visual + text encoder)
- **Learnable**: Normal prompt + Anomaly prompt (text tokens)
- **Multi-scale**: Intermediate CLIP visual features로 anomaly map 생성
- **No anomaly data required**: 합성 anomaly도 불필요

## Key Design
- **Semantic concatenation**: Normal prompt가 anchor, anomaly prompt는 complement
- Normal prompt → "이 카테고리의 정상은 이렇게 생겼다"
- Anomaly prompt → Normal의 반대편으로 자동 학습

## 성능
- MVTec AD (1-shot): Image AUROC ~94-96%, Pixel AUROC ~96%+
- MVTec AD (4-shot): 더 높은 성능
- VisA에서도 strong

## 핵심 인사이트
- Anomaly prompt를 normal 데이터만으로 학습 가능 — CLIP embedding space의 구조 활용
- 합성 anomaly 생성이 불필요하다는 점이 실용적
- Normal/anomaly의 경계가 CLIP space에서 자연스럽게 형성됨

## 관련 노트
- [AnoPLe](./anople.md) — Bi-directional prompt learning
- [One-for-All](./one_for_all.md) — Instance-induced prompt
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
