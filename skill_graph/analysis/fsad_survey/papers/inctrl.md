# InCTRL — Toward Generalist AD via In-context Residual Learning

> CVPR 2024 | In-context Learning | Generalist Model

## 핵심 방법론
Few-shot normal samples를 "in-context prompts"로 사용하는 generalist AD.
Test image와 few-shot reference 간의 residual feature를 학습하여 정상으로부터의 deviation 포착.

## Architecture
- **Visual encoder**: test image + reference set 동시 처리
- **Residual learning module**: query feature - reference feature aggregation
- **Single model**: 카테고리별 학습 없이 모든 카테고리 처리
- **Cross-domain**: Industrial, medical 등 다양한 도메인에서 동작

## Key Design
- In-context learning: few-shot normals = LLM의 in-context examples와 동일한 역할
- Residual = deviation from normality
- Generalist: 하나의 모델로 모든 카테고리

## 성능
- MVTec AD (few-shot): Image AUROC ~95%+
- VisA: Competitive
- 핵심 장점: cross-category 일반화

## 핵심 인사이트
- AD를 in-context learning 문제로 재정의
- "정상을 외우는 것"이 아니라 "정상과의 차이를 감지하는 능력"을 학습
- Generalist model 트렌드의 핵심 논문

## 관련 노트
- [PromptAD](./promptad.md) — 또 다른 CVPR 2024 few-shot AD
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
