# CARL — Few-Shot Anomaly Detection via Category-Agnostic Registration Learning

> 2024 | Category-agnostic Registration | Single-model FSAD

## 핵심 방법론
RegAD의 registration 관점을 category-agnostic representation learning으로 확장한다. 다양한 카테고리의 정상 이미지 쌍을 self-supervised registration proxy task로 학습해, 새 카테고리에서도 재학습 없이 support-query 정렬 기반 anomaly scoring을 수행한다.

## Architecture
- **Backbone**: category-agnostic feature extractor
- **Proxy task**: 정상 이미지 간 registration 학습
- **Test-time input**: query image + 같은 카테고리의 few-shot normal support set
- **Scoring**: registration 후 support-query feature residual

## 핵심 설계
- **Single model for all categories**: 카테고리별 별도 모델 학습이 필요 없다.
- **Registration as self-supervision**: "정상을 서로 잘 맞추는 능력"을 학습해 anomaly detection으로 전이한다.
- **Few-shot inference**: 새로운 카테고리에서는 support set만 바꿔서 바로 적용한다.

## 성능
- MVTec, MPDD에서 당시 SOTA 대비 각각 **11.3%**, **8.3%** 개선을 보고
- 논문의 핵심 포인트는 수치 자체보다도, **novel category에 대해 fine-tuning 없는 단일 FSAD 모델**을 제시했다는 점

## 핵심 인사이트
- RegAD의 "정렬 후 차이" 아이디어는 category-specific 모델 없이도 유지될 수 있다.
- FSAD를 카테고리별 normality modeling이 아니라 **category-agnostic correspondence learning** 문제로 재정의한다.
- 실제 산업 환경에서는 카테고리 추가 때마다 모델을 다시 학습하지 않아도 된다는 점에서 운영상 의미가 크다.

## 관련 노트
- [RegAD](./regad.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
