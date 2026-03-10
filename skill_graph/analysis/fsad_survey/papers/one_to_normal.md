# One-to-Normal — Anomaly Personalization for Few-shot AD

> NeurIPS 2024 | Reconstruction/Generation 기반 | Personalization

## 핵심 방법론
Test image의 anomaly 영역을 정상으로 복원(personalization)한 뒤, 원본과 복원 이미지의 차이로 anomaly 검출. Few-shot reference로 personalized normal reconstruction model 구축.

## Architecture
- **Generative model**: Diffusion model 기반
- **Personalization**: Few-shot normal reference로 카테고리별 normal appearance 학습
- **Anomaly map**: |original - reconstructed_normal|

## 성능
- MVTec AD: Image AUROC ~96%+, Pixel AUROC ~97%+ (k=1~4)
- Reconstruction 기반이라 pixel-level 성능이 특히 강함

## 핵심 인사이트
- "Anomaly detection = anomaly를 정상으로 personalize한 뒤 비교"
- Feature matching 대비 richer anomaly signal 제공
- Diffusion model의 few-shot personalization 기법이 핵심 기술

## 관련 노트
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
