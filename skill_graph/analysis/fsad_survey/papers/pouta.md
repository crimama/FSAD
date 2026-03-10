# POUTA — Produce Once, Utilize Twice for Anomaly Detection

> 2023 | Reconstruction + Discriminative Reuse | Efficient AD / FSAD-capable

## 핵심 방법론
재구성 기반 anomaly detection에서 encoder와 decoder representation을 한 번 생성한 뒤, 이를 재구성 경로와 판별 경로에 동시에 재사용하는 방식. 원본과 재구성 이미지의 symmetric representation discrepancy를 coarse-to-fine하게 정제해 anomaly localization을 수행한다.

## Architecture
- **Reconstructive network**: 정상 이미지 복원
- **Representation reuse**: encoder feature는 원본 표현, decoder feature는 복원 표현으로 재활용
- **Coarse-to-fine calibration**: 상위 semantic representation으로 각 판별 layer를 보정
- **Scoring**: symmetric discrepancy 기반 anomaly map

## 핵심 설계
- **Produce once, utilize twice**: feature를 중복 추출하지 않아 효율이 높다.
- **Reconstruction + discrimination 결합**: 재구성 네트워크만으로도 더 정교한 localization 신호를 얻는다.
- **FSAD transferability**: 별도 few-shot 특화 모듈 없이도 few-shot setting에서 강한 결과를 보였다.

## 성능
- 기존 재구성 기반 방법 대비 정확도와 효율을 함께 개선
- 원문은 별도 special design 없이도 기존 few-shot anomaly detection 방법보다 더 나은 성능을 보고

## 핵심 인사이트
- FSAD 개선이 항상 새로운 detector 구조에서만 오는 것은 아니며, **이미 계산한 표현을 어떻게 재사용하느냐**도 중요한 연구 축이다.
- 재구성과 판별의 중복 계산을 줄이는 방향은 실용성과 학술성을 동시에 가진다.

## 관련 노트
- [나머지 논문 요약](./remaining_papers.md)
- [SeaS](./seas.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
