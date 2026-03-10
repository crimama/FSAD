# FOCT — Few-shot Industrial Anomaly Detection with Foreground-aware Online Conditional Transport

> ACM MM 2024 | Optimal Transport | Foreground-aware support adaptation

## 핵심 방법론
Few-shot support와 query 사이의 feature alignment를 foreground-aware conditional optimal transport로 수행하는 FSAD 방법. defect detection에 핵심인 foreground 영역을 우선적으로 맞추고, online transport로 test-time query마다 대응 관계를 갱신한다.

## Architecture
- **Feature extractor**: support/query feature 추출
- **Foreground-aware semantic construction**: 배경보다 defect 관련 foreground 표현 강화
- **Online conditional transport**: query 조건부로 support-query 대응 행렬 계산
- **Scoring**: transport 후 residual inconsistency 기반 anomaly score

## 핵심 설계
- **Foreground priority**: few-shot setting에서는 support 수가 적기 때문에 배경 noise를 줄이는 것이 중요하다.
- **Online adaptation**: query마다 transport plan을 다시 구해 정렬을 세밀하게 조정한다.
- **Transport-based correspondence**: registration보다 더 유연한 support-query matching을 제공한다.

## 성능
- MVTec AD와 VisA에서 당시 SOTA 대비 우수한 image/pixel anomaly detection 성능을 보고
- 핵심 강점은 support 수가 매우 적을 때도 foreground correspondence를 안정적으로 찾는 점

## 핵심 인사이트
- Few-shot AD에서는 global similarity보다 **어느 영역을 대응시킬 것인가**가 더 중요할 수 있다.
- Optimal transport는 registration과 memory matching 사이의 중간 지점으로, **soft correspondence**를 제공한다는 학술적 의미가 있다.

## 관련 노트
- [RegAD](./regad.md)
- [CARL](./carl.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
