# RFS Energy — Anomaly Detection of Defect using Energy of Point Pattern Features within Random Finite Set Framework

> 2023 | Statistical Modeling | Local point-pattern FSAD

## 핵심 방법론
전역 CNN feature 대신 local point-pattern feature를 추출하고, 이를 random finite set (RFS)로 모델링해 anomaly score를 계산한다. 기존 likelihood 대신 **RFS energy**를 anomaly score로 사용해 few-shot 분포 추정의 불안정성을 줄인다.

## Architecture
- **Feature extraction**: transfer learning 기반 local/point feature 추출
- **Statistical model**: point-pattern feature를 multivariate Gaussian 기반 RFS로 표현
- **Scoring**: RFS likelihood가 아니라 energy 함수 사용
- **Target**: 산업 결함과 다중 뷰 needle defect detection

## 핵심 설계
- **Local geometry emphasis**: global feature가 놓치는 기하학적 정보를 local point 패턴으로 포착한다.
- **Lightweight estimation**: heavy training 없이도 few-shot normal distribution을 다룰 수 있다.
- **Few-shot robustness**: support 수가 적을 때 likelihood 추정보다 energy score가 더 안정적이라는 가설을 검증한다.

## 성능
- MVTec AD few-shot setting에서 당시 SOTA 대비 우수한 성능을 보고
- needle real-world dataset에서도 DifferNet 대비 우수한 결과를 보고

## 핵심 인사이트
- Few-shot anomaly detection의 핵심은 backbone 크기만이 아니라 **어떤 통계 객체를 모델링하느냐**에도 있다.
- Point-pattern + energy formulation은 memory bank와 prompt learning 외의 통계적 대안을 제공한다.

## 관련 노트
- [나머지 논문 요약](./remaining_papers.md)
- [PatchCore](./patchcore.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
