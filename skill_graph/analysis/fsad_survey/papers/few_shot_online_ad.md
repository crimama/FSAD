# Few-shot Online Anomaly Detection and Segmentation

> 2024 | Online Adaptation | Streaming anomaly-aware memory

## 핵심 방법론
Few-shot support만 보고 끝나는 정적 FSAD가 아니라, 테스트 스트림이 들어오면서 normal memory를 점진적으로 갱신하는 online anomaly detection/segmentation 프레임워크. Neural Gas 기반 update로 새 정상 패턴을 흡수하면서 anomaly contamination을 억제한다.

## Architecture
- **Initial memory**: few-shot normal support로 시작
- **Streaming update**: 테스트 데이터가 순차적으로 들어올 때 memory 갱신
- **Neural Gas adaptation**: prototype를 online하게 이동/재배치
- **Scoring**: 현재 memory와의 discrepancy로 image/pixel anomaly 계산

## 핵심 설계
- **Online setting**: 실제 운영 환경의 분포 변화와 신규 정상 패턴 유입을 명시적으로 다룬다.
- **Contamination robustness**: anomaly sample이 memory를 오염시키지 않도록 update rule을 설계한다.
- **Segmentation support**: image-level뿐 아니라 pixel-level localization까지 포함한다.

## 성능
- MPDD와 BTAD의 online few-shot setting에서 기존 학습 기반 online/open-set 적응 방법보다 우수한 성능을 보고
- 핵심 기여는 정적 FSAD benchmark보다 **streaming deployment setting**을 문제 정의에 포함시켰다는 점

## 핵심 인사이트
- 산업 현장에서는 initial few-shot 성능보다도, 시간이 지나며 **정상 분포가 바뀔 때 어떻게 추적할지**가 더 중요한 문제일 수 있다.
- FSAD 연구를 offline benchmark에서 online adaptation 문제로 확장한 초기 사례로 볼 수 있다.

## 관련 노트
- [PatchCore](./patchcore.md)
- [UniVAD](./univad.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
