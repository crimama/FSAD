# FineGrainedAD — Towards Fine-Grained Vision-Language Alignment for Few-Shot Anomaly Detection

> 2025 | Fine-grained VLM Alignment | Component-level anomaly localization

## 핵심 방법론
기존 VLM 기반 FSAD의 image-level prompt가 patch-level anomaly와 semantic mismatch를 일으킨다는 문제를 겨냥한다. 이를 위해 **Multi-Level Fine-Grained Semantic Caption (MFSC)** 과 **FineGrainedAD**를 제안하고, component-level semantic prompt와 visual region을 다층적으로 정렬한다.

## Architecture
- **MFSC**: anomaly detection 데이터셋용 fine-grained semantic caption 자동 생성
- **MLLP**: image-level prompt를 multi-level learnable prompt로 분해/확장
- **MLSA**: progressive region aggregation과 multi-level alignment training 수행
- **Inference**: patch마다 가장 적절한 prompt token을 동적으로 할당

## 핵심 설계
- **Component-level semantics**: defect를 물체 전체가 아니라 부품/구성요소 단위로 기술한다.
- **Multi-level prompt learning**: coarse-to-fine semantic granularity를 prompt 자체에 내장한다.
- **No extra inference cost**: fine-grained localization을 추가 추론 모듈 없이 수행한다.

## 성능
- MVTec-AD와 VisA few-shot setting에서 superior overall performance와 SOTA급 localization을 보고
- 논문의 초점은 image-level classification보다 **patch-level localization 품질** 개선에 가깝다.

## 핵심 인사이트
- VLM 기반 FSAD의 병목은 anomaly text가 부족한 것이 아니라, **텍스트의 semantic granularity가 patch 수준과 맞지 않는 것**이다.
- Fine-grained caption을 자동 생성해 prompt 학습에 넣는 방향은 이후 anomaly reasoning 연구와도 잘 연결된다.

## 관련 노트
- [PromptAD](./promptad.md)
- [KAG-prompt](./kag_prompt.md)
- [One-for-All](./one_for_all.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
