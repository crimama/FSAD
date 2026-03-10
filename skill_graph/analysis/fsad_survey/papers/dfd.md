# DFD — Dual-Path Frequency Discriminators for Few-shot Anomaly Detection

> Knowledge-Based Systems 2024 | Frequency-domain FSAD | Pseudo-anomaly discrimination

## 핵심 방법론
미세한 이상은 spatial domain보다 frequency domain에서 더 두드러질 수 있다는 관찰에 기반해, 원본 이미지를 다중 주파수 이미지로 변환하고 dual-path discriminator로 anomaly를 판별한다. 논문은 전체 프레임워크를 anomaly generation, multi-frequency information construction, fine-grained feature construction, dual-path feature discrimination의 네 부분으로 구성한다.

## Architecture
- **Anomaly generation**: pseudo anomaly 생성
- **Multi-frequency construction**: 원본 이미지를 여러 주파수 성분으로 분해
- **Fine-grained feature construction**: subtle defect 표현 강화
- **Dual-path discriminator**: frequency 관점의 joint representation 학습

## 핵심 설계
- **Frequency-first FSAD**: 눈에 잘 띄지 않는 이상을 frequency representation에서 더 분리되게 만든다.
- **Pseudo-anomaly aided learning**: normal-only few-shot setting의 supervision 부족을 합성 이상으로 보완한다.
- **Dual-path discrimination**: 서로 다른 frequency branch를 함께 사용해 anomaly signal을 강화한다.

## 성능
- MVTec AD와 VisA에서 기존 SOTA보다 더 나은 성능을 보고
- 원문 highlight는 benchmark 전반에서 current state of the art를 능가했다고 정리한다

## 핵심 인사이트
- FSAD의 정보 부족은 sample 수뿐 아니라 **representation domain 선택**의 문제이기도 하다.
- Spatial feature만으로 놓치는 subtle anomaly를 frequency branch로 보완하는 방향은 texture-heavy defect에서 특히 의미가 있다.

## 관련 노트
- [PatchCore](./patchcore.md)
- [나머지 논문 요약](./remaining_papers.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
