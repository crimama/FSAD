# KAG-prompt — Kernel-Aware Graph Prompt Learning for Few-Shot Anomaly Detection

> AAAI 2025 | Graph Prompt Learning | Cross-layer contextual reasoning

## 핵심 방법론
CLIP 계열 prompt learning에 그래프 기반 cross-layer reasoning을 결합한 FSAD 프레임워크. 서로 다른 vision layer feature를 anomaly scale이 다른 노드로 보고, 이들 사이 관계를 message passing으로 통합해 anomaly prediction을 강화한다.

## Architecture
- **Base model**: vision-language backbone 기반 prompt learning
- **Kernel-aware hierarchical graph**: 여러 visual layer feature를 노드로 구성
- **Edges**: 레이어 간 상호작용과 anomaly scale 관계를 표현
- **Message passing**: cross-layer contextual information 집계
- **Image-level scoring**: multi-level anomaly signal fusion

## 핵심 설계
- **Cross-layer reasoning**: prompt만 학습하는 기존 방법이 놓치던 layer 간 문맥 관계를 명시적으로 모델링한다.
- **Kernel-aware graph**: 작은 defect와 큰 defect를 서로 다른 receptive field 관점에서 동시에 포착한다.
- **Multi-level fusion**: pixel map뿐 아니라 image-level score도 여러 anomaly signal을 함께 사용해 안정화한다.

## 성능
- MVTec AD, VisA에서 image-level/pixel-level 모두 SOTA를 보고
- 후속 비교 문헌 기준으로 1-shot MVTec Image AUROC 약 **96.6**, VisA Image AUROC 약 **92.7** 수준으로 인용됨

## 핵심 인사이트
- VLM 기반 FSAD의 병목은 text prompt 설계만이 아니라 **visual layer 간 관계를 어떻게 묶느냐**에 있다.
- Patch scale이 다른 anomaly를 한 번에 다루려면 단일 layer보다 **cross-layer graph reasoning**이 유리하다.

## 관련 노트
- [PromptAD](./promptad.md)
- [One-for-All](./one_for_all.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
