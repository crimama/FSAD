# One-for-All — Towards Universal Industrial Anomaly Detection and Reasoning with Instance-Induced Vision-Language Prompting

> ICLR 2025 | Vision-Language Prompting | Universal multi-class FSAD

## 핵심 방법론
단일 vision-language 모델로 여러 산업 카테고리를 함께 처리하는 universal AD 프레임워크. 카테고리별 고정 prompt 대신, 입력 인스턴스에 따라 prompt를 동적으로 생성하는 **instance-induced prompting**을 사용해 few-shot multi-class anomaly detection과 reasoning을 동시에 다룬다.

## Architecture
- **Base model**: CLIP 계열 vision-language backbone
- **Category-aware prompt bank**: 카테고리별 semantic prior 저장
- **Instance-specific prompt generator**: 입력 이미지 feature에 따라 prompt를 동적으로 생성
- **Prototype guidance module**: support prototype를 이용해 normal/anomaly 경계 보정

## 핵심 설계
- **Dynamic prompting**: 카테고리 이름만 넣는 정적 prompt보다, 현재 입력의 문맥을 반영한 prompt가 더 강하다.
- **Universal setting**: 카테고리별 분리 모델 대신 하나의 모델로 다중 클래스 처리
- **Reasoning-aware AD**: anomaly score뿐 아니라 defect semantics를 설명 가능한 방향으로 확장

## 성능
- MVTec AD 1-shot: Image AUROC **95.7**, Pixel AUROC **96.7**, PRO **90.3**
- MVTec AD 4-shot: Image AUROC **96.1**, Pixel AUROC **97.0**, PRO **91.2**
- VisA 1-shot: Image AUROC **86.7**, Pixel AUROC **97.2**
- VisA 4-shot: Image AUROC **88.3**, Pixel AUROC **97.4**

## 핵심 인사이트
- Few-shot VLM 기반 AD의 병목은 backbone보다도 **prompt의 정적성**에 있다.
- Support sample이 단순 reference가 아니라 **instance-conditioned prompt 생성 신호**로 쓰일 수 있다.
- Universal AD와 anomaly reasoning을 함께 묶는 흐름에서 대표적인 2025년형 확장이다.

## 관련 노트
- [PromptAD](./promptad.md)
- [InCTRL](./inctrl.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
