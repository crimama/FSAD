# SeaS — Few-shot Industrial Anomaly Image Generation with Separation and Sharing

> ICCV 2025 | Few-shot Anomaly Generation | Latent diffusion fine-tuning

## 핵심 방법론
Few-shot normal image만으로 산업 이상 이미지를 생성하기 위한 latent diffusion 기반 생성 프레임워크. 정상 고유 정보와 이상 공통 정보를 분리해 다루는 **Separation and Sharing fine-tuning**으로, 데이터가 적어도 다양한 defect pattern을 합성할 수 있게 한다.

## Architecture
- **Base model**: pretrained latent diffusion model
- **Separation and Sharing fine-tuning**: 정상 카테고리 정보와 이상 표현을 분리해 업데이트
- **UA text prompt**: 카테고리별 handcrafted defect prompt 없이도 anomaly generation 유도
- **Output**: downstream detector 학습에 쓸 synthetic anomaly image

## 핵심 설계
- **Normal-specific / anomaly-shared decomposition**: few-shot setting에서는 카테고리별 정상성은 적고, 이상 패턴은 부분적으로 공유된다는 가정을 활용한다.
- **Generation for FSAD**: detector 자체를 바꾸기보다, anomaly 데이터 부족을 생성으로 보완한다.
- **Prompt simplification**: 복잡한 defect description engineering 없이 unified prompt로 이상 이미지를 생성한다.

## 성능 및 의미
- MVTec AD, VisA에서 few-shot anomaly image generation 품질과 downstream AD 성능 향상을 보고
- 핵심 기여는 단순 augmentation이 아니라, **few-shot에서 anomaly prior를 공유 가능한 요소와 정상 고유 요소로 분리해 generation을 안정화**했다는 점

## 핵심 인사이트
- Few-shot anomaly generation의 병목은 데이터 수보다도 **무엇을 공유하고 무엇을 분리해 학습할지**에 있다.
- 생성 모델은 FSAD에서 직접 detector를 대체하기보다, support 부족을 보완하는 data engine으로 더 적합할 수 있다.

## 관련 노트
- [One-to-Normal](./one_to_normal.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
