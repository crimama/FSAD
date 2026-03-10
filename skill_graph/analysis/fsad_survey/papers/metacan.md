# MetaCAN — Improving Generalizability of Few-shot Anomaly Detection with Meta-learning

> CIKM 2025 | Meta-learning + LVLM | Category-to-anomaly transformation

## 핵심 방법론
대형 비전-언어 모델의 category semantic 정보를 anomaly 정보로 바꾸는 **category-to-anomaly network**를 제안하고, 이를 few-shot AD에 맞춘 **AD meta-learning** scheme으로 학습한다. auxiliary dataset에서 여러 카테고리 기반 task를 구성해 cross-category, cross-domain generalizability를 높이는 것이 핵심이다.

## Architecture
- **Backbone**: CLIP visual encoder ViT-B/16+와 text encoder
- **Image-image anomaly discriminator**: multi-level visual feature로 anomaly 정보 추출
- **Image-text anomaly detector**: image-text anomaly map 생성
- **AD meta-learning**: task construction, parameter update를 few-shot AD용으로 재설계

## 핵심 설계
- **Category-to-anomaly transformation**: LVLM의 class semantics를 anomaly-sensitive representation으로 전환한다.
- **AD-specific meta-learning**: 분류용 meta-learning을 그대로 가져오지 않고 support/query 구성과 update를 AD에 맞게 조정한다.
- **Cross-domain generalization**: industrial, medical, semantic AD까지 함께 다룬다.

## 성능
- VisA를 auxiliary training dataset으로 썼을 때 MVTec AD에서 2/4/8-shot AUROC가 각각 **0.946 / 0.955 / 0.958**
- 같은 설정에서 AITEX AUROC는 **0.765 / 0.774 / 0.802**
- 저자들은 VisA, AITEX뿐 아니라 cross-domain 설정에서도 기존 SOTA 대비 일관된 향상을 보고한다

## 핵심 인사이트
- few-shot AD의 일반화 병목은 support 수보다도 **카테고리 semantic을 anomaly signal로 변환하는 능력**에 있을 수 있다.
- Meta-learning은 FSAD에서 단순 initialization보다 **task construction 자체**가 중요하다는 점을 보여준다.

## 관련 노트
- [InCTRL](./inctrl.md)
- [KAG-prompt](./kag_prompt.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
