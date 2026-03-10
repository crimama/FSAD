# UniVAD — A Training-free Unified Model for Few-shot Visual Anomaly Detection

> CVPR 2025 | Training-free Foundation Model | Unified few-shot VAD

## 핵심 방법론
추가 학습 없이 다양한 도메인에서 few-shot anomaly detection을 수행하는 unified 모델. 핵심은 query와 support 사이의 문맥 보존 대응점을 찾는 **Contextual Component Clustering (C3)** 이며, local-global correspondence를 함께 활용한다.

## Architecture
- **Backbone**: frozen foundation model feature extractor
- **C3**: query/support feature를 문맥 단위로 묶어 correspondence 형성
- **CAPM**: local contextual anomaly perception 강화
- **GECM**: global semantic correspondence 보정
- **Inference**: training-free matching 기반 anomaly scoring

## 핵심 설계
- **Training-free**: 새로운 도메인이나 카테고리에서도 파라미터 업데이트가 없다.
- **Unified**: 산업, 의료, logical anomaly를 하나의 추론 프레임워크로 처리한다.
- **Context-aware matching**: 패치 단위 독립 비교보다, 문맥을 보존한 component 단위 비교가 더 안정적이다.

## 성능
- 1-shot MVTec-AD: Image AUROC **97.8**, Pixel AUROC **96.5**
- 1-shot VisA: Image AUROC **93.5**, Pixel AUROC **98.2**
- 의료/논리 anomaly 데이터셋에서도 기존 zero-shot, few-shot 방법 대비 일관된 우세를 보고

## 핵심 인사이트
- Few-shot training 자체를 줄이는 것이 아니라, **추론 시 correspondence 품질을 높이는 것**만으로도 강한 성능이 가능하다.
- Foundation model feature의 성능 한계는 feature 자체보다 **matching granularity와 context preservation**에 더 가깝다.
- 산업 AD를 넘어서 cross-domain unified anomaly detection으로 확장되는 흐름의 대표 사례다.

## 관련 노트
- [AnomalyDINO](./anomalydino.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
