# PatchCore — Towards Total Recall in Industrial Anomaly Detection

> CVPR 2022 | Memory-bank 기반 | Full-shot SOTA, Few-shot 핵심 baseline

## 핵심 방법론
Pretrained backbone에서 patch-level feature를 추출하여 memory bank에 저장, test 시 k-NN distance로 anomaly scoring.

## Architecture
- **Backbone**: WideResNet50-2 (ImageNet pretrained, frozen)
- **Feature layers**: Layer 2, 3 (mid-level)
- **Local neighbourhood aggregation**: 주변 patch feature 통합으로 receptive field 확대
- **Coreset subsampling**: Greedy coreset selection으로 memory bank 1-10%로 축소
- **Scoring**: k-NN distance (k=1)

## 핵심 수식
```
s(x_test) = max_i { min_j ||φ(x_test)_i - m_j||₂ }
```
- φ: feature extractor, m_j: memory bank의 j번째 patch feature

## 성능
- MVTec AD (full-shot): Image AUROC **99.1%**, Pixel AUROC **98.1%**
- Few-shot에서도 적용 가능하나 coverage 부족이 한계

## 핵심 인사이트
- Pretrained feature를 저장하고 비교하는 것만으로 SOTA
- Mid-level feature가 texture + semantic 정보 모두 포함
- Coreset subsampling이 실용성을 크게 높임
- **Few-shot 한계**: memory bank coverage 부족 → GraphCore가 GNN으로 보완

## Few-shot 설정 시 주의사항 (Opt. PatchCore)
- Coreset subsampling OFF (이미 적은 데이터)
- Layer 선택이 texture/object에 따라 달라야 함
- Feature preprocessing (centering, normalization) 중요

## 관련 노트
- [GraphCore](./graphcore.md) — GNN 확장
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
