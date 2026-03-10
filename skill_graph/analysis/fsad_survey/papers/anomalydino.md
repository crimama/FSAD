# AnomalyDINO — Boosting Patch-based Few-shot AD with DINOv2

> 2024 | Foundation Model (DINOv2) | Training-free FSAD

## 핵심 방법론
DINOv2 ViT의 patch token을 few-shot normal reference와 비교. Multi-scale patch feature aggregation + attention-guided matching.

## Architecture
- **Backbone**: DINOv2 ViT (frozen)
- **Feature**: Patch token embeddings (multi-scale)
- **Matching**: Attention-guided few-shot reference matching
- **Training**: Training-free 또는 minimal training

## 성능
- MVTec AD: Image AUROC ~95-97% (k=4), Pixel AUROC ~97%+
- VisA에서도 competitive

## 핵심 인사이트
- DINOv2의 self-supervised feature가 ImageNet-supervised보다 AD에 우월
- Self-supervised pretraining이 "normality"의 일반적 표현 학습에 유리
- Patch-level semantic correspondence가 supervised feature보다 robust
- Training-free로도 매우 높은 성능 달성 가능

## 관련 노트
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
