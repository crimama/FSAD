# GraphCore — Pushing the Limits of Few-Shot AD in Industry Vision

> ICLR 2023 | Memory-bank + GNN | Few-shot PatchCore 확장

## 핵심 방법론
PatchCore의 few-shot 확장. GNN으로 patch feature 간 관계를 모델링하여 contextual feature를 학습, few-shot memory bank의 coverage 부족을 보완.

## Architecture
- **Backbone**: WideResNet50-2 (frozen)
- **GNN**: GCN/GAT 계열, patch feature를 노드로 하는 graph 구축
- **Graph construction**: Spatial proximity + feature similarity
- **Message passing**: 노드 간 contextual information propagation
- **Scoring**: GNN-enhanced feature의 k-NN distance

## Pipeline
```
Few-shot images → Backbone → Patch features → Graph construction → GNN → Enhanced features → Memory bank → k-NN scoring
```

## 성능
- MVTec AD few-shot: PatchCore, RegAD 상회
- 특히 1-shot, 2-shot 극한 few-shot에서 개선 폭 큼

## 핵심 인사이트
- "Few-shot에서는 individual patch가 아닌 patch 간 관계가 핵심"
- GNN이 contextual information을 propagation → 단독 patch feature의 한계 보완
- PatchCore의 단순 memory-bank 방식의 few-shot 한계를 graph-based reasoning으로 극복

## 관련 노트
- [PatchCore](./patchcore.md) — 원본 baseline
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
