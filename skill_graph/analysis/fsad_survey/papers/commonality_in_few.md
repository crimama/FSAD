# CIF — Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory

> 2025 | Multimodal FSAD | Hypergraph-enhanced memory

## 핵심 방법론
few-shot multimodal industrial anomaly detection에서 support 수가 적어 test pattern을 충분히 덮지 못하는 문제를 structural commonality 추출로 해결한다. CIF는 hypergraph로 intra-class structural information을 추출해 memory bank를 만들고, training-free hypergraph message passing과 hyperedge-guided memory search로 test-query와 memory 간 격차를 줄인다.

## Architecture
- **Semantic-aware hypergraph construction**: training sample의 공통 구조 추출
- **Memory bank**: intra-class structural prior 저장
- **Training-free hypergraph message passing**: test feature 업데이트
- **Hyperedge-guided memory search**: structural information으로 memory search 보조

## 핵심 설계
- **Multimodal FSAD**: MVTec 3D-AD, Eyecandies처럼 RGB와 추가 구조 정보가 있는 setting을 겨냥한다.
- **Higher-order correlation modeling**: pairwise graph 대신 hypergraph로 공통 구조를 뽑는다.
- **Training-free update**: test-time feature를 구조 priors에 맞게 보정한다.

## 성능
- MVTec 3D-AD와 Eyecandies few-shot setting에서 SOTA를 상회했다고 보고
- 현재 확인 가능한 공개 원문 범위에서는 정확한 수치보다 구조 공통성 기반 memory 설계가 핵심 기여다

## 핵심 인사이트
- multimodal FSAD에서는 modality fusion 자체보다 **few support에서 무엇이 공통 구조인지 추출하는가**가 더 중요할 수 있다.
- hypergraph memory는 pairwise retrieval보다 richer structural prior를 제공한다.

## 관련 노트
- [FOCT](./foct.md)
- [UniVAD](./univad.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
