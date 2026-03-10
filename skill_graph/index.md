# Skill Graph Index

> 프로젝트 변경 기록의 키워드 기반 인덱스.
> 키워드 → 문서 링크로 그래프 탐색 가능.
> 새 문서 추가 시 이 파일도 함께 갱신할 것.

---

## 키워드 그래프

```
                    ┌──────────────┐
                    │   FSAD 서베이  │
                    └──────┬───────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌────────────┐ ┌──────────────┐
    │ Memory-bank │ │ VLM/Prompt │ │ Foundation    │
    │ (PatchCore) │ │ (PromptAD) │ │ Model (DINO) │
    └──────┬──────┘ └─────┬──────┘ └──────┬───────┘
           │              │               │
    ┌──────▼──────┐ ┌─────▼──────┐ ┌──────▼───────┐
    │ GraphCore   │ │ InCTRL     │ │ AnomalyDINO  │
    │ (GNN확장)   │ │ One-for-All│ │ UniVAD       │
    └─────────────┘ └────────────┘ └──────────────┘
           │
    ┌──────▼──────┐     ┌────────────┐
    │ Registration│     │ Generation │
    │ (RegAD)     │     │ (SeaS)     │
    └─────────────┘     └────────────┘
```

---

## 키워드 → 문서 매핑

| 키워드 | 문서 | 카테고리 |
|--------|------|----------|
| **FSAD-survey** | [FSAD 종합 서베이](analysis/fsad_survey/2026-03-10_comprehensive_survey.md) | analysis |
| **FSAD-gaps** | [FSAD Research Gaps](analysis/fsad_survey/2026-03-10_research_gaps.md) | analysis |
| **FSAD-outline** | [FSAD Paper Outline](analysis/fsad_survey/2026-03-10_paper_outline_support_memory.md) | analysis |
| **memory-bank** | [PatchCore](analysis/fsad_survey/papers/patchcore.md), [GraphCore](analysis/fsad_survey/papers/graphcore.md) | analysis |
| **registration** | [RegAD](analysis/fsad_survey/papers/regad.md) | analysis |
| **VLM-prompt** | [PromptAD](analysis/fsad_survey/papers/promptad.md), [InCTRL](analysis/fsad_survey/papers/inctrl.md) | analysis |
| **foundation-model** | [AnomalyDINO](analysis/fsad_survey/papers/anomalydino.md) | analysis |
| **generation** | [SeaS](analysis/fsad_survey/papers/seas.md), [One-to-Normal](analysis/fsad_survey/papers/one_to_normal.md) | analysis |
| **meta-learning** | [나머지 논문](analysis/fsad_survey/papers/remaining_papers.md) | analysis |
| **lessons** | [FSAD Lessons](analysis/fsad_survey/_lessons.md) | analysis |

---

## 문서 → 키워드 역매핑

### analysis/

| 문서 | 상태 | 키워드 |
|------|------|--------|
| [FSAD 종합 서베이](analysis/fsad_survey/2026-03-10_comprehensive_survey.md) | 🟢 | `FSAD-survey` `few-shot` `anomaly-detection` |
| [FSAD Research Gaps](analysis/fsad_survey/2026-03-10_research_gaps.md) | 🟢 | `FSAD-gaps` `research-direction` `methodology` |
| [FSAD Paper Outline](analysis/fsad_survey/2026-03-10_paper_outline_support_memory.md) | 🟢 | `FSAD-outline` `paper-outline` `support-memory` |
| [PatchCore](analysis/fsad_survey/papers/patchcore.md) | 🟢 | `memory-bank` `k-NN` `coreset` |
| [GraphCore](analysis/fsad_survey/papers/graphcore.md) | 🟢 | `memory-bank` `GNN` `few-shot` |
| [RegAD](analysis/fsad_survey/papers/regad.md) | 🟢 | `registration` `STN` `cross-category` |
| [PromptAD](analysis/fsad_survey/papers/promptad.md) | 🟢 | `VLM-prompt` `CLIP` `normal-only` |
| [InCTRL](analysis/fsad_survey/papers/inctrl.md) | 🟢 | `VLM-prompt` `in-context` `generalist` |
| [AnomalyDINO](analysis/fsad_survey/papers/anomalydino.md) | 🟢 | `foundation-model` `DINOv2` `training-free` |
| [SeaS](analysis/fsad_survey/papers/seas.md) | 🟢 | `generation` `diffusion` `disentangle` |
| [One-to-Normal](analysis/fsad_survey/papers/one_to_normal.md) | 🟢 | `generation` `reconstruction` `personalization` |
| [나머지 논문](analysis/fsad_survey/papers/remaining_papers.md) | 🟢 | `meta-learning` `normalizing-flow` `frequency` |
| [FSAD Lessons](analysis/fsad_survey/_lessons.md) | 🟢 | `lessons` `patterns` |

### experiments/

| 문서 | 상태 | 키워드 |
|------|------|--------|
_(아직 없음)_

### bugfix/
_(아직 없음)_

---

## 문서 간 연결 (관련 노트)

```
FSAD 종합 서베이
        │
        ├──▶ PatchCore ──▶ GraphCore (GNN 확장)
        │         │
        │         └──▶ Opt. PatchCore (하이퍼파라미터 최적화)
        │
        ├──▶ RegAD ──▶ CARL (Category-agnostic 확장)
        │
        ├──▶ PromptAD ──┬──▶ AnoPLe (Bi-directional)
        │               └──▶ One-for-All (Instance-induced)
        │
        ├──▶ InCTRL (Generalist in-context learning)
        │
        ├──▶ AnomalyDINO ──▶ Foundation Encoders (ICLR 2026)
        │         │
        │         └──▶ UniVAD (Training-free unified)
        │
        ├──▶ SeaS ──▶ One-to-Normal (Reconstruction approach)
        │
        └──▶ _lessons.md (검증된 패턴)
```

---

## 타임라인

| 날짜 | 문서 | 요약 |
|------|------|------|
| 2026-03-10 | [FSAD 종합 서베이](analysis/fsad_survey/2026-03-10_comprehensive_survey.md) | 41편 FSAD 논문 서베이 — 방법론 분류, 트렌드, 인사이트 |
| 2026-03-10 | [FSAD Research Gaps](analysis/fsad_survey/2026-03-10_research_gaps.md) | 서베이 기반 연구 갭 정리 — 유망 방향과 피해야 할 방향 |
| 2026-03-10 | [FSAD Paper Outline](analysis/fsad_survey/2026-03-10_paper_outline_support_memory.md) | support utilization 병목 가설 기반 논문 아웃라인 정리 |
