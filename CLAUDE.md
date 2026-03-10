# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Principle (연구 철칙)

**단순한 성능 향상이 목표가 아니다.** 연구로서 가치가 있으며 동시에 성능을 향상시킬 수 있는 **novelty 있는 방법론을 제안**하는 것이 최종 목표다. 모든 실험과 방법론 설계는 "왜 이 방법이 학술적으로 의미가 있는가?"를 먼저 답할 수 있어야 한다. Trick 나열이 아닌, 명확한 insight와 원리에 기반한 방법론을 추구한다.

추가로 모든 연구 작업은 아래 4가지를 동시에 만족하도록 설계한다.
- **Claim**: 무엇을 주장하는가
- **Evidence**: 그 주장을 어떤 실험/분석으로 지지하는가
- **Boundary**: 어디까지 성립하고 어디서 깨지는가
- **Positioning**: 기존 방법 대비 어떤 차별성이 있는가

## Project Summary

Few-Shot Anomaly Detection (FSAD) 연구 프로젝트. 소수의 정상 샘플(few-shot)만으로 이미지 내 이상(anomaly)을 탐지하는 방법론을 연구한다. 산업 결함 검출, 의료 영상 등 라벨링 비용이 높은 도메인에서 실용적 가치를 가진다.

### Current Phase

**연구 설계 단계** — 41편 논문 서베이 완료, 논문 아웃라인 작성 중. 구현 코드는 아직 없음.
- 서베이: `skill_graph/analysis/fsad_survey/2026-03-10_comprehensive_survey.md`
- Research gaps: `skill_graph/analysis/fsad_survey/2026-03-10_research_gaps.md`
- 논문 아웃라인: `skill_graph/analysis/fsad_survey/2026-03-10_paper_outline_support_memory.md`
- 개별 논문 노트 (26편): `skill_graph/analysis/fsad_survey/papers/`
- 스킬 그래프 인덱스: `skill_graph/index.md`

## Commands (구현 후 사용)

```bash
# 아래 커맨드는 구현 완료 후 사용 예정 — 현재는 해당 파일이 없음
# python main.py --config configs/default.yaml
# python main.py --config configs/default.yaml DATASET.shot=4 MODEL.backbone=resnet50
# python main.py EXPANSION.use=false
# bash scripts/run_experiments.sh
```

No test suite exists. Validation is done by running experiments and checking metrics.

## Architecture (Target Design)

구현 시 아래 구조를 따를 예정:

### Pipeline Flow

```
Training: Few-shot Normal Images → Pretrained Backbone → Feature Extraction → Distribution Modeling
Testing:  Test Image → Features → Anomaly Scoring (distance to normal distribution) → Pixel/Image-level Metrics
```

### Planned Modules

- `main.py` — 진입점, 실험 오케스트레이션
- `models/` — 모델 정의 (feature extractor, anomaly scorer 등)
- `utils/` — 유틸리티 (metrics, visualization 등)
- `configs/` — 실험 설정 (YAML)
- `data_provider/` — 데이터 로더 (MVTec, VisA 등)

### Score Convention

Higher score = anomaly (이상일수록 높은 점수)

## Data Layout

```
{data_path}/{dataset_name}/{category}/train/good/
{data_path}/{dataset_name}/{category}/test/{defect_type}/
{data_path}/{dataset_name}/{category}/ground_truth/{defect_type}/
```

## Dependencies

- PyTorch
- torchvision (pretrained backbone)
- scikit-learn
- opencv-python
- omegaconf / hydra (config management)

## Config Parameter Tags

`[TUNE]` = 자주 튜닝하는 하이퍼파라미터
`[ARCH]` = 아키텍처 선택 (덜 변경)
`[DEPRECATED]` = 호환성 유지용 (미사용)
`[ABLATION]` = 논문 ablation에서 독립 변수로 다룰 항목
`[REPRO]` = 재현성에 직접 영향 주는 항목 (seed, split, eval protocol 등)

---

## Agents

프로젝트에서 활용 가능한 전문 에이전트. `agents/` 디렉토리에 정의.

| 에이전트 | 모델 | 용도 | 활성화 시점 |
|---------|------|------|-----------|
| planner | opus | 실험/구현 계획 수립 | 3단계+ 작업, 방법론 설계 |
| code-reviewer | sonnet | 코드 품질/보안 리뷰 | 코드 변경 후 |

에이전트 호출: Subagent Strategy에 따라 서브에이전트로 실행하거나 참조 문서로 활용.

## Context Modes

세션 중 작업 모드 전환. `contexts/` 디렉토리의 모드 파일 참조.

| 모드 | 파일 | 포커스 |
|------|------|--------|
| dev | `contexts/dev.md` | 구현 집중 — 코드 먼저, 설명 후 |
| research | `contexts/research.md` | 탐색 집중 — 이해 먼저, 코드 후 |
| review | `contexts/review.md` | 리뷰 집중 — 품질, 보안, 유지보수성 |

활성화: "이 세션은 [모드] 모드로 진행합니다" 또는 해당 파일 참조 요청.

## Hooks

자동 실행되는 품질 관리 훅. `.claude/settings.local.json`에 설정.

| 훅 | 이벤트 | 동작 |
|----|--------|------|
| suggest-compact | PreToolUse (Edit/Write) | 도구 호출 50회+ 시 전략적 /compact 제안 |
| git-push-reminder | PreToolUse (Bash) | git push 전 리뷰 리마인더 |
| lessons-reminder | Stop | 세션 중 교훈 기록 리마인더 |

---

## Workflow Orchestration

### 1. Plan Node Default
- 3단계 이상이거나 아키텍처 결정이 필요한 작업은 **반드시 plan mode 먼저**
- 작업 중 예상치 못한 문제가 생기면 STOP → 즉시 재계획. 억지로 밀어붙이지 말 것
- 구현뿐 아니라 **검증 단계**에도 plan mode 활용
- 스펙을 앞에서 상세히 작성해 모호함을 최소화

### 2. Subagent Strategy
- 메인 컨텍스트 윈도우를 깨끗하게 유지하기 위해 **서브에이전트를 적극 활용**
- 리서치, 탐색, 병렬 분석은 서브에이전트에 오프로드
- 복잡한 문제일수록 서브에이전트를 통해 컴퓨팅을 더 투입
- 서브에이전트 하나에 한 가지 작업만 (focused execution)

### 3. Self-Improvement Loop
- **사용자의 수정/지적이 있을 때마다**: `tasks/lessons.md`에 해당 패턴을 기록
- 같은 실수를 반복하지 않도록 스스로 규칙을 작성
- 세션 시작 시 `tasks/lessons.md`를 먼저 확인하여 과거 교훈 리뷰
- 반복 검증된 패턴은 `skill_graph/analysis/{주제}/_lessons.md`로 승격

### 4. Verification Before Done
- **작동을 증명하지 않은 채 완료 처리 금지**
- 변경 사항이 있을 때: main 동작과 diff하여 확인
- "시니어 엔지니어가 이 코드를 승인할 것인가?" 자문
- 테스트 실행, 로그 확인, 정확성 시연 후 완료
- 연구 작업은 추가로 "이 결과가 논문 figure/table/claim에 들어갈 수 있는가?"까지 확인

### 5. Demand Elegance (Balanced)
- 비자명한 변경에는 "더 우아한 방법이 있지 않은가?" 자문
- 수정이 hacky하게 느껴지면: "지금 알고 있는 모든 것을 감안해 우아한 해결책을 구현"
- 단순·명백한 수정에는 생략 — 과잉 설계 금지

### 6. Autonomous Bug Fixing
- 버그 리포트가 주어지면: **그냥 고친다**. 손을 잡아달라고 하지 말 것
- 로그, 에러, 실패 테스트를 직접 분석하여 해결
- 사용자의 컨텍스트 전환 없이 처리

---

## Research Decision Rules

### 1. Novelty Gate
- 새 아이디어 제안 전 반드시 아래를 한 줄씩 답한다.
  - 기존 방법의 핵심 한계는 무엇인가?
  - 제안 방법이 건드리는 메커니즘은 무엇인가?
  - 이 차이가 논문 contribution 한 줄로 요약 가능한가?
- 위 질문에 답하지 못하면 구현보다 **문제 재정의 또는 related work 조사**를 우선한다

### 2. Claim-Evidence Discipline
- claim 하나당 최소 하나의 직접 증거를 연결한다
- 성능 주장은 평균 성능만이 아니라 **분산, seed 민감도, category별 편차**를 함께 본다
- "개선됨"이라는 표현은 baseline, 비교군, metric, split이 명시될 때만 사용
- 증거 없는 직관 서술은 `가설`, 검증 완료된 서술은 `결론`으로 명확히 구분한다

### 3. Baseline and Ablation First
- 새 방법은 반드시 **강한 baseline**과 비교한다. 약한 baseline을 이긴 것만으로 기여를 주장하지 않는다
- 성능이 좋아도 ablation 없이 핵심 모듈 기여를 주장하지 않는다
- ablation은 가능한 한 one-factor change로 설계한다
- 부가 모듈이 많아질수록 "없어도 되는 것"을 제거하는 ablation을 우선한다

### 4. Reproducibility Minimum Bar
- 최소 기록 항목: commit hash, config diff, seed, 데이터 split, 체크포인트/로그 경로
- 재현 불가능한 결과는 좋은 결과가 아니라 **미완료 결과**로 취급한다
- 평가 스크립트와 학습 스크립트의 metric 정의가 다를 가능성을 항상 점검한다

### 5. Negative Results Are Assets
- 실패 실험도 버리지 않는다. 왜 실패했는지까지 기록해야 탐색 공간이 줄어든다
- "안 됨"이 아니라 "어떤 조건에서 왜 안 됨"으로 정리한다
- 반복 실패 패턴은 `_lessons.md`로 승격해 이후 탐색 비용을 줄인다

### 6. Paper-Oriented Prioritization
- 구현 난이도보다 **논문 메시지 밀도**가 높은 실험을 우선한다
- 결과가 좋아도 설명이 약하면 후순위, 결과가 약간 부족해도 설명력이 강하면 보존한다
- 최종적으로는 contribution, figure, table, related work 문단으로 재사용 가능한 형태를 선호한다

---

## Task Management

1. **Plan First**: 구현 시작 전 `tasks/todo.md`에 체크리스트 형태로 계획 작성
2. **Verify Plan**: 구현 착수 전 계획 확인
3. **Track Progress**: 진행하면서 완료 항목에 체크
4. **Explain Changes**: 각 단계마다 고수준 요약 제공
5. **Document Results**: 완료 후 `tasks/todo.md`에 결과 섹션 추가
6. **Capture Lessons**: 수정/지적 발생 시 즉시 `tasks/lessons.md` 업데이트

```
tasks/
├── todo.md        # 현재 세션 계획·진행·결과 (세션마다 갱신)
└── lessons.md     # 수정·지적으로부터 추출한 누적 교훈 (영속적)
```

---

## Experiment Process

실험은 반드시 아래 **6단계 프로세스**를 따른다. 템플릿: `skill_graph/experiments/_TEMPLATE.md`

```
1. 문제 분석 (Problem Analysis)  → 현상 + 원인 추정 + 관련 선행 노트
2. 가설 설정 (Hypothesis)        → "X하면 Y 개선" + 근거 + 예상 수치
3. 실험 설정 (Experiment Design) → 대조군/실험군 + config diff + 실행 커맨드
4. 결과 (Results)                → 정량 결과표 + 로그 경로
5. 결과 분석 (Analysis)          → 가설 검증 (✅/❌/⚠️) + 상세 분석 + 부수 발견
6. 피드백 (Feedback)             → 교훈 + 다음 실험 제안 + _lessons.md 승격 여부
```

**실험 진행 규칙:**
- 실험 시작 전 1~3단계 먼저 작성 후 실행
- 실험 완료 후 4~6단계 기록
- 가설 검증 결과가 반복 활용 가능하면 `analysis/{주제}/_lessons.md`로 승격
- `## 관련 노트`로 실험 간 연쇄 추적
- 모든 실험은 가능한 경우 baseline 대비 **단일 주장 검증(single claim test)** 형태로 쪼갠다
- 메인 결과 표에 들어갈 후보 실험은 seed, confidence interval 또는 반복 측정 여부를 표시한다
- 성능 향상 실험과 별도로 실패 사례, category 편차, calibration/robustness를 보는 분석 실험을 분리한다

**실험 디렉토리 구조:**
```
skill_graph/experiments/YYYY-MM-DD_실험명/
├── report.md            # 6단계 프로세스 전체 기록
├── config_diff.yaml     # baseline 대비 변경된 config
└── logs/                # 실험 로그
```

## Update Notes

```
skill_graph/
├── experiments/
│   ├── _TEMPLATE.md              # 실험 보고서 템플릿 (6단계 프로세스)
│   └── YYYY-MM-DD_실험명/
├── papers/
│   ├── _TEMPLATE.md              # 관련 논문/선행연구 요약
│   └── YYYY-MM-DD_논문명.md
├── analysis/
│   └── 주제명/
│       ├── YYYY-MM-DD_설명.md
│       └── _lessons.md           # tasks/lessons.md에서 승격된 검증 패턴
├── bugfix/
│   └── YYYY-MM-DD_설명.md
└── ideas/
    └── YYYY-MM-DD_설명.md
```

**스킬 그래프:**
- 노트 간 `## 관련 노트`로 상대 경로 링크
- 실험 → 분석 → 아이디어 → 논문 포지셔닝 → 후속 실험 흐름 추적
- 반복되는 패턴이나 검증된 기법은 `analysis/{주제}/_lessons.md`로 승격

---

## Core Principles

- **Research Over Tricks**: 성능 숫자보다 연구 메시지와 메커니즘 설명을 우선한다
- **Strong Baselines Only**: 약한 비교군 위의 승리는 기여가 아니다
- **One Claim, One Test**: 한 실험이 여러 주장을 동시에 떠안지 않게 설계한다
- **Evidence Before Narrative**: 서사는 결과 뒤에 온다. 먼저 증거를 만든다
- **Reproducibility Is Part of Quality**: 재현성 없는 결과는 미완성으로 본다

- **Simplicity First**: 모든 변경은 가능한 한 단순하게. 최소한의 코드에만 영향을 줄 것
- **No Laziness**: 근본 원인을 찾아라. 임시방편 금지. 시니어 개발자 기준을 적용
- **Minimal Impact**: 변경은 필요한 것만. 불필요한 버그 유입 방지
- **모듈화 필수**: 새롭게 추가하는 모든 모듈/기능은 config에서 `enable: true/false`로 on/off 가능하게 구현. 과거 실험을 config만으로 정확히 재현할 수 있어야 한다.
- **Reproducibility**: seed 고정, config 기록, 환경 명시
- **Ablation-friendly**: 각 컴포넌트를 독립적으로 on/off 가능하게 설계

## Important Conventions

- Few-shot setting: k-shot은 카테고리당 k개의 정상 이미지만 사용
- 평가 지표: Image-level AUROC, Pixel-level AUROC, PRO (Per-Region Overlap)
- 데이터셋: MVTec AD, VisA가 주 벤치마크
- Backbone: ImageNet pretrained (WideResNet50, ResNet18 등)
