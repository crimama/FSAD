# AGENTS.md

This file provides guidance to Codex when working in this repository. It mirrors the existing Claude setup in `CLAUDE.md` and `.claude/skills/`.

## First Read

At the start of each meaningful task, review these files in order:

1. `CLAUDE.md`
2. `tasks/lessons.md`
3. `tasks/todo.md`

Treat `CLAUDE.md` as the primary project playbook unless a direct user instruction overrides it.

## Project Purpose

This repository is for Few-Shot Anomaly Detection (FSAD) research. The goal is not only better metrics, but research-worthy, novel methodology with clear insight and academic value.

Before proposing or implementing a method, answer:

- Why is this method academically meaningful?
- What principle or insight does it test beyond a one-off trick?

All research work must satisfy four criteria simultaneously:
- **Claim**: What are we asserting?
- **Evidence**: What experiment/analysis supports the claim?
- **Boundary**: Where does it hold, where does it break?
- **Positioning**: What differentiates it from existing methods?

## Core Working Rules

- Keep changes minimal, simple, and reproducible.
- Prefer root-cause fixes over patches.
- Any newly added module or feature must be switchable from config with an explicit `enable: true/false` style control where applicable.
- Preserve ablation-friendliness: components should be independently toggled when practical.
- Do not mark work complete without verification evidence.

## Research Decision Rules

### 1. Novelty Gate
Before proposing a new idea, answer:
- What is the core limitation of existing methods?
- What mechanism does the proposed method change?
- Can this difference be summarized as one contribution line?
If these cannot be answered, prioritize problem redefinition or related work survey over implementation.

### 2. Claim-Evidence Discipline
- One claim = at least one direct evidence.
- Performance claims must include variance, seed sensitivity, category-level deviation.
- "Improved" requires explicit baseline, comparison, metric, and split.
- Separate `hypothesis` (unverified intuition) from `conclusion` (verified).

### 3. Baseline and Ablation First
- Compare against strong baselines only.
- No module contribution claim without ablation.
- One-factor change ablation design preferred.

### 4. Reproducibility Minimum Bar
- Record: commit hash, config diff, seed, data split, checkpoint/log path.
- Unreproducible results = incomplete results.

### 5. Negative Results Are Assets
- Record why experiments failed, not just that they failed.
- Repeated failure patterns → promote to `_lessons.md`.

### 6. Paper-Oriented Prioritization
- Prioritize experiments with high paper message density over implementation difficulty.
- Prefer results with strong explanatory power.

## Required Workflow

### 1. Task Tracking

Before substantial implementation, update `tasks/todo.md` with:

- `## 현재 작업`
- `## 계획`
- progress checkmarks while working
- `## 결과` after completion

Do not overwrite unrelated existing content.

### 2. Lessons

When the user corrects your approach or a recurring mistake becomes clear, record it in `tasks/lessons.md` using:

```md
### [YYYY-MM-DD] 제목
발생 상황: ...
잘못한 것: ...
올바른 방법: ...
```

If a lesson becomes a repeated, validated pattern, promote it into `skill_graph/analysis/<topic>/_lessons.md`.

### 3. Verification

Do not finish on intent alone. Verify with the strongest practical signal available, such as:

- running the relevant command or experiment entrypoint
- checking logs or generated artifacts
- confirming config wiring and execution path

If full experiment execution is too heavy, state exactly what was and was not verified.
For research work, additionally verify: "Can this result go into a paper figure/table/claim?"

## Experiment Process

For experiment work, follow the 6-step process defined in `CLAUDE.md` and `skill_graph/experiments/_TEMPLATE.md`:

1. Problem Analysis
2. Hypothesis
3. Experiment Design
4. Results
5. Analysis
6. Feedback

Do not run an experiment before steps 1-3 are written. Hypotheses must include an expected quantitative effect.
Design experiments as single claim tests when possible.

## Agents and Contexts

### Agents (`agents/` directory)
| Agent | Model | Purpose |
|-------|-------|---------|
| planner | opus | Implementation/experiment planning |
| code-reviewer | sonnet | Code quality/security review |

### Context Modes (`contexts/` directory)
| Mode | File | Focus |
|------|------|-------|
| dev | `contexts/dev.md` | Implementation — code first |
| research | `contexts/research.md` | Exploration — understand first |
| review | `contexts/review.md` | Quality, security, maintainability |

## Codex Mapping For Existing Claude Skills

Codex cannot auto-register the local `.claude/skills/*` files as native skills, so use them as workflow references:

- `.claude/skills/todo/SKILL.md`: how to maintain `tasks/todo.md`
- `.claude/skills/lessons/SKILL.md`: how to record and promote lessons
- `.claude/skills/experiment/SKILL.md`: experiment note lifecycle
- `.claude/skills/update-note/SKILL.md`: create notes under `skill_graph/`
- `.claude/skills/analyze/SKILL.md`: analysis workflow reference
- `.claude/skills/link-notes/SKILL.md`: related-note linking workflow
- `.claude/skills/verify/SKILL.md`: build/type/lint/test verification
- `.claude/skills/checkpoint/SKILL.md`: git-based checkpoint management
- `.claude/skills/compact/SKILL.md`: strategic compaction guide
- `.claude/skills/learn/SKILL.md`: session learning pipeline

When a user request clearly matches one of these workflows, open the corresponding `SKILL.md` and follow it manually.

## Key Project Conventions

- Higher score means more anomalous.
- Few-shot means k normal images per category.
- Main metrics: Image-level AUROC, Pixel-level AUROC, PRO.
- Main benchmarks: MVTec AD, VisA.
- Typical backbone family: ImageNet-pretrained ResNet / WideResNet.

## Typical Entry Points

```bash
python main.py --config configs/default.yaml
python main.py --config configs/default.yaml DATASET.shot=4 MODEL.backbone=resnet50
python main.py EXPANSION.use=false
bash scripts/run_experiments.sh
```

Validation is experiment-driven; there is no dedicated test suite unless the repository later adds one.
