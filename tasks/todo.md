# TODO

## 현재 작업
MVTec AD baseline 실험 실행 + 로깅 시스템 구축

## 계획
- [x] 환경 확인 (GPU, 데이터셋, 의존성)
- [x] 전체 카테고리 1-shot baseline 실행 (eval_clf + eval_segm)
- [x] 로깅 시스템 구현 (`src/logging_utils.py`)
- [x] `main.py`에 로깅 통합
- [x] 로깅 시스템 검증
- [ ] (선택) `run_anomalydino.py`의 print()를 logging으로 점진적 전환

## 결과

### 로깅 시스템
- `src/logging_utils.py` 신규 생성 — dual logging (console + file), metadata, summary
- `main.py` 수정 — 실험 시작 시 자동으로 `logs/{timestamp}/` 디렉토리 생성
- 각 실험 실행 시 저장되는 파일:
  - `run.log` — 전체 콘솔 출력 (print + logging 모두 캡처)
  - `config.yaml` — resolved OmegaConf config
  - `metadata.json` — git commit, branch, timestamp, python version, command
  - `metrics_seed={N}.json` — 실험 완료 후 결과 복사 + 요약 테이블 출력

### Baseline 결과 (1-shot, dinov2_vits14_448, agnostic)
| Metric | Mean |
|--------|------|
| Image AUROC | 0.9702 |
| Image AP | 0.9849 |
| Image F1 | 0.9621 |
| Pixel AUROC | 0.9646 |
| AUPRO | 0.9181 |
| Pixel F1 | 0.5886 |
