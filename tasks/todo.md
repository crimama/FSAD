# TODO

## 현재 작업
로컬 MVTec AD 데이터셋(`../MVTecAD`) 기준으로 단일 카테고리 실행 검증이 가능하도록 카테고리 필터를 메인 파이프라인에 연결

## 계획
- [x] 현재 엔트리포인트와 데이터 경로 전달 구조 확인
- [x] 설정/CLI에 카테고리 필터 추가
- [x] 선택된 카테고리만 처리하도록 실행 경로 제한
- [x] README와 기본 설정에 검증용 사용 예시 반영
- [x] 단일 카테고리 기준 최소 실행 검증 후 결과 기록

## 결과
- 완료:
  - `configs/default.yaml`에 `DATASET.category: null`을 추가해 단일 카테고리 제한을 설정 파일에서 제어 가능하게 함
  - `main.py`가 `DATASET.category`를 `run_anomalydino.py`의 `--category` 인자로 전달하도록 연결
  - `run_anomalydino.py`, `run_anomalydino_batched.py`에 `--category` 옵션과 카테고리 유효성 검증을 추가
  - `src/utils.py`에 공통 카테고리 필터 유틸리티를 추가해 선택된 카테고리만 처리하도록 제한
  - `README.md`에 단일 카테고리 검증용 실행 예시를 추가
- 검증:
  - `python -m compileall main.py run_anomalydino.py run_anomalydino_batched.py src` 통과
  - `python - <<'PY' ...` 스니펫으로 `DATASET.category=bottle` 설정 시 `--category bottle` 인자 전파, `/workspace/MVTecAD` 존재, `bottle/train/good` 존재를 확인
  - `timeout 30s python main.py --config configs/default.yaml DATASET.category=bottle RUN.warmup_iters=0 RUN.save_examples=false RUN.eval_clf=false RUN.eval_segm=false SYSTEM.device=cpu` 실행으로 실제 DINOv2 로딩 후 `bottle` 단일 카테고리 메모리뱅크 구축 및 테스트 샘플 처리 진행을 확인
- 미검증 범위:
  - 위 실제 실행은 30초 제한으로 종료되어 최종 메트릭 파일 생성 완료까지는 확인하지 못함
