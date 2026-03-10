# SOFS — Small Object Few-shot Segmentation for Vision-based Industrial Inspection

> 2024 | Few-shot Segmentation for Inspection | Support-mask based anomaly localization

## 핵심 방법론
산업 결함은 크기가 작아 기존 few-shot segmentation이 semantic distortion과 background false positive를 크게 겪는다는 문제를 다룬다. SOFS는 이미지 resizing을 피하는 non-resizing procedure, support annotation의 prototype intensity downsampling, abnormal prior map, mixed normal Dice loss를 결합해 small defect를 few-shot으로 분할한다.

## Architecture
- **Non-resizing pipeline**: 작은 defect semantic이 망가지지 않도록 원본 해상도 보존
- **Prototype intensity downsampling**: support mask의 target intensity를 더 정확히 반영
- **Abnormal prior map**: false positive 억제
- **Mixed normal Dice loss**: background 과검출 감소

## 핵심 설계
- **Small-object emphasis**: 자연영상 few-shot segmentation 가정을 산업 defect에 그대로 쓰지 않는다.
- **Mask-conditioned setting**: support mask가 주어질 때 unseen defect를 locate한다.
- **Bridge between FSS and AD**: 원문은 support mask에 따라 few-shot semantic segmentation과 few-shot anomaly detection 모두 가능하다고 설명한다.

## 성능
- 원문은 다양한 실험에서 superior performance를 보고
- 다만 이 방법은 일반적인 normal-only FSAD보다 **support mask supervision이 추가된 설정**에 가깝다

## 핵심 인사이트
- 산업 inspection에서는 anomaly detection과 few-shot segmentation의 경계가 완전히 분리되지 않는다.
- 작은 defect 문제는 backbone보다도 **전처리와 supervision granularity**가 더 직접적인 병목일 수 있다.

## 관련 노트
- [나머지 논문 요약](./remaining_papers.md)
- [FineGrainedAD](./fine_grained_vl.md)
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
