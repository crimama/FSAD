# FSAD Survey Lessons

> 확인된 논문에서만 추출한 패턴 및 교훈

---

## Pretrained Feature 관련

1. **Mid-level feature가 AD에 최적**: Layer 2-3 (ResNet 기준)이 texture + semantic 정보 모두 포함 (PatchCore, CVPR 2022)
2. **DINOv2 > ImageNet-supervised**: Self-supervised pretraining이 "normality"의 일반적 표현 학습에 유리 (AnomalyDINO)
3. **Feature space > Image space**: Reconstruction, matching 모두 feature space에서 수행하는 것이 few-shot에 robust (FastRecon, ICCV 2023)

## Few-shot 특화 교훈

4. **Few-shot에서는 coreset subsampling 불필요**: 이미 적은 데이터를 더 줄이면 해로움 (Opt. PatchCore)
5. **Feature 간 관계 모델링이 핵심**: 개별 feature 비교보다 GNN 등으로 context propagation (GraphCore, ICLR 2023)
6. **Same algorithm, different settings**: Few-shot과 full-shot은 같은 알고리즘이라도 최적 설정이 다름 (Opt. PatchCore)
7. **Contrastive fine-tuning이 effective**: Few-shot으로도 backbone fine-tuning 가능 (COFT-AD, TIP 2024)

## VLM/Prompt Learning 관련

8. **Normal-only prompt learning 가능**: Anomaly prompt는 normal의 complement로 학습 가능 (PromptAD, CVPR 2024)
9. **CLIP의 granularity gap**: Image-level alignment으로 학습된 CLIP은 pixel-level AD에 부적합 (SOWA)
10. **CLIP global attention의 한계**: Global self-attention이 local anomaly detail을 놓침 → window attention으로 개선 (SOWA)

## Architecture/Design 관련

11. **Registration = natural few-shot framework**: Support를 query에 정렬하고 차이를 보는 것이 직관적 (RegAD, ECCV 2022)
12. **In-context learning for AD**: Few-shot normals를 LLM의 in-context examples처럼 활용 (InCTRL, CVPR 2024)
13. **AD를 VQA로 재정의**: LVLM으로 detection + NL explanation 통합 (AnomalyGPT, AAAI 2024)

## 평가 관련

14. **논문마다 실험 세팅 차이 큼**: Shot 수, support 선택 방법, seed 등이 다르므로 직접 비교에 주의
15. **Texture vs Object 카테고리 구분 필요**: 방법론에 따라 한쪽에서만 잘 작동하는 경우 많음

---

## 관련 노트
- [종합 서베이](./2026-03-10_comprehensive_survey.md)
