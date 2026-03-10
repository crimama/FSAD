# 나머지 논문 요약 노트

> 개별 상세 노트를 별도로 생성하지 않은 논문들의 요약

---

## Meta-Learning 계열

### Metaformer [ICCV 2021]
- Transformer + episode-based meta-learning for AD
- Pseudo-anomaly 생성으로 meta-training 수행
- 카테고리-agnostic 이상 탐지 능력 학습
- **카테고리**: Meta-learning

### HTD [ICCV 2021]
- Hierarchical transformation discrimination + generative model
- Coarse-to-fine transformation 분류로 정상 representation 학습
- 이상 이미지는 transformation discrimination이 잘 안 됨을 이용
- Loss: L_cls + λ·L_recon
- **카테고리**: Meta-learning / Generation hybrid

### FewSOME [CVPRW 2023]
- Siamese network로 support-query similarity 학습
- Contrastive/triplet loss 기반 metric learning
- Pseudo-anomaly와 정상의 쌍으로 학습
- **한계**: Pixel-level localization 약함
- **카테고리**: Metric learning

### MetaUAS [NeurIPS 2024]
- One-prompt meta-learning for universal anomaly segmentation
- 1개 정상 이미지로 cross-domain anomaly segmentation
- Industrial, medical 등 다양한 도메인 일반화
- **카테고리**: Meta-learning / Universal

---

## Normalizing Flow / Statistical

### DifferNet [WACV 2021]
- Pretrained CNN + RealNVP normalizing flow
- Negative log-likelihood를 anomaly score로 직접 사용
- Loss: L = -log p(z) - log|det(dz/dx)|
- MVTec full-shot: Image AUROC ~95%
- 이후 CFLOW-AD, FastFlow 등의 시초
- **카테고리**: Normalizing Flow

---

## Backbone Adaptation

### COFT-AD [TIP 2024]
- Contrastive fine-tuning으로 pretrained backbone을 AD에 adapt
- Normal feature를 더 compact하게 clustering
- Few-shot에서 overfitting 방지 전략 (LR scheduling, early stopping)
- Plug-in 방식: backbone만 교체하면 기존 AD 방법과 결합 가능
- MVTec 4-shot: Image AUROC ~93-95%
- **인사이트**: Pretrained feature가 AD에 최적이 아니다 — fine-tuning으로 개선 가능
- **카테고리**: Backbone Fine-tuning

---

## VLM/Prompt 기반

### AnomalyGPT [AAAI 2024]
- LVLM (ImageBind + Vicuna) + image decoder로 anomaly detection + NL explanation
- 시뮬레이션된 anomaly (DTD 텍스처 합성)로 학습, in-context few-shot 지원
- MVTec 1-shot: Image AUROC ~86-94% (세팅에 따라 다름)
- **인사이트**: Detection과 explanation을 통합. AD를 VQA 문제로 재정의.

### AnoPLe [2024]
- CLIP 기반 bi-directional prompt learning (text + visual 동시 학습)
- Normal만으로 학습, prompt alignment loss
- Text→visual, visual→text 양방향 alignment 강화
- MVTec: Image AUROC ~92-94% (few-shot)

### FADE [BMVC 2024]
- GPT-4V 등 대형 VLM 활용 zero-shot + few-shot 통합
- Multi-round prompting 전략, domain-specific prompt template
- Training-free 접근. Pixel-level은 VLM spatial reasoning 한계로 제한적.
- MVTec: ~90-94% (few-shot)

### SOWA [2024]
- CLIP ViT의 global self-attention → hierarchical frozen window self-attention으로 변경
- Global attention이 local anomaly detail을 놓치는 문제 해결
- Frozen backbone + attention 패턴만 변경 → parameter-efficient
- Multi-scale local + global 정보 활용
- **인사이트**: CLIP의 global attention은 AD에 오히려 해로움 — local detail이 희석됨

### CLIP-FSAC++ [2024]
- Learnable anomaly descriptor in CLIP text space
- Anomaly 유형별 text description ("scratched", "broken" 등) 활용
- Few-shot normal prototype + anomaly descriptor 결합
- Image-level 분류 위주

---

## Reconstruction / Generation

### FastRecon [ICCV 2023]
- Feature-level fast reconstruction from few-shot support
- Training-free 또는 minimal optimization
- Feature space reconstruction이 image space보다 few-shot에 robust
- **카테고리**: Feature Reconstruction

### Text-Guided VAE [CVPR 2024]
- Text guidance + variational generation for anomaly detection/segmentation
- Language-guided anomaly generation/reconstruction
- VAE loss + CLIP alignment loss
- **카테고리**: Multimodal Generation

### POUTA [2023]
- Few-shot AD의 support 부족을 보완하기 위해 정상 정보와 anomaly prior를 함께 활용하는 dual-utilization 계열 접근
- detector 구조 자체보다는 support usage/augmentation 설계에 초점
- **카테고리**: Generation / Augmentation 계열

---

## 특수 접근

### Crossmodal Feature Mapping [CVPR 2024]
- RGB + 3D cross-modal feature mapping 학습
- 정상에서의 cross-modal consistency가 anomaly에서 깨짐
- MVTec 3D-AD 중심
- **카테고리**: Multimodal (RGB+3D)

### Learning Multi-class 1-shot [ECCV 2024]
- 1개 정상 이미지로 multi-class AD
- VLM 기반 visual prompt 활용
- **카테고리**: 1-shot Multi-class

### Optimizing PatchCore [2023]
- PatchCore 하이퍼파라미터를 few-shot/many-shot 양쪽에서 체계적 최적화
- Few-shot에서는 coreset subsampling OFF, layer 선택이 texture/object에 따라 달라야 함
- Feature preprocessing (centering, normalization) 중요
- **인사이트**: Same algorithm, different optimal settings for different data regimes

---

## ⚠️ 내용 미확인 논문 (타이틀/venue만 기록)

현재 이 노트에서 별도로 관리하던 미확인 논문은 모두 원문 또는 공식 페이지 기준으로 개별 노트화했다.
추가 후보가 생기면 1차 출처 확인 후 다시 이 섹션에 기록한다.


---

## 관련 노트
- [종합 서베이](../2026-03-10_comprehensive_survey.md)
- [_lessons.md](../_lessons.md)
- [CARL](./carl.md)
- [One-for-All](./one_for_all.md)
- [UniVAD](./univad.md)
- [SeaS](./seas.md)
- [KAG-prompt](./kag_prompt.md)
- [Search is All You Need](./search_is_all_you_need.md)
- [FOCT](./foct.md)
- [Few-shot Online AD](./few_shot_online_ad.md)
- [POUTA](./pouta.md)
- [RFS Energy](./rfs_energy.md)
- [FineGrainedAD](./fine_grained_vl.md)
- [UniADC](./uniadc.md)
- [DFD](./dfd.md)
- [SOFS](./sofs.md)
- [MetaCAN](./metacan.md)
- [CIF](./commonality_in_few.md)
- [FoundAD](./foundation_visual_encoders.md)
- [D24FAD](./dual_distillation.md)
