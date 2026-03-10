# FSAD Baseline Setup

This repository now uses [AnomalyDINO](https://github.com/dammsi/AnomalyDINO) as the default baseline codebase for few-shot anomaly detection experiments.

## Baseline entrypoints

```bash
python main.py --config configs/default.yaml
python main.py --config configs/default.yaml DATASET.data_root=../MVTecAD
python main.py --config configs/default.yaml DATASET.category=bottle RUN.warmup_iters=0 RUN.save_examples=false RUN.eval_segm=false
python run_anomalydino.py --dataset MVTec --shots 1 --num_seeds 1 --preprocess agnostic --data_root ../MVTecAD --faiss_on_cpu
python run_anomalydino.py --dataset MVTec --category bottle --shots 1 --num_seeds 1 --preprocess agnostic --data_root ../MVTecAD --warmup_iters 0 --faiss_on_cpu
python run_anomalydino_batched.py --dataset MVTec --data_root ../MVTecAD --device cuda:0
```

## Notes

- Existing research notes under `skill_graph/` and task tracking under `tasks/` are preserved.
- `requirements.txt` is adjusted for this environment and uses `faiss-cpu` by default.
- If GPU FAISS is installed separately, the detection code will use it automatically unless `--faiss_on_cpu` is set.
- The default config now targets the local MVTec AD directory at `../MVTecAD` and resolves it to an absolute path at runtime.
- `DATASET.category` or `--category` can be used to restrict execution to one MVTec category for faster verification.
