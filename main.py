import os
from argparse import ArgumentParser

from omegaconf import OmegaConf

from run_anomalydino import parse_args as parse_baseline_args
from run_anomalydino import run as run_fewshot


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def load_config(path, overrides):
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def build_fewshot_args(cfg):
    run_cfg = cfg.RUN
    shots = run_cfg.shots if run_cfg.shots else [cfg.DATASET.shot]
    data_root = os.path.abspath(cfg.DATASET.data_root)
    argv = [
        "--dataset", cfg.DATASET.name,
        "--data_root", data_root,
        "--model_name", cfg.MODEL.model_name,
        "--resolution", str(cfg.MODEL.resolution),
        "--preprocess", run_cfg.preprocess,
        "--knn_metric", run_cfg.knn_metric,
        "--k_neighbors", str(run_cfg.k_neighbors),
        "--num_seeds", str(run_cfg.num_seeds),
        "--device", str(cfg.SYSTEM.device),
        "--warmup_iters", str(run_cfg.warmup_iters),
    ] + ["--shots", *[str(shot) for shot in shots]]
    if cfg.DATASET.category is not None:
        argv.extend(["--category", str(cfg.DATASET.category)])
    return argv


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    data_root = os.path.abspath(cfg.DATASET.data_root)

    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Configured dataset root does not exist: {data_root}. "
            "Set DATASET.data_root to a valid MVTec AD directory."
        )
    cfg.DATASET.data_root = data_root

    argv = build_fewshot_args(cfg)
    if cfg.RUN.faiss_on_cpu:
        argv.append("--faiss_on_cpu")
    if cfg.RUN.mask_ref_images:
        argv.append("--mask_ref_images")
    if not cfg.RUN.save_examples:
        argv.append("--no-save_examples")
    if not cfg.RUN.eval_clf:
        argv.append("--no-eval_clf")
    if cfg.RUN.eval_segm:
        argv.append("--eval_segm")
    if cfg.RUN.just_seed is not None:
        argv.extend(["--just_seed", str(cfg.RUN.just_seed)])
    if cfg.RUN.tag:
        argv.extend(["--tag", cfg.RUN.tag])

    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    run_fewshot(parse_baseline_args(argv))


if __name__ == "__main__":
    main()
