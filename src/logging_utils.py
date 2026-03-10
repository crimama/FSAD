"""Experiment logging utilities.

Sets up dual logging (console + file), captures experiment metadata,
and writes a post-run summary.
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def get_git_info():
    """Return dict with current git commit hash and dirty status."""
    info = {"commit": None, "dirty": None, "branch": None}
    try:
        info["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["dirty"] = len(status) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def make_log_dir(base_dir="logs", tag=None):
    """Create and return a timestamped log directory.

    Structure: logs/{YYYY-MM-DD_HHMMSS}[_{tag}]/
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = f"{ts}_{tag}" if tag else ts
    log_dir = os.path.join(base_dir, name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_logging(log_dir, level=logging.INFO):
    """Configure root logger to write to both console and log_dir/run.log.

    Returns the logger instance.
    """
    log_file = os.path.join(log_dir, "run.log")

    # Clear any existing handlers on root logger
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler — captures everything
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler — same level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Also redirect print() output to the log file via a tee-like wrapper
    sys.stdout = _TeeStream(sys.stdout, fh.stream)

    return root


class _TeeStream:
    """Write to two streams simultaneously (stdout + log file)."""

    def __init__(self, stream1, stream2):
        self._s1 = stream1
        self._s2 = stream2

    def write(self, data):
        self._s1.write(data)
        try:
            self._s2.write(data)
        except ValueError:
            pass  # file already closed during interpreter shutdown

    def flush(self):
        self._s1.flush()
        try:
            self._s2.flush()
        except ValueError:
            pass  # file already closed during interpreter shutdown

    # Support attributes that tqdm and others may probe
    def isatty(self):
        return hasattr(self._s1, "isatty") and self._s1.isatty()

    @property
    def encoding(self):
        return getattr(self._s1, "encoding", "utf-8")


def save_metadata(log_dir, cfg_dict, extra=None):
    """Write experiment metadata to log_dir/metadata.json.

    Parameters
    ----------
    cfg_dict : dict
        Resolved config (from OmegaConf.to_container).
    extra : dict, optional
        Additional key-value pairs (e.g. start_time, results_dir).
    """
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "config": cfg_dict,
        "python": sys.version,
        "command": " ".join(sys.argv),
    }
    if extra:
        meta.update(extra)

    path = os.path.join(log_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def save_summary(log_dir, results_dir, seed=0):
    """Copy final metrics into the log directory and log a summary table.

    Looks for metrics_seed={seed}.json in results_dir.
    """
    logger = logging.getLogger()
    metrics_src = os.path.join(results_dir, f"metrics_seed={seed}.json")

    if not os.path.exists(metrics_src):
        logger.warning("Metrics file not found: %s", metrics_src)
        return None

    # Copy metrics to log dir
    metrics_dst = os.path.join(log_dir, f"metrics_seed={seed}.json")
    with open(metrics_src) as f:
        metrics = json.load(f)
    with open(metrics_dst, "w") as f:
        json.dump(metrics, f, indent=2)

    # Log a human-readable summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT RESULTS SUMMARY (seed=%d)", seed)
    logger.info("=" * 60)

    # Per-category results
    categories = [
        k for k in metrics if not k.startswith("mean_")
    ]
    if categories:
        # Determine which metric keys exist
        sample_metrics = metrics[categories[0]]
        has_clf = "classification_AUROC" in sample_metrics
        has_seg = "seg_AUROC" in sample_metrics

        header_parts = [f"{'Category':<15}"]
        if has_clf:
            header_parts.append(f"{'Img AUROC':>10} {'Img AP':>10} {'Img F1':>10}")
        if has_seg:
            header_parts.append(f"{'Px AUROC':>10} {'AUPRO':>10} {'Px F1':>10}")
        header = " | ".join(header_parts)
        logger.info(header)
        logger.info("-" * len(header))

        for cat in categories:
            m = metrics[cat]
            parts = [f"{cat:<15}"]
            if has_clf:
                parts.append(
                    f"{m.get('classification_AUROC', 0):>10.4f} "
                    f"{m.get('classification_AP', 0):>10.4f} "
                    f"{m.get('classification_F1', 0):>10.4f}"
                )
            if has_seg:
                parts.append(
                    f"{m.get('seg_AUROC', 0):>10.4f} "
                    f"{m.get('seg_AUPRO', 0):>10.4f} "
                    f"{m.get('seg_F1', 0):>10.4f}"
                )
            logger.info(" | ".join(parts))

        logger.info("-" * len(header))

    # Mean results
    mean_parts = [f"{'MEAN':<15}"]
    if "mean_classification_au_roc" in metrics:
        mean_parts.append(
            f"{metrics['mean_classification_au_roc']:>10.4f} "
            f"{metrics.get('mean_classification_ap', 0):>10.4f} "
            f"{metrics.get('mean_classification_f1', 0):>10.4f}"
        )
    if "mean_segmentation_au_roc" in metrics:
        mean_parts.append(
            f"{metrics['mean_segmentation_au_roc']:>10.4f} "
            f"{metrics.get('mean_au_pro', 0):>10.4f} "
            f"{metrics.get('mean_segmentation_f1', 0):>10.4f}"
        )
    logger.info(" | ".join(mean_parts))
    logger.info("=" * 60)

    logger.info("Metrics saved to: %s", metrics_dst)
    logger.info("Full results at:  %s", os.path.abspath(results_dir))

    return metrics
