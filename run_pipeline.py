"""One-command end-to-end FlowCast pipeline runner (M1 -> M6).

Satisfies PRD Section 3.3 reproducibility criterion:
Full pipeline reruns deterministically from raw inputs with a single command.
"""

import argparse
import time

from src.clean import run_cleaning
from src.dl_model import run_deep_learning_pipeline
from src.features import run_feature_engineering
from src.ingest import run_ingestion
from src.ml_models import train_classical_models
from src.utils import setup_logger

logger = setup_logger("FlowCast_Pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FlowCast end-to-end pipeline.")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--skip-dl",
        action="store_true",
        help="Run M1->M5 only (skip PyTorch LSTM/BiLSTM training).",
    )
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=== Starting FlowCast End-to-End Pipeline (M1 -> M6) ===")

    run_ingestion(args.config)
    run_cleaning(args.config)
    run_feature_engineering(args.config)
    train_classical_models(args.config)

    if not args.skip_dl:
        run_deep_learning_pipeline(args.config)
    else:
        logger.info("Skipping M6 Deep Learning stage (--skip-dl flag set).")

    logger.info(f"=== FlowCast Pipeline Finished in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()