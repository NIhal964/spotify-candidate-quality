# src/train.py

import argparse
import logging

from src.train_logistic import train_logistic
from src.train_lgbm import train_lgbm

logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="Train models and run experiments")
    parser.add_argument("--model", choices=["logistic", "lightgbm"], default="logistic")
    parser.add_argument("--experiment", choices=["audio", "genre", "full"], default="full")
    parser.add_argument("--save-model", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to persist the trained model (True/False).")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info(f"Running model={args.model} experiment={args.experiment} save_model={args.save_model}")

    if args.model == "logistic":
        train_logistic(experiment=args.experiment, save_model=args.save_model)
    else:
        train_lgbm(experiment=args.experiment, save_model=args.save_model)


if __name__ == "__main__":
    main()

# Legacy monolithic train function removed.
# Use the CLI dispatcher at the top of this file to run
# model-specific training: src/train_logistic.py and src/train_lgbm.py
