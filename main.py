"""
Decision Intelligence Framework — CLI entry point.

Usage:
    python main.py --pipeline train    --config configs/global.yaml
    python main.py --pipeline validate --config configs/global.yaml
    python main.py --pipeline simulate --config configs/global.yaml
"""
from __future__ import annotations
import argparse
import yaml

from decision_intelligence.pipelines.train_pipeline import TrainPipeline
from decision_intelligence.pipelines.validate_pipeline import ValidatePipeline
from decision_intelligence.pipelines.simulate_pipeline import SimulatePipeline

PIPELINES = {
    "train":    TrainPipeline,
    "validate": ValidatePipeline,
    "simulate": SimulatePipeline,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision Intelligence Framework")
    parser.add_argument("--pipeline", choices=PIPELINES.keys(), required=True)
    parser.add_argument("--config",   default="configs/global.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    PIPELINES[args.pipeline](config=config).run()


if __name__ == "__main__":
    main()
