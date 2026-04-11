from __future__ import annotations

import argparse
import json
from pathlib import Path

from recipe_ranker import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_RECIPES_PATH,
    RecipeRankerService,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the recipe ranking model without data leakage.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH, help="Path to recipes_dataset_ver1.csv")
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES_PATH, help="Path to recipes_processed.json")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH, help="Output artifact path (.joblib)")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH, help="Output metrics path (.json)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    service = RecipeRankerService.train_from_paths(
        labels_path=args.labels,
        recipes_path=args.recipes,
        artifact_path=args.artifact,
        metrics_path=args.metrics,
    )

    print(f"Saved artifact to: {args.artifact}")
    print(f"Saved metrics to: {args.metrics}")
    print(json.dumps(service.evaluation_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
