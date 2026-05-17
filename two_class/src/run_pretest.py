"""Command-line entry point for the liquid residue pretest workflow."""

import argparse
from pathlib import Path

from config import DEFAULT_ANNOTATION_CSV, PROJECT_ROOT


def str_to_bool(value: str) -> bool:
    """Parse common string booleans for CLI arguments."""
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Bottle liquid residue pretest workflow")
    parser.add_argument(
        "--feature_mode",
        choices=["traditional", "deep", "fused"],
        default="traditional",
        help="Feature type: traditional, deep, or fused.",
    )
    parser.add_argument("--csv", default=str(DEFAULT_ANNOTATION_CSV), help="Path to annotations.csv.")
    parser.add_argument(
        "--image_root",
        default=str(PROJECT_ROOT),
        help="Root directory for relative image_path values in annotations.csv.",
    )
    parser.add_argument("--test_size", type=float, default=0.25, help="Test split ratio.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--save_crops", type=str_to_bool, default=True, help="Whether to save bbox crops.")
    parser.add_argument(
        "--run_name",
        default=None,
        help="Experiment name under outputs/runs/. If omitted, a timestamp is used.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Optional root directory for experiment runs. Defaults to outputs/runs/.",
    )
    parser.add_argument(
        "--deep_model",
        choices=["resnet18", "resnet50"],
        default="resnet18",
        help="Torchvision ResNet model for deep embeddings.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        choices=["LogisticRegression", "SVM_RBF", "RandomForest"],
        default=None,
        help="Optional classifier list. Defaults to all candidate models.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the workflow and print a compact summary."""
    args = parse_args()
    try:
        from train_eval import run_experiment

        result = run_experiment(
            csv_path=Path(args.csv),
            image_root=Path(args.image_root),
            feature_mode=args.feature_mode,
            test_size=args.test_size,
            random_state=args.random_state,
            save_crops=args.save_crops,
            deep_model=args.deep_model,
            model_names=args.model_names,
            run_name=args.run_name,
            output_root=Path(args.output_root) if args.output_root else None,
        )
    except ModuleNotFoundError as exc:
        print("\n[Pretest stopped]")
        print(f"Missing Python dependency: {exc.name}")
        print("\nInstall required packages first:")
        print("pip install -r requirements.txt")
        return 1
    except Exception as exc:
        print("\n[Pretest stopped]")
        print(str(exc))
        print("\nChecklist:")
        print("1. Put images under data/images/.")
        print("2. Update data/annotations.csv with correct image_path and bbox values.")
        print("3. Make sure has_liquid contains both 0 and 1 classes.")
        return 1

    print("\n[Pretest completed]")
    print(f"Run directory: {result['run_dir']}")
    print(f"Best model: {result['best_model']}")
    print(f"Feature status: {result['feature_status']}")
    print(f"Features: {result['feature_csv']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Best model file: {result['best_model_path']}")
    print(f"Scaler file: {result['scaler_path']}")
    print("\nMetrics table:")
    print(result["metrics"].to_string(index=False))
    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
