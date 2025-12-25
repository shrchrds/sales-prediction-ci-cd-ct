# continuous_training.py

import json
from pathlib import Path

from training_pipeline import train_and_evaluate


def main():
    # 1. Train model using the shared pipeline
    metrics = train_and_evaluate()

    print("=== Continuous Training Run Completed ===")
    print(f"R2 Score: {metrics['r2']:.3f}")
    print(f"MSE: {metrics['mse']:.3f}")
    print(f"Model saved at: {metrics['model_path']}")
    print(f"Data used: {metrics['data_path']}")
    print(f"Trained at (UTC): {metrics['trained_at']}")

    # 2. Save metrics with timestamp so every CT run is tracked
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)

    # make filename filesystem-safe
    run_id = metrics["trained_at"].replace(":", "-")
    metrics_path = metrics_dir / f"metrics_{run_id}.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
