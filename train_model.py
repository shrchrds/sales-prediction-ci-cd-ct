# train_model.py

from training_pipeline import train_and_evaluate


if __name__ == "__main__":
    metrics = train_and_evaluate()

    print("Model trained successfully, ready to push into remote repository")
    print(f"R2 Score: {metrics['r2']:.3f}")
    print(f"MSE: {metrics['mse']:.3f}")
