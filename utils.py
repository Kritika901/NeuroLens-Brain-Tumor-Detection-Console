import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model


def save_model(model, model_path="models/brain_tumor_model.keras"):
    """Save a trained Keras model to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved at: {model_path}")


def load_saved_model(model_path="models/brain_tumor_model.keras"):
    """Load a Keras model that was saved earlier."""
    model = load_model(model_path)
    print(f"Model loaded from: {model_path}")
    return model


def plot_training_history(history, save_path="images/training_history.png"):
    """Plot training/validation accuracy and loss curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training history plot saved at: {save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="images/confusion_matrix.png"):
    """Plot a confusion matrix heatmap for the test set predictions."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved at: {save_path}")

    plt.show()


def print_classification_report(y_true, y_pred, class_names):
    """Print precision/recall/F1 score for each class."""
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report\n")
    print(report)
    return report
