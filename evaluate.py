import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


def load_model():
    return tf.keras.models.load_model("forgery_detector2.keras")

model = load_model()

# Load arrays
X_test = np.load("test_data/X_test.npy")
y_test = np.load("test_data/y_test.npy")

# Recreate tf.data.Dataset
BATCH_SIZE = 32  # same as training

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Evaluation
loss, acc = model.evaluate(test_ds)
print(f"\nFinal Test Accuracy: {acc:.4f}")


history = np.load("history.npy", allow_pickle=True).item()

# Accuracy
plt.figure()
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.savefig("results/accuracy_curve.png")
plt.show()

# Loss
plt.figure()
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.savefig("results/loss_curve.png")
plt.show()


y_probs = model.predict(test_ds).ravel()
y_pred = (y_probs > 0.5).astype(int)


cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["FAKE", "REAL"])
plt.yticks([0, 1], ["FAKE", "REAL"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()


fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="AUC = 0.500 (random guess)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig("results/roc_curve.png")
plt.show()