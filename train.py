import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DATA_PATH = "test_data"
TRAIN_DATA_PATH = "train_data"
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 60
MODEL_KERAS  = "forgery_detector.keras"

# Save histories from 2 separate model.fits as one combined
def combine_histories(h1, h2):
    combined = {}
    for key in h1.history:
        combined[key] = h1.history[key] + h2.history[key]
    return combined

gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)
print(tf.__version__)


# Load test arrays
X_test = np.load(f"{TEST_DATA_PATH}/X_test.npy")
y_test = np.load(f"{TEST_DATA_PATH}/y_test.npy")

# Load train arrays
X_train = np.load(f"{TRAIN_DATA_PATH}/X_train.npy")
y_train = np.load(f"{TRAIN_DATA_PATH}/y_train.npy")

# Augment data (rotation, flip, zoom, contrast)
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1)
])

train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).map(
    lambda x, y: (data_augmentation(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Compute class weights to create balance between real and fake
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: class_weights[0], 1: class_weights[1]}
print("Class weights:", class_weights)


# MODEL: MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False # forbid to change pretrained model weights during initial training

# Simple model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# Initial training
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights
)

# Fine-tuning
base_model.trainable = True # Allow pre-trained model's weights to be changed while fine-tuning

# Freeze bottom layers
for layer in base_model.layers[:-100]:
    layer.trainable = False

# Compile with smaller learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
# 
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# Start fine-tuning
history_fine = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

combined_history = combine_histories(history, history_fine)
np.save("history.npy", combined_history)

# Evaluation
loss, acc = model.evaluate(test_ds)
print(f"\nFinal Test Accuracy: {acc:.4f}")


# Save Keras native format (.keras)
model.save(MODEL_KERAS)
print(f"Saved model to: {MODEL_KERAS}")
