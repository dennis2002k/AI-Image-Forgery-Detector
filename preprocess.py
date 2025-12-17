import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATASET_DIR = r"dataset/original"  
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DATA_PATH = "test_data"
TRAIN_DATA_PATH = "train_data"

# Pre-process data with ela
def create_ela_image(image_path, quality=90, scale=10):
    original = Image.open(image_path).convert("RGB")
    buffer = "temp.jpg"
    original.save(buffer, 'JPEG', quality=quality)
    compressed = Image.open(buffer)
    ela_image = ImageChops.difference(original, compressed)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale_factor = scale * 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale_factor)
    
    ela_image = ela_image.resize(IMG_SIZE)
    return np.array(ela_image).astype("float32") / 255.0


def preprocess():
    # Load dataset (true and false) and pre-process data
    X, y = [], []
    print("Creating ELA images...")
    for label, classname in enumerate(["fake", "real"]):
        folder = os.path.join(DATASET_DIR, classname)
        for filename in os.listdir(folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")):
                continue
            try:
                img = create_ela_image(os.path.join(folder, filename))
                X.append(img)
                y.append(label)
            except:
                print("Skipped bad image:", filename)

    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} images: Real={sum(y==1)}, Fake={sum(y==0)}")

    # Split data to train and test data (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    np.save(f"{TEST_DATA_PATH}/X_test.npy", X_test)
    np.save(f"{TEST_DATA_PATH}/y_test.npy", y_test)

    np.save(f"{TRAIN_DATA_PATH}/X_train.npy", X_train)
    np.save(f"{TRAIN_DATA_PATH}/y_train.npy", y_train)


if __name__ == "__main__":
    preprocess()