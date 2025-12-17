### Deepfake Image Detection 

-This project implements a deep learning–based deepfake image detection system
 using a **pre-trained MobileNetV2** model (ImageNet weights) fine-tuned on the
 CASIA 2.0 dataset.

-Error Level Analysis (ELA) is used as a preprocessing technique to highlight
 compression artifacts commonly introduced during image manipulation.
  
-A Streamlit web application allows users to upload images and receive real-time predictions.

---

### Project Details

- Model: MobileNetV2 (ImageNet pre-trained, fine-tuned)
- Dataset: [CASIA 2.0](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset)
    - Total images: 12,614  
    - Real: 7,491  
    - Fake: 5,123
- Test Accuracy: ~90%
- Trained model saved to forgery_detector.keras

---

### Model Pipeline 

1. Load image
2. Apply ELA preprocessing
3. Resize and normalize image
4. Data Augmentation (rotation, flip, zoom, contrast)
5. Feed into MobileNetV2 pre-trained on ImageNet
6. Fine-tune on CASIA 2.0 dataset
7. Classify image as REAL or FAKE
8. Save final weights 

The same preprocessing pipeline is applied consistently during:
- Training
- Evaluation
- Inference (Streamlit app)


---

### Evaluation

Model performance is analyzed using:
- Accuracy and loss curves
- Confusion matrix
- ROC–AUC curve

All evaluation results and plots are saved in the `results/` folder.

---

### Files 
- preprocess.py ---> Generates ELA-preprocessed images
- train.py ----------> Trains and fine-tunes the MobileNetV2 model
- evaluate.py ------> Evaluates the trained model
- app.py -----------> Runs the streamlit application for Interactive testing
- requirements.txt -> Python  dependencies

---

### How to run 
``` bash
pip install -r requirements.txt
```

## How to train and evaluate the model (optional)(model already trained) 
```bash
python preprocess.py
python train.py
python evaluate.py
```

---


### Test the app
A few sample images are included in `demo_images/` to try the app:

1. Run the app: `python -m streamlit run app.py`  
2. Upload any image from `demo_images/`  
3. See the model’s prediction: REAL or FAKE
