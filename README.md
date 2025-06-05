# 🌸 Project: Image Classifier using TensorFlow

This repository contains an image classification model built using TensorFlow and Transfer Learning. It is trained on the Oxford Flowers 102 dataset and enables users to classify flower images via a command-line application.

## 🌟 Overview
The goal of this project is to:
- Train an image classifier using a pre-trained model (MobileNet, InceptionV3, etc.).
- Fine-tune the classifier for the Oxford Flowers 102 dataset.
- Save the trained model in `.h5` format.
- Use a command-line script (`predict.py`) to:
  - Predict the top K most likely classes.
  - Display class names using a JSON label map.

---

## 📊 Dataset
**Oxford Flowers 102**
- **Classes:** 102 categories of flowers.
- **Images:** Color images of different flowers.
- **Splits:** Training, validation, and testing sets.

---

## 🛠️ Implementation Details

### Libraries Used
- `TensorFlow`
- `TensorFlow Hub`
- `NumPy`
- `Matplotlib`
- `Pandas`
- `json`

  ### Model Pipeline
1. **Data Preprocessing**
   - Normalization and resizing.
   - Splitting into training/validation/test.

2. **Transfer Learning**
   - Use a pre-trained model from `tensorflow_hub`.
   - Freeze base layers and add custom dense layers.

3. **Training**
   - Loss function: Categorical Crossentropy.
   - Optimizer: Adam.
   - Regularization: Dropout / L2 to prevent overfitting.

4. **Saving**
   - Export the trained model to `.h5`.

5. **Inference**
   - Use `predict.py` to classify new images.
   - Show top-K predictions with class names using a JSON label map.

---

## 📁 Files in the Repository

| File/Folder | Description |
|-------------|-------------|
| `Image_Classifier_Project.ipynb` | Jupyter Notebook with model training code |
| `predict.py` | Script for classifying images from command line |
| `label_map.json` | JSON file mapping class indices to names |
| `test_images/` | Sample images for inference |
| `1738440917.h5` | Saved model (HDF5 format) |

---

## 🚀 How to Run the Project

### Prerequisites
Install dependencies:
```bash
pip install tensorflow tensorflow-hub numpy pandas matplotlib
```

### Steps
1. Clone the repo:
```bash
git clone <https://github.com/LinaAbufaraha/Flower_Image_Classifier----TensorFlow>
cd <Create-Your-Own-Image-Classifier---TensorFlow>
```

2. Run the notebook:
```bash
jupyter notebook Image_Classifier_Project.ipynb
```

3. Predict using command line:
```bash
python predict.py --image path/to/image.jpg --model 1738440917.h5 --top_k 5 --category_names label_map.json
```

---

