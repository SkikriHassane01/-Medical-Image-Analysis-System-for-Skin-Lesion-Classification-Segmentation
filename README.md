# 🩺 Skin Lesion Classification & Segmentation System

## Project Overview
This project builds an intelligent system for analyzing dermatoscopic images to:
1. **Classify** skin lesions as either malignant(melanoma) or non-malignant(none-melanoma)
2. **Segment** lesion regions in dermoscopic images

## 📁 Project Structure
```
skin_lesion_project/
│
├── README.md
├── LICENSE
├── config.py
├── pyprject.toml (poetry)
│
├── data/
│   ├── Classification/
|   |   |──Test_Data
|   |   └──Training_Data
│   └── Segmentation/
|       |──Test_Data
|       └──Training_Data
│
├── preprocessing/
│   ├── __init__.py
│   ├── image_utils.py
│   └── data_loader.py
│
├── classification/
│   ├── __init__.py
│   ├── model.py (pytorch model)
│   ├── train.py
│   └── predict.py
│
├── segmentation/
│   ├── __init__.py
│   ├── model.py (keras model)
│   ├── train.py
│   └── predict.py
│
├── inference/
│   ├── __init__.py
│   └── visualize.py
│
└── app/
    ├── streamlit_app.py
    └── utils.py


```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python preprocessing/data_loader.py
```

### 3. Train Models
```bash
# Train classification model (PyTorch)
python classification/train.py

# Train segmentation model (TensorFlow/Keras)
python segmentation/train.py
```

### 4. Run Inference
```bash
python inference/pipeline.py --image path/to/image.jpg
```

### 5. Launch Web App
```bash
streamlit run app/streamlit_app.py
```

## 📊 Model Architecture

### Classification (PyTorch)
- **Base Model**: ResNet50 or EfficientNet
- **Classes**: 7 types of skin lesions
- **Input**: 224x224 RGB images
- **Output**: Class probabilities

### Segmentation (TensorFlow/Keras)
- **Architecture**: U-Net
- **Input**: 256x256 RGB images
- **Output**: Binary mask (lesion/background)

## 🔧 Technologies Used
- **PyTorch**: Classification model training
- **TensorFlow/Keras**: Segmentation model training
- **OpenCV**: Image preprocessing and visualization
- **scikit-image**: Advanced image processing
- **Streamlit**: Web interface

## 📈 Performance Metrics
- **Classification**: Accuracy, Precision, Recall, F1-score
- **Segmentation**: IoU, Dice coefficient, Pixel accuracy


## 📝 Usage Example
```python
from inference.pipeline import SkinLesionAnalyzer

# Initialize analyzer
analyzer = SkinLesionAnalyzer()

# Analyze image
results = analyzer.analyze_image('path/to/image.jpg')

print(f"Classification: {results['classification']}")
print(f"Confidence: {results['confidence']:.2f}")
# Segmentation mask is saved automatically
```

## 📚 Dataset
Designed to work with:
- **ISIC 2016 Dataset**: [**here**](https://challenge.isic-archive.com/data/)
- **Classification Challenge**: [**here**](https://challenge.isic-archive.com/landing/2016/39/)
- **Segmentation Challenge** : [**here**](https://challenge.isic-archive.com/landing/2016/37/)

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License
MIT License - feel free to use for research and educational purposes.

---