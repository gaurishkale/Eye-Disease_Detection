# 👁️ Eye Disease Detection
### AI-Powered Retinal Image Screening using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Accuracy](https://img.shields.io/badge/Accuracy-~90%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What is This Project?

Eye diseases like **Diabetic Retinopathy, Cataract, Glaucoma, and Macular Degeneration** 
are among the leading causes of blindness worldwide — yet most cases are **preventable 
with early detection.**

This project uses a **ConvNeXtV2-based deep learning model** trained on the **ODIR-19 
(Ocular Disease Intelligent Recognition)** dataset to automatically classify retinal 
fundus images into 8 disease categories — making professional-grade eye screening 
accessible without a specialist.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| 🏆 Accuracy | **~90%** |
| 📋 Evaluation | Precision, Recall, F1-Score |
| 🧠 Architecture | ConvNeXtV2 (Transfer Learning) |
| 📦 Dataset | ODIR-19 + Ocular Disease Dataset |
| 🔢 Classes | 8 Eye Disease Categories |

---

## 🔬 Detectable Conditions

| Label | Condition |
|-------|-----------|
| N | Normal (Healthy Eye) |
| D | Diabetic Retinopathy |
| G | Glaucoma |
| C | Cataract |
| A | Age-related Macular Degeneration |
| H | Hypertensive Retinopathy |
| M | Myopia (Pathological) |
| O | Other Abnormalities |

---

## 🏗️ Project Structure

```
Eye-Disease_Detection/
│
├── main.py                # Entry point — runs prediction pipeline
├── predict.py             # Preprocessing + model inference logic
├── model_loader.py        # Loads trained ConvNeXtV2 model
├── disease_mapping.json   # Maps model output indices to disease names
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/gaurishkale/Eye-Disease_Detection.git
cd Eye-Disease_Detection
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run prediction
```bash
python main.py --image_path /path/to/fundus_image.jpg
```

### 5. Example output
```
Loading model...           ✅
Preprocessing image...     ✅
Running inference...       ✅

Predicted Condition : Diabetic Retinopathy
Confidence          : 91.4%
Recommendation      : Please consult an ophthalmologist immediately.
```

---

## ⚙️ How It Works

```
Input Fundus Image
        ↓
Preprocess (resize 224×224, normalize pixel values)
        ↓
ConvNeXtV2 Model Inference
        ↓
Softmax → Disease Class Probabilities
        ↓
disease_mapping.json → Human-readable label
        ↓
Output: Disease Name + Confidence + Recommendation
```

---

## 🧠 Model Architecture

- **Base Model:** ConvNeXtV2 (pretrained on ImageNet)
- **Technique:** Transfer Learning — frozen base layers, fine-tuned classification head
- **Input Shape:** 224 × 224 × 3 (RGB fundus images)
- **Output:** Softmax over 8 disease classes
- **Augmentation:** Random flip, rotation, zoom, brightness adjustment
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam with learning rate scheduling

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Core language |
| **TensorFlow / Keras** | Deep learning framework |
| **ConvNeXtV2** | Pretrained image classification backbone |
| **OpenCV / PIL** | Image loading and preprocessing |
| **NumPy** | Array operations |
| **JSON** | Disease label mapping |

---

## 📊 Dataset

- **Source:** [ODIR-19 — Ocular Disease Intelligent Recognition](https://odir2019.grand-challenge.org/)
- **Images:** Fundus photographs (left and right eye)
- **Labels:** 8 disease categories (multi-label)
- **Preprocessing:** Resized to 224×224, normalized to [0, 1]

> Dataset not included in this repo due to size and licensing.
> Download from the official ODIR-19 challenge page or Kaggle.

---

## 🔮 Future Enhancements

- [ ] Streamlit / Flask web interface for live image upload
- [ ] Grad-CAM visualization to highlight disease regions in the retina
- [ ] Mobile app integration for on-device screening
- [ ] Multi-label classification support (patient can have multiple conditions)
- [ ] Integration with nearby ophthalmologist recommendation system

---

## 💡 Why This Matters

> According to the **WHO**, at least **2.2 billion people** have vision impairment globally.
> Over **1 billion cases** could have been prevented with timely detection and treatment.
> AI-powered screening tools can bridge the gap where ophthalmologists are scarce.

---

## 🙋 About

Built by **Gaurish Kale** as part of an AI/ML project exploring deep learning 
applications in medical image analysis.

- 📧 kalegaurish03@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/gaurishkale16)
- 🐙 [GitHub](https://github.com/gaurishkale)

---

## 📄 License

This project is licensed under the MIT License.
Dataset is owned by ODIR-19 Challenge organizers — not included in this repo.

---

⭐ If you found this project useful, please give it a star!
