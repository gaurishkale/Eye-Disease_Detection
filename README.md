👁️ Eye Disease Detection

A Machine Learning-powered Eye Disease Detection system that analyzes eye images and predicts potential eye conditions using a pretrained model. This project enables automated classification of retinal images—making eye health screening more accessible and efficient.

🧠 Project Overview

Eye diseases such as Diabetic Retinopathy, Cataract, Glaucoma, and other vision-impairing conditions are major global health challenges. Early detection plays a crucial role in preventing severe vision loss. This project uses deep learning and image classification techniques to automate the process of recognizing eye diseases from retinal images.

💡 The system loads a trained model and performs inference on input images to classify them into disease categories. It serves as the foundation for building web or mobile applications for fast eye health screening.

📌 Key Features

🖼️ Image-based Disease Prediction – Input retinal/eye images and receive a predicted disease label.

🤖 Pretrained Model Inference – Uses a machine learning model to generate predictions for unseen images.

📟 Modular Codebase – Separate scripts for loading the model, preprocessing, and prediction logic.

🧪 Supports Extension – Can be integrated into a Flask or FastAPI server to power a web app interface.

🧰 Tech Stack

Language: Python

ML Framework: TensorFlow / PyTorch (depending on your model choice)

Image Processing: OpenCV / PIL

Prediction Pipeline: Model loader + preprocess + inference

Utilities: JSON mapping file for label decoding

📁 Project Structure
Eye-Disease_Detection/
├── main.py              # Main script to run prediction
├── predict.py           # Prediction logic (preprocessing + model inference)
├── model_loader.py      # Loads the trained model
├── disease_mapping.json # Maps numeric model outputs to disease names
├── requirements.txt     # Python dependency list
├── README.md            # Project documentation
🔍 How It Works

Load Model: A trained deep learning model is loaded into memory.

Preprocess Image: Uploaded retinal image is resized and normalized.

Predict: The model processes the image and predicts the eye disease class.

Output Result: Returns the disease label (e.g., Normal, Cataract, Diabetic Retinopathy, Glaucoma).

🚀 Getting Started
Prerequisites

Make sure you have Python 3.8+ installed.

Installation

Clone the repository:

git clone https://github.com/gaurishkale/Eye-Disease_Detection.git
cd Eye-Disease_Detection

Install dependencies:

pip install -r requirements.txt
Usage

To run predictions:

python main.py --image_path /path/to/eye_image.jpg

Replace /path/to/eye_image.jpg with the retina/fundus image you want to classify.

📈 Future Enhancements

Add a web interface (Flask/Streamlit) for uploading images live.

Integrate a mobile app for on-device screening.

Improve model accuracy using larger medical imaging datasets.

📚 References & Resources

This project is inspired by open-source eye disease classification systems that use deep learning to predict conditions from retinal images.
