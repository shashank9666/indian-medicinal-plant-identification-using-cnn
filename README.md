# 🌿 Indian Medicinal Plant Identification using CNN

## 📌 Project Description
This project aims to identify **Indian medicinal plants** using a **Convolutional Neural Network (CNN)** based on **MobileNetV2**.  
A **Flask web application** is used to serve the trained model and provide an interactive interface where users can upload plant leaf images and receive predictions along with their **medicinal uses**.

---

## 📂 Dataset
The dataset used in this project consists of images of various Indian medicinal plant leaves.

🔗 **Dataset Link (Kaggle):**  
https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset

---

## ⚠️ IMPORTANT: TensorFlow & Keras Compatibility
> 🚨 **This project works ONLY with supported versions of TensorFlow and Keras.**

The model was trained and tested using:
- **TensorFlow:** `2.13.x`
- **Keras:** `2.13.x`

Using newer or incompatible versions **may cause runtime errors or model loading failures**.

✔ Recommended versions are listed in `requirements.txt`  
❌ Do NOT upgrade TensorFlow/Keras arbitrarily

---

## 🗂️ Project Structure
/
│
├── app.py
├── Model_Mobilenet.h5
├── requirements.txt
│
├── static/
│ ├── css/
│ ├── js/
│ └── image/
│
└── templates/
├── index.html
├── home.html
├── prediction.html
└── code.html


---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone <repository-url>
cd <repository-folder>
2️⃣ Create required folders

Create the following structure if not already present:

static/

css/ → CSS files

js/ → JavaScript files

image/ → Images

templates/

HTML files
# (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

## Install dependencies
pip install -r requirements.txt
⚠️ Do not install TensorFlow manually without version control.

## Run the Flask application
python app.py

# Usage

Open your browser and go to:

http://127.0.0.1:5000


Upload an image of a medicinal plant leaf.

The application predicts:

Plant name

Medicinal uses

# Deployment

This project can be deployed on Render (CPU-only) using:

tensorflow-cpu

gunicorn

Refer to deployment instructions for proper configuration.

## Technologies Used

Python

Flask

TensorFlow (CNN – MobileNetV2)

Keras

NumPy

Pillow

HTML / CSS / Bootstrap


# Notes

Initial startup may be slow due to TensorFlow model loading.

Large models are not ideal for free hosting tiers.

Designed for educational and demonstration purposes.