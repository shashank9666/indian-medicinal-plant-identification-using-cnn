from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
from io import BytesIO
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

# =========================
# Load model safely (Render compatible)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Model_Mobilenet.h5")
model = load_model(MODEL_PATH)

TARGET_SIZE = (224, 224)

# =========================
# Data (Classes & Uses)
# =========================
leavesArray = [
    "Aloevera","Amla","Amruthaballi","Arali","Astma_weed","Badipala","Balloon_Vine",
    "Bamboo","Beans","Betel","Bhrami","Bringaraja","Caricature","Castor",
    "Catharanthus","Chakte","Chilly","Citron lime (herelikai)","Coffee",
    "Commonrue(naagdalli)","Coriender","Curry","Doddpathre","Drumstick","Ekka",
    "Eucalyptus","Ganigale","Ganike","Gasagase","Ginger","Globe Amarnath","Guava",
    "Henna","Hibiscus","Honge","Insulin","Jackfruit","Jasmine","Kambajala",
    "Kasambruga","Kohlrabi","Lantana","Lemon","Lemongrass","Malabar_Nut",
    "Malabar_Spinach","Mango","Marigold","Mint","Neem","Nelavembu","Nerale",
    "Nooni","Onion","Padri","Palak(Spinach)","Papaya","Parijatha","Pea","Pepper",
    "Pomoegranate","Pumpkin","Raddish","Rose","Sampige","Sapota","Seethaashoka",
    "Seethapala","Spinach1","Tamarind","Taro","Tecoma","Thumbe","Tomato","Tulsi",
    "Turmeric","ashoka","camphor","kamakasturi","kepala"
]

usesArray = [
    "Aloe Vera: Widely used for skin care, digestion, oral health, and wound healing.",
    "Amla: Rich in Vitamin C, boosts immunity and digestion.",
    "Amruthaballi: Boosts immunity, helps diabetes and respiratory issues.",
    "Arali: Anti-inflammatory, improves brain oxygenation (toxic if misused).",
    "Asthma weed: Used in respiratory relief.",
    "Badipala: Used traditionally for kidney and urinary problems.",
    "Balloon Vine: Relieves joint pain and inflammation.",
    "Bamboo: Antibacterial, nutritious bamboo shoots.",
    "Beans: High protein and fiber, good for digestion.",
    "Betel: Anti-diabetic, antioxidant, wound healing.",
    "Bhrami: Enhances memory and reduces stress.",
    "Bringaraja: Promotes hair growth.",
    "Caricature: Ornamental plant.",
    "Castor: Castor oil used for skin and digestion.",
    "Catharanthus: Used in cancer treatment.",
    "Chakte: Hardwood tree used traditionally.",
    "Chilly: Improves metabolism and pain relief.",
    "Citron lime: Rich in Vitamin C.",
    "Coffee: Antioxidant-rich stimulant.",
    "Common rue: Used traditionally for medicinal purposes.",
    "Coriander: Digestive and anti-inflammatory.",
    "Curry leaves: Antioxidant and anti-inflammatory.",
    "Doddpathre: Nutrient-rich leafy plant.",
    "Drumstick: Anti-inflammatory and nutritious.",
    "Ekka: Used in traditional medicine.",
    "Eucalyptus: Respiratory relief.",
    "Ganigale: Medicinal aromatic grass.",
    "Ganike: Antibacterial and anti-inflammatory.",
    "Gasagase: Nutritious poppy seeds.",
    "Ginger: Digestive and anti-inflammatory.",
    "Globe Amarnath: Ornamental use.",
    "Guava: Rich in Vitamin C.",
    "Henna: Hair care and skin treatment.",
    "Hibiscus: Blood pressure regulation.",
    "Honge: Medicinal and construction use.",
    "Insulin plant: Helps control blood sugar.",
    "Jackfruit: Nutritious fruit.",
    "Jasmine: Ornamental and fragrance.",
    "Kambajala: Traditional medicine.",
    "Kasambruga: Traditional medicine.",
    "Kohlrabi: Nutrient-rich vegetable.",
    "Lantana: Ornamental plant.",
    "Lemon: Vitamin C rich.",
    "Lemongrass: Herbal tea and digestion.",
    "Malabar Nut: Respiratory benefits.",
    "Malabar Spinach: Iron-rich leafy green.",
    "Mango: Nutritious tropical fruit.",
    "Marigold: Decorative and medicinal.",
    "Mint: Digestive and cooling.",
    "Neem: Antibacterial and antifungal.",
    "Nelavembu: Immune booster.",
    "Nerale: Culinary and medicinal.",
    "Nooni: Traditional medicine.",
    "Onion: Culinary staple.",
    "Padri: Medicinal leaves.",
    "Palak: Iron-rich leafy green.",
    "Papaya: Digestive benefits.",
    "Parijatha: Ornamental flowers.",
    "Pea: Protein-rich legume.",
    "Pepper: Digestive spice.",
    "Pomegranate: Antioxidant-rich fruit.",
    "Pumpkin: Nutritious vegetable.",
    "Radish: Digestive root vegetable.",
    "Rose: Fragrance and decoration.",
    "Sampige: Ornamental flower.",
    "Sapota: Sweet tropical fruit.",
    "Seethaashoka: Ornamental tree.",
    "Seethapala: Custard apple fruit.",
    "Spinach: Iron-rich leafy green.",
    "Tamarind: Digestive fruit.",
    "Taro: Starchy root vegetable.",
    "Tecoma: Ornamental flowering plant.",
    "Thumbe: Traditional medicinal plant.",
    "Tomato: Antioxidant-rich fruit.",
    "Tulsi: Immunity booster.",
    "Turmeric: Anti-inflammatory spice.",
    "Ashoka: Ornamental tree.",
    "Camphor: Respiratory relief.",
    "Kamakasturi: Fragrant medicinal plant.",
    "Kepala: Traditional medicinal uses."
]

# =========================
# Image preprocessing
# =========================
def preprocess_image(file):
    img = image.load_img(BytesIO(file.read()), target_size=TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# =========================
# Routes
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/code")
def code():
    return render_template("code.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "GET":
        return render_template("prediction.html")

    if "file" not in request.files:
        return render_template("home.html", prediction="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("home.html", prediction="No selected file")

    img_array = preprocess_image(file)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    if predicted_class >= len(leavesArray):
        return render_template(
            "prediction.html",
            prediction="Prediction error",
            predictionUse="Invalid model output"
        )

    answer = leavesArray[predicted_class]
    use = usesArray[predicted_class]

    return render_template(
        "prediction.html",
        prediction=f"Predicted plant : {answer} Leaf",
        predictionUse=use
    )

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    app.run()
