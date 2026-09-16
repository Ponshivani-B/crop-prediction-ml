from flask import Flask, render_template, request
import joblib
import json

app = Flask(__name__)


# --------------------------------------------------
# Load the trained ML model
# --------------------------------------------------

model = joblib.load("crop_model.pkl")


# --------------------------------------------------
# Load crop information from JSON
# --------------------------------------------------

with open("crop_info.json", "r", encoding="utf-8") as file:
    crop_info = json.load(file)


# --------------------------------------------------
# Crop image mapping
# --------------------------------------------------

crop_images = {
    "rice": "rice.jpg",
    "maize": "maize.jpg",
    "chickpea": "chickpea.jpg",
    "kidneybeans": "kidneybeans.jpg",
    "pigeonpeas": "pigeonpeas.jpg",
    "mothbeans": "mothbeans.jpg",
    "mungbean": "mungbean.jpg",
    "blackgram": "blackgram.jpg",
    "lentil": "lentil.jpg",
    "pomegranate": "pomegranate.jpg",
    "banana": "banana.jpg",
    "mango": "mango.jpg",
    "grapes": "grapes.jpg",
    "watermelon": "watermelon.jpg",
    "muskmelon": "muskmelon.jpg",
    "apple": "apple.jpg",
    "orange": "orange.jpg",
    "papaya": "papaya.jpg",
    "coconut": "coconut.jpg",
    "cotton": "cotton.jpg",
    "jute": "jute.jpg",
    "coffee": "coffee.jpg"
}


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Crop Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Get selected language
    language = request.form.get("language", "english")

    # Get input values
    nitrogen = float(request.form["nitrogen"])
    phosphorus = float(request.form["phosphorus"])
    potassium = float(request.form["potassium"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])


    # Prepare input for ML model
    input_data = [[
        nitrogen,
        phosphorus,
        potassium,
        temperature,
        humidity,
        ph,
        rainfall
    ]]


    # Predict crop
    prediction = model.predict(input_data)[0]

    # Convert crop name to lowercase
    crop = prediction.lower()


    # Get crop image
    image = crop_images.get(crop)


    # Get crop information
    info = crop_info.get(crop, {})


    # Render result page
    return render_template(
        "result.html",
        crop=prediction,
        image=image,
        info=info,
        language=language,
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        temperature=temperature,
        humidity=humidity,
        ph=ph,
        rainfall=rainfall
    )


# --------------------------------------------------
# Run Flask Application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)