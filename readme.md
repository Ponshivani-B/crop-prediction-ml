#  AI-Based Crop Recommendation System

An AI-powered web application that recommends the most suitable crop based on soil and environmental conditions.

##  Project Overview

The system uses a Machine Learning model to recommend a suitable crop based on:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

The application is built using Flask and a Random Forest Classification model.

##  Features

-  AI-based crop recommendation
-  Supports 22 crop categories
-  Crop-specific images
-  Prediction confidence
-  Multilingual output (Hindi, English, Kannada)
-  Printable crop recommendation report
-  Responsive web interface

##  Machine Learning

### Algorithm

Random Forest Classifier

### Dataset

Crop Recommendation Dataset containing 2,200 records and 22 crop classes.

### Input Features

| Feature | Description |
|---|---|
| N | Nitrogen |
| P | Phosphorus |
| K | Potassium |
| Temperature | Temperature in °C |
| Humidity | Humidity percentage |
| pH | Soil pH |
| Rainfall | Rainfall in mm |

## 🌾 Supported Crops

- Rice
- Maize
- Chickpea
- Kidney Beans
- Pigeon Peas
- Moth Beans
- Mung Bean
- Black Gram
- Lentil
- Pomegranate
- Banana
- Mango
- Grapes
- Watermelon
- Muskmelon
- Apple
- Orange
- Papaya
- Coconut
- Cotton
- Jute
- Coffee

## 🛠️ Technologies Used

- Python
- Flask
- Scikit-learn
- Random Forest
- Pandas
- NumPy
- HTML
- CSS
- JavaScript
- JSON

## 📂 Project Structure

```text
crop_prediction_system/
│
├── app.py
├── train_model.py
├── crop_model.pkl
├── crop_info.json
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html
│   └── result.html
|
├── dataset/
|   |__crop_data.csv
|
└── static/
    ├── css/
    └── images/
