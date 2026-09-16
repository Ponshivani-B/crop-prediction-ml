import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

data = pd.read_csv("dataset\crop_data.csv")

print("Dataset shape:", data.shape)
print("\nColumns:")
print(data.columns)

# --------------------------------------------------
# 2. Check Missing Values
# --------------------------------------------------

print("\nMissing values:")
print(data.isnull().sum())

# --------------------------------------------------
# 3. Separate Features and Target
# --------------------------------------------------

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

X = data[features]
y = data["label"]

# --------------------------------------------------
# 4. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 5. Create Random Forest Model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# --------------------------------------------------
# 6. Train Model
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed!")

# --------------------------------------------------
# 7. Test Model
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 8. Save Model
# --------------------------------------------------

joblib.dump(model, "crop_model.pkl")

print("\nModel saved as crop_model.pkl")