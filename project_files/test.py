import pickle
import numpy as np
import pandas as pd
import os

# Proje pkl dosyalarına path yoluyla erişme
knn_path = "../model_files/knn_model.pkl"
scaler_path = "../model_files/scaler.pkl"

# Modeli yükle
with open(knn_path, "rb") as file:
    knn_model = pickle.load(file)

# Scaler'ı yükle
with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

# Diabetes.csv'den Outcome = 0 olan bir veri (örnek)
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
test_data = pd.DataFrame([[12,140,82,43,325,39.2,0.528,58,1]], columns = features)  

# Test verisini normalize et outcome dahil etmeden
test_data_normalized = scaler.transform(test_data.iloc[:, :-1])

# Tahmin yap
prediction = knn_model.predict(test_data_normalized)
print("Tahmin Sonucu:", "Diyabet" if prediction[0] == 1 else "Sağlıklı")  # 0 veya 1 döndürmeli

