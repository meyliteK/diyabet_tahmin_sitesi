import pickle
import numpy as np

# Modeli yükle
with open("knn_model.pkl", "rb") as file:
    knn_model = pickle.load(file)

# Scaler'ı yükle
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Diabetes.csv'den Outcome = 0 olan bir veri (örnek)
test_data = np.array([[10,125,70,26,115,31.1,0.205,41]])  # Bu veriyi normalize etmeden kullanırsak hata alırız

# Test verisini normalize et
test_data_normalized = scaler.transform(test_data)

# Tahmin yap
prediction = knn_model.predict(test_data_normalized)
print("Tahmin Sonucu:", prediction)  # 0 veya 1 döndürmeli

