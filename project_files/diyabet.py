from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------------------------------------------------
# Flask uygulaması, templates ve static klasörleri bir üst dizinde olduğu 
# için template_folder ve static_folder parametrelerini belirtiyoruz.
# ------------------------------------------------------------------------
app = Flask(__name__,
            template_folder="../templates",
            static_folder="../static")

# ------------------------------------------------------------------------
# Model ve veri dosyalarının konumlarını tanımla:
# (Bu kodun çalışacağı dizin: project_files)
# ------------------------------------------------------------------------
# Şu an "project_files" klasöründeyiz. model_files ise bir üst dizinde.
# Dolayısıyla pkl dosyalarına "../model_files/..." şeklinde erişiyoruz.
knn_path = "../model_files/knn_model.pkl"
svc_path = "../model_files/svc_model.pkl"
rf_path  = "../model_files/rf_model.pkl"
scaler_path = "../model_files/scaler.pkl"

# diabetes.csv ise bu dosyanın bulunduğu (project_files) klasörde.
csv_path = "diabetes.csv"

# Oluşturulan scatter plot’u kaydedeceğimiz yer: ../static/scatter_plot.png
plot_path = "../static/scatter_plot.png"

# ------------------------------------------------------------------------
# Modelleri ve scaler'ı yükleme
# ------------------------------------------------------------------------
with open(knn_path, "rb") as file:
    knn_model = pickle.load(file)

with open(svc_path, "rb") as file:
    svc_model = pickle.load(file)

with open(rf_path, "rb") as file:
    rf_model = pickle.load(file)

with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

# ------------------------------------------------------------------------
# Veri setini yükleme (scatter plot için gerekli)
# ------------------------------------------------------------------------
dataset = pd.read_csv(csv_path)

# Veri düzenleme (scatter plot için preprocessing)
sifir = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"]
for sutun in sifir:
    dataset[sutun] = dataset[sutun].replace(0, np.nan)
    mean = dataset[sutun].mean(skipna=True)
    dataset[sutun] = dataset[sutun].fillna(mean)

# ------------------------------------------------------------------------
# Modelleri bir sözlükte topluyoruz
# ------------------------------------------------------------------------
models = {
    "KNN": knn_model,
    "SVC": svc_model,
    "Random Forest": rf_model
}

# ------------------------------------------------------------------------
# Anasayfa route'u (index.html)
# ------------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Formdan gelen verileri al
            pregnancies = float(request.form['Pregnancies'])
            glucose = float(request.form['Glucose'])
            blood_pressure = float(request.form['BloodPressure'])
            skin_thickness = float(request.form['SkinThickness'])
            insulin = float(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            dpf = float(request.form['DiabetesPedigreeFunction'])
            age = float(request.form['Age'])
            selected_model = request.form.get('model', 'KNN')  # Varsayılan model: KNN

            # Girdi verilerini bir numpy array'e çevir
            inputs = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
            inputs_normalized = scaler.transform(inputs)

            # Seçilen model ile tahmin yap
            model = models[selected_model]
            prediction = model.predict(inputs_normalized)

            # Scatter plot için veri düzenleme
            scatter_df = dataset.copy()
            scatter_df["Outcome"] = scatter_df["Outcome"].replace({0: "Sağlıklı", 1: "Hasta"})

            # Glucose ve BMI'yi normalize et
            scatter_df.iloc[:, :-1] = scaler.transform(scatter_df.iloc[:, :-1])

            # Scatter plot çizimi
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=scatter_df,
                x="Glucose",
                y="BMI",
                hue="Outcome",
                palette={"Sağlıklı": "blue", "Hasta": "orange"},
                alpha=0.6
            )

            # Kullanıcı verisini ekleme (kırmızı nokta)
            plt.scatter(inputs_normalized[0][1], inputs_normalized[0][5],
                        color="red", s=100, label="Kullanıcı Verisi", edgecolor="black")

            # Eksen sınırlarını ayarla
            plt.xlim([
                min(scatter_df["Glucose"].min(), inputs_normalized[0][1]) - 5,
                max(scatter_df["Glucose"].max(), inputs_normalized[0][1]) + 5
            ])
            plt.ylim([
                min(scatter_df["BMI"].min(), inputs_normalized[0][5]) - 5,
                max(scatter_df["BMI"].max(), inputs_normalized[0][5]) + 5
            ])

            plt.title(f"Kullanıcının Veri Setindeki Konumu ({selected_model})")
            plt.xlabel("Glucose")
            plt.ylabel("BMI")
            plt.legend()
            plt.grid(True)

            # Grafiği kaydetme
            plt.savefig(plot_path)
            plt.close()

            # Tahmin sonucuna göre yanıtı belirle
            is_diabetic = (prediction[0] == 1)  # 1: Diyabet Pozitif, 0: Diyabet Negatif

            # Sonucu result.html'e gönder
            return render_template('result.html',
                                   is_diabetic=is_diabetic,
                                   model=selected_model,
                                   plot_path="static/scatter_plot.png")
            # DİKKAT: HTML'de resmi göstermek için genelde 'static/scatter_plot.png'
            # gibi bir URL kullanılır. O nedenle return'de "../" kullanmıyoruz.

        except Exception as e:
            # Hata durumunda kullanıcıya geri dön
            return render_template('index.html', error=f"Girdi hatası: {str(e)}")

    return render_template('index.html')

# ------------------------------------------------------------------------
# Uygulama başlatma
# ------------------------------------------------------------------------
if __name__ == '__main__':
    # Flask varsayılan olarak 5000 portunu kullanır
    app.run(debug=True)

