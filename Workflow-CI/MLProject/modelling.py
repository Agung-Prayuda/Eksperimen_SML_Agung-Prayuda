import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Atau ganti dengan algoritma pilihan Anda (e.g., LinearRegression)
from sklearn.metrics import mean_squared_error, r2_score

# 1. Konfigurasi Repositori DagsHub Anda
REPO_OWNER = "Agung-Prayuda"
REPO_NAME = "Eksperimen_SML_Agung-Prayuda"
EXPERIMENT_NAME = "Student_Performance_CI_Automation"

def main():
    # 2. Inisialisasi Autentikasi DagsHub secara Aman
    print("[+] Menginisialisasi koneksi pelacakan DagsHub...")
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    
    # Memaksa tracking disimpan secara lokal & remote sekaligus
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 3. Memuat Dataset (Sesuaikan dengan lokasi file Anda)
    print("[+] Memuat dataset performa siswa...")
    dataset_path = "namadataset_preprocessing/dataset_clean.csv"
    
    if not os.path.exists(dataset_path):
        # Fallback jika struktur folder bergeser saat di eksekusi di GitHub Actions
        dataset_path = "dataset_clean.csv"
        
    df = pd.read_csv(dataset_path)

    # 4. Pemisahan Fitur dan Target (Contoh generik target: 'G3' atau 'Final_Grade')
    # Ubah 'G3' di bawah sesuai dengan nama kolom target/label di dataset Anda!
    target_column = 'G3' if 'G3' in df.columns else df.columns[-1]
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Mengonversi kolom kategorikal menjadi numerik jika belum dipreprocessing
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Memulai Eksperimen Tunggal MLflow (Aman dari Nested Run Error)
    print("[+] Memulai pelatihan model dan pencatatan metrik...")
    with mlflow.start_run(run_name="CI_Automated_Retrain") as run:
        
        # Inisialisasi dan Pelatihan Model
        n_estimators = 100
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # Prediksi dan Evaluasi
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"    -> Mean Squared Error: {mse:.4f}")
        print(f"    -> R2 Score: {r2:.4f}")

        # Log Parameter dan Metrik ke DagsHub Dashboard
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # 6. JAMINAN UTAMA: Simpan Model Secara Fisik Menghasilkan Berkas 'MLmodel'
        print("[+] Menyimpan fisik model ke dalam folder artifact lokal...")
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="ci_trained_model"
        )
        
        print("[+] Seluruh proses kriteria penulisan model selesai dengan sukses!")

if __name__ == "__main__":
    main()
