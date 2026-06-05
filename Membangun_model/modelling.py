# File: Membangun_model/modelling.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def main():
    # 1. Inisialisasi DagsHub & MLflow Tracking (Sesuai Akun Anda)
    REPO_OWNER = "Agung-Prayuda" 
    REPO_NAME = "Eksperimen_SML_Agung-Prayuda"
    
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("Student_Performance_Baseline")

    # Ubah kembali ke path relatif proyek sebelum push ke GitHub
    data_path = "namadataset_preprocessing/dataset_clean.csv"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File dataset tidak dijumpai di lokasi: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Pisahkan Fitur dan Target (Kolom terakhir otomatis dianggap sebagai target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Menjalankan MLflow Run dengan Manual Logging
    with mlflow.start_run(run_name="Random_Forest_Baseline"):
        # Definisi Hyperparameter Baseline
        n_estimators = 100
        max_depth = 10
        
        # Inisialisasi Model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Evaluasi Metrik
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # --- ADVANCE: MANUAL LOGGING PARAMETER & METRIKS ---
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # --- ADVANCE: ARTEFAK TAMBAHAN 1 (Feature Importance Plot) ---
        plt.figure(figsize=(10, 6))
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title("Top 10 Feature Importances - Baseline")
        plt.tight_layout()
        
        plot_path = "feature_importance_baseline.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)
        if os.path.exists(plot_path):
            os.remove(plot_path)
        
        # --- ADVANCE: ARTEFAK TAMBAHAN 2 (Residual Plot) ---
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, color='b')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Aktual")
        plt.ylabel("Prediksi")
        plt.title("Prediksi vs Aktual - Baseline")
        plt.tight_layout()
        
        res_path = "residual_plot_baseline.png"
        plt.savefig(res_path)
        plt.close()
        mlflow.log_artifact(res_path)
        if os.path.exists(res_path):
            os.remove(res_path)

        # --- ADVANCE: ARTEFAK TAMBAHAN 3 (Log Model Mentah) ---
        mlflow.sklearn.log_model(model, "baseline_model")
        
        print("[-] Baseline Model sukses dilatih dan dihantar ke DagsHub Online!")
        print(f"Metriks -> RMSE: {rmse:.4f} | R2 Score: {r2:.4f}")

if __name__ == "__main__":
    main()