# File: Workflow-CI/MLProject/modelling.py
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
    # 1. Inisialisasi Tracking ke DagsHub Online secara Otomatis
    REPO_OWNER = "Agung-Prayuda" 
    REPO_NAME = "Eksperimen_SML_Agung-Prayuda"
    
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("Student_Performance_CI_Automation")

    # 2. Path Relatif Dataset Bersih di dalam MLProject
    data_path = "namadataset_preprocessing/dataset_clean.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {data_path}")
        
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Eksekusi Model Training & Logging Manual (Advance)
    with mlflow.start_run(run_name="CI_Automated_Retrain"):
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Log parameter & metrik secara manual
        mlflow.log_param("model_type", "RandomForest_CI")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        # Simpan Artefak Grafik Pentingnya Fitur
        plt.figure(figsize=(10, 6))
        pd.Series(model.feature_importances_, index=X.columns).nlargest(10).plot(kind='barh')
        plt.title("Feature Importance - CI Retrain")
        plt.tight_layout()
        plt.savefig("feature_importance_ci.png")
        plt.close()
        
        mlflow.log_artifact("feature_importance_ci.png")
        if os.path.exists("feature_importance_ci.png"):
            os.remove("feature_importance_ci.png")

        # Registrasi Model Resmi ke MLflow
        mlflow.sklearn.log_model(model, "ci_trained_model")
        
        print("[+] Re-training Sukses! Model dan artefak dikirim ke DagsHub.")

if __name__ == "__main__":
    main()