# File: Membangun_model/modelling_tuning.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def main():
    # 1. Inisialisasi DagsHub & MLflow Tracking (Sesuai Akun Anda)
    REPO_OWNER = "Agung-Prayuda"
    REPO_NAME = "Eksperimen_SML_Agung-Prayuda"
    
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("Student_Performance_Tuning")

    # Ubah kembali ke path relatif proyek sebelum push ke GitHub
    data_path = "namadataset_preprocessing/dataset_clean.csv"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File dataset tidak dijumpai di lokasi: {data_path}")
        
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Hyperparameter Tuning dengan GridSearchCV
    base_model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5]
    }
    
    print("[*] Memulakan Proses GridSearchCV Tuning...")
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Prediksi menggunakan model terbaik
    y_pred = best_model.predict(X_test)
    
    # Hitung Metrik
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 4. Logging Hasil Terbaik ke MLflow secara Manual (Skilled/Advance Syarat)
    with mlflow.start_run(run_name="Random_Forest_Best_Tuned"):
        # --- LOG PARAMETER TERBAIK ---
        mlflow.log_param("model_type", "RandomForestRegressor_Tuned")
        for param_name, param_val in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_val)
            
        # --- LOG METRIKS ---
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # --- ARTEFAK 1: Feature Importance Plot Model Terbaik ---
        plt.figure(figsize=(10, 6))
        feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh', color='green')
        plt.title("Top 10 Feature Importances - Tuned Model")
        plt.tight_layout()
        
        plot_path = "feature_importance_tuned.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)
        if os.path.exists(plot_path):
            os.remove(plot_path)
        
        # --- ARTEFAK 2: Residual Plot Model Terbaik ---
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, color='g')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Aktual")
        plt.ylabel("Prediksi")
        plt.title("Prediksi vs Aktual - Tuned Model")
        plt.tight_layout()
        
        res_path = "residual_plot_tuned.png"
        plt.savefig(res_path)
        plt.close()
        mlflow.log_artifact(res_path)
        if os.path.exists(res_path):
            os.remove(res_path)

        # --- ARTEFAK 3: Log File Model Mentah ---
        mlflow.sklearn.log_model(best_model, "tuned_best_model")
        
        print("[+] Tuned Model sukses disimpan ke DagsHub Online!")
        print(f"Parameter Terbaik: {best_params}")
        print(f"Metriks Terbaik -> RMSE: {rmse:.4f} | R2 Score: {r2:.4f}")

if __name__ == "__main__":
    main()