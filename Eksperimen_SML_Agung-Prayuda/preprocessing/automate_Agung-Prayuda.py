# File: preprocessing/automate_Agung-Prayuda.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_automated_preprocessing(input_path, output_path):
    """
    Fungsi otomatisasi data preprocessing SML untuk Student Performance Dataset.
    Mengubah data mentah menjadi data siap latih (fully scaled & encoded).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset mentah tidak ditemukan di lokasi: {input_path}")
        
    print(f"[*] Memulai Otomatisasi Pipeline Preprocessing...")
    print(f"[1/5] Membaca data dari {input_path}...")
    
    # -------------------------------------------------------------------------
    # STEP 1: PEMBACAAN DATA DENGAN AUTO-SEPARATOR DETECTOR
    # -------------------------------------------------------------------------
    df = None
    for sep_try in [';', ',', '\t', '|']:
        try:
            df_test = pd.read_csv(input_path, sep=sep_try, quotechar='"')
            if df_test.shape[1] > 1:
                df = df_test
                break
        except Exception:
            continue
            
    if df is None:
        df = pd.read_csv(input_path)
    
    # Bersihkan nama kolom dari spasi atau tanda petik liar
    df.columns = df.columns.str.replace('"', '').str.replace("'", "").str.strip()
    
    # -------------------------------------------------------------------------
    # STEP 2: PENANGANAN MISSING VALUES & DUPLIKAT
    # -------------------------------------------------------------------------
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    print("[2/5] Penanganan missing values dan duplikasi selesai.")
            
    # -------------------------------------------------------------------------
    # STEP 3: LABEL ENCODING LENGKAP (AGRESIF)
    # -------------------------------------------------------------------------
    # Bersihkan isi sel string dari tanda petik yang mengunci tipe data
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('"', '').str.replace("'", "").str.strip()

    # Deteksi semua kolom non-numerik
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or not np.issubdtype(df[col].dtype, np.number):
            categorical_cols.append(col)
            
    # Amankan kolom target (paling kanan) agar tidak masuk fitur scaling
    target_col = df.columns[-1]
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
        
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    print(f"[3/5] Transformasi selesai. Berhasil melakukan Label Encoding pada {len(categorical_cols)} kolom kategorikal.")
        
    # -------------------------------------------------------------------------
    # STEP 4: PEMISAHAN FITUR & TARGET DAN STANDARD SCALER
    # -------------------------------------------------------------------------
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    if X.shape[1] == 0:
        raise ValueError("Eror Kritis: Matriks X kosong (0 kolom). Ekstraksi fitur gagal.")
        
    # Memastikan fitur numerik tulen sebelum masuk StandardScaler
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    df_final = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Cek tipe target
    if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
        df_final['target'] = le.fit_transform(y.astype(str))
    else:
        df_final['target'] = y.values
        
    print("[4/5] Standardisasi fitur menggunakan StandardScaler berhasil diterapkan.")
    
    # -------------------------------------------------------------------------
    # STEP 5: EKSPOR HASIL PREPROCESSING
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"[5/5] Pipeline Sukses! File siap latih disimpan di: {output_path}")
    print(f"[+] Dimensi Akhir Data Bersih: {df_final.shape[0]} baris, {df_final.shape[1]} kolom.")
    
    return df_final

if __name__ == "__main__":
    # Menggunakan path relatif yang aman dari root folder proyek Windows & Linux (GitHub)
    INPUT_FILE = "student_performance_raw/student_data.csv"
    OUTPUT_FILE = "preprocessing/student_performance_preprocessing/dataset_clean.csv"
    
    try:
        run_automated_preprocessing(INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        print(f"[-] Terjadi kesalahan fatal pada pipeline: {e}")