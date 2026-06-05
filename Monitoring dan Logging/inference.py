import requests
import time
import random

url = "http://localhost:8001/predict"

print("[+] Memulai serangan data uji (inference requests) ke Proxy Exporter...")
print("[+] Tekan Ctrl+C untuk menyudahi proses perekaman data.")

request_id = 1
try:
    while True:
        # Menyesuaikan dengan cetak biru format input dataframe milik MLflow
        payload = {
            "dataframe_records": [
                {
                    "study_time": random.randint(1, 12),
                    "failures": random.choice([0, 0, 0, 1]),
                    "absence": random.randint(0, 25),
                    "score_history": random.uniform(50.0, 98.0)
                }
            ]
        }
        
        try:
            res = requests.post(url, json=payload, timeout=4)
            print(f"[{time.strftime('%X')}] Paket #{request_id} Terkirim -> HTTP {res.status_code} | Hasil: {res.json()}")
        except Exception as e:
            print(f"[-] Koneksi terputus. Pastikan prometheus_exporter.py sudah aktif: {e}")
            
        request_id += 1
        time.sleep(random.uniform(0.3, 1.2)) # Kecepatan fluktuasi data

except KeyboardInterrupt:
    print("\n[+] Aktivitas simulasi request selesai.")