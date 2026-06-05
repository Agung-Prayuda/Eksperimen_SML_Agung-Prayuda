# File: inference.py
import requests
import time
import random

url = "http://localhost:8000/predict"

print("[+] Memulai pengiriman data uji (inference request) secara berkala...")
print("[+] Tekan Ctrl+C untuk menghentikan loop.")

request_id = 1
try:
    while True:
        # Menyiapkan payload fitur dummy performa siswa
        payload = {
            "student_id": request_id,
            "study_time": random.randint(1, 10),
            "failures": random.choice([0, 0, 0, 1]),
            "absence": random.randint(0, 20)
        }
        
        try:
            response = requests.post(url, json=payload, timeout=2)
            print(f"[{time.strftime('%X')}] Request #{request_id} dikirim -> Status: {response.status_code} | Response: {response.json()}")
        except Exception as e:
            print(f"[-] Gagal terhubung ke model server: {e}")
            
        request_id += 1
        # Jeda acak antara 0.5 hingga 1.5 detik agar grafik berfluktuasi indah di Grafana
        time.sleep(random.uniform(0.5, 1.5))

except KeyboardInterrupt:
    print("\n[+] Loop pengujian dihentikan.")