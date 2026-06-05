# File: prometheus_exporter.py
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import random
import json
from prometheus_client import (
    generate_latest, REGISTRY, Counter, Gauge, Histogram, Summary
)

# ==========================================
# DEFINISI 10 METRIKS BERBEDA (SYARAT ADVANCE)
# ==========================================
# 1. Counter: Total request masuk
REQUEST_COUNT = Counter('ml_inference_requests_total', 'Total number of model inference requests')
# 2. Counter: Total prediksi sukses
SUCCESS_COUNT = Counter('ml_inference_success_total', 'Total number of successful model predictions')
# 3. Counter: Total error
ERROR_COUNT = Counter('ml_inference_errors_total', 'Total number of model inference errors')
# 4. Gauge: Penggunaan CPU Sistem
CPU_USAGE = Gauge('ml_system_cpu_utilization', 'Current CPU utilization percentage')
# 5. Gauge: Penggunaan Memori
MEMORY_USAGE = Gauge('ml_system_memory_usage_bytes', 'Current memory usage in bytes')
# 6. Gauge: Koneksi Aktif saat ini
ACTIVE_CONNS = Gauge('ml_system_active_connections', 'Number of active connections to the model server')
# 7. Gauge: Waktu aktif sistem (Uptime)
SYSTEM_UPTIME = Gauge('ml_system_uptime_seconds', 'Total system uptime in seconds')
# 8. Histogram: Latensi Response HTTP
REQUEST_LATENCY = Histogram('ml_inference_latency_seconds', 'Inference request latency in seconds', buckets=[0.1, 0.5, 1.0, 2.0, 5.0])
# 9. Summary: Rata-rata nilai hasil prediksi (untuk deteksi data drift)
PREDICTION_MEAN = Summary('ml_prediction_value_mean', 'Summary of model prediction output values')
# 10. Gauge: Varians nilai prediksi terbaru
PREDICTION_VAR = Gauge('ml_prediction_value_variance', 'Current variance of model prediction outputs')

START_TIME = time.time()

class MetricsServer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            # Memperbarui metriks sistem secara dinamis sebelum ditarik Prometheus
            CPU_USAGE.set(random.uniform(20.0, 85.0)) # Simulasi CPU naik turun
            MEMORY_USAGE.set(random.choice([1024**3 * 1.2, 1024**3 * 1.5, 1024**3 * 1.8]))
            SYSTEM_UPTIME.set(time.time() - START_TIME)
            ACTIVE_CONNS.set(random.randint(1, 10))
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(generate_latest(REGISTRY))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == '/predict':
            start_time = time.time()
            REQUEST_COUNT.inc()
            
            # Simulasi penanganan request
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Simulasi error acak 5% untuk keperluan visualisasi grafik error di Grafana
                if random.random() < 0.05:
                    raise Exception("Model timeout or memory overflow simulation")
                
                # Simulasi hasil prediksi
                mock_prediction = random.uniform(50.0, 100.0)
                PREDICTION_MEAN.observe(mock_prediction)
                PREDICTION_VAR.set(random.uniform(5.0, 25.0))
                
                SUCCESS_COUNT.inc()
                response = {"status": "success", "prediction": mock_prediction}
                self.send_response(200)
                
            except Exception as e:
                ERROR_COUNT.inc()
                response = {"status": "error", "message": str(e)}
                self.send_response(500)
                
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
            # Catat latensi
            REQUEST_LATENCY.observe(time.time() - start_time)
        else:
            self.send_error(404, "Not Found")

def run(port=8000):
    print(f"[+] Memulai Exporter Server di port {port}...")
    print(f"[+] Akses metriks di http://localhost:{port}/metrics")
    print(f"[+] Kirim request inference ke http://localhost:{port}/predict")
    server_address = ('', port)
    httpd = HTTPServer(server_address, MetricsServer)
    httpd.serve_forever()

if __name__ == '__main__':
    run()