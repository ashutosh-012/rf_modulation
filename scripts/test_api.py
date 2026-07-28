import subprocess
import time
import requests
import sys

def test_api():
    print("Starting FastAPI server in the background...")
    server = subprocess.Popen([sys.executable, "src/deployment/server.py"])
    
    time.sleep(5)
    
    try:
        print("\n--- Testing Health Check ---")
        res = requests.get("http://localhost:8000/health")
        print(f"Status Code: {res.status_code}")
        print(f"Response: {res.json()}")
        
        print("\n--- Testing Inference Prediction ---")
        
        fake_signal = [0.0] * 256
        res = requests.post(
            "http://localhost:8000/predict", 
            json={"iq_data": fake_signal}
        )
        print(f"Status Code: {res.status_code}")
        print(f"Response: {res.json()}")
        
    except Exception as e:
        print(f"API test failed: {e}")
    finally:
        print("\nTerminating server...")
        server.terminate()

if __name__ == "__main__":
    test_api()
