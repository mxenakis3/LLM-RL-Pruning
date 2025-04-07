import subprocess
import time
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

def start_server():
    """Launch vLLM server in background"""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "deepseek-ai/deepseek-llm-7b-chat",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--api-key", "local-key"
    ]
    return subprocess.Popen(cmd)

def wait_for_server():
    """Wait until server healthcheck passes"""
    health_url = "http://localhost:8000/health"
    start_time = time.time()
    while time.time() - start_time < 300:
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            time.sleep(2)
    raise TimeoutError("Server failed to start")

def query_model(i):
    """Send sample request using OpenAI client"""
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="local-key"
    )
    response = client.chat.completions.create(
        model="deepseek-ai/deepseek-llm-7b-chat",
        messages=[{"role": "user", "content": f"Explain rules of Settlers of Catan"}]
    )
    return f'[{i}]: {response.choices[0].message.content}'

if __name__ == "__main__":
    server_process = start_server()
    try:
        wait_for_server()

        # Manual pool management without context manager
        executor = ThreadPoolExecutor(max_workers=4)
        futures = []

        # Submit all tasks concurrently
        for i in range(10):
            future = executor.submit(query_model, i)
            futures.append((i, future))  # Store with submission order

        # Wait for results sequentially in submission order
        for idx, future in futures:
            try:
                result = future.result()  # Blocks until completion
                print(f"Completed #{idx}: {result}")
            except Exception as e:
                print(f"Query {idx} failed: {str(e)}")

    finally:
        # Clean up resources
        executor.shutdown(wait=True)
        server_process.terminate()
