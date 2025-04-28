import subprocess
import time
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import freeze_support
def start_server():
    """Launch vLLM server in background"""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "deepseek-ai/deepseek-llm-7b-chat",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--api-key", "local-key",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "llama3_json",
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

def query_model(model, messages, tools=None):
    """Send sample request using OpenAI client"""
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="local-key"
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )
    return response

def query_model_ex(i, tools):
    """Send sample request using OpenAI client"""
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="local-key"
    )
    response = client.chat.completions.create(
        model="deepseek-ai/deepseek-llm-7b-chat",
        messages=[{"role": "user", "content": f"Explain rules of Settlers of Catan"}],
        tools=tools,
    )
    return f'[{i}]: {response.choices[0].message.content}'

if __name__ == "__main__":
    server_process = start_server()
    try:
        wait_for_server()

        game_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_game_rules",
                    "description": "Get official rules for a board game",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "game_name": {
                                "type": "string",
                                "description": "Name of the board game, e.g. Catan, Carcassonne"
                            }
                        },
                        "required": ["game_name"]
                    }
                }
            }
        ]

        # Manual pool management without context manager
        executor = ThreadPoolExecutor(max_workers=4)
        futures = []

        # Submit all tasks concurrently
        for i in range(10):
            if i % 2 == 0:
                future = executor.submit(query_model_ex, i=i, tools=game_tools)
            else:
                future = executor.submit(query_model_ex, i=i, tools=None)
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
