import os
import multiprocessing
import uvicorn
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
from huggingface_hub import hf_hub_download

print("Downloading/Verifying Model from HuggingFace...")
model_path = hf_hub_download(
    repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
    filename="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
)

# Grab the number of CPU cores available on Railway, minus 1 for system stability
threads = max(1, multiprocessing.cpu_count() - 1)
print(f"Starting model with {threads} CPU threads...")

settings = Settings(
    model=model_path,
    n_ctx=4096,       
    n_threads=threads, # <--- FORCES MAX CPU USAGE
    host="0.0.0.0",
    port=int(os.getenv("PORT", 8080))
)

app = create_app(settings=settings)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port)