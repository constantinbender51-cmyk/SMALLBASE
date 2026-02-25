import os
import uvicorn
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings  # <--- CHANGED THIS LINE
from huggingface_hub import hf_hub_download

print("Downloading/Verifying Model from HuggingFace...")
# This safely downloads the GGUF file and caches it.
model_path = hf_hub_download(
    repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
    filename="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
)

# Use the unified Settings object to configure everything
settings = Settings(
    model=model_path,
    n_ctx=4096,       # 4k context window
    host="0.0.0.0",
    port=int(os.getenv("PORT", 8080)) # Dynamically catches Railway's assigned port
)

# This creates the FastAPI app and injects all the /v1/ endpoints
app = create_app(settings=settings)

if __name__ == "__main__":
    print(f"Starting server on port {settings.port}...")
    uvicorn.run(app, host="0.0.0.0", port=settings.port)