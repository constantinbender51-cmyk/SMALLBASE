FROM python:3.11-slim

RUN apt-get update && apt-get install -y wget build-essential cmake gcc g++

WORKDIR /app

RUN pip install --no-cache-dir llama-cpp-python[server]

RUN wget -O model.gguf "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf?download=true"

# Force the environment variable so Railway and Uvicorn agree
ENV PORT=8000
EXPOSE 8000

# Hardcode the port to 8000 to match the EXPOSE directive
CMD ["python", "-m", "llama_cpp.server", "--host", "0.0.0.0", "--port", "8000", "--model", "model.gguf", "--n_ctx", "4096"]