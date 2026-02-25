# Use a lightweight Python base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y wget build-essential cmake gcc g++

WORKDIR /app

# Install the llama-cpp-python server
RUN pip install --no-cache-dir llama-cpp-python[server]

# Download the Qwen2.5 1.5B Instruct model
RUN wget -O model.gguf "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf?download=true"

# Start the server using Railway's dynamic $PORT variable.
# We use 'sh -c' so that $PORT is evaluated correctly at runtime.
CMD ["sh", "-c", "python -m llama_cpp.server --host 0.0.0.0 --port ${PORT:-8080} --model model.gguf --n_ctx 4096"]