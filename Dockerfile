# Use a lightweight Python base image
FROM python:3.11-slim

# Install system dependencies needed to compile llama.cpp and download the model
RUN apt-get update && apt-get install -y wget build-essential cmake gcc g++

WORKDIR /app

# Install the llama-cpp-python server. 
# We build it for CPU usage since Railway standard tiers don't have GPUs.
RUN pip install --no-cache-dir llama-cpp-python[server]

# Download the Qwen2.5 1.5B Instruct model directly from HuggingFace
RUN wget -O model.gguf "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf?download=true"

# Expose the port the server runs on
EXPOSE 8000

# Start the OpenAI-compatible server.
# We set n_ctx to 4096 so it has a large enough memory for the chat history.
CMD ["python", "-m", "llama_cpp.server", "--host", "0.0.0.0", "--port", "8000", "--model", "model.gguf", "--n_ctx", "4096"]