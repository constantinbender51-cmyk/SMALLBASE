import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# We use Llama from llama_cpp because it handles new GGUF files better than ctransformers
from llama_cpp import Llama 
from huggingface_hub import hf_hub_download
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
# MODEL: Pythia 1B (Base Model)
# REPO: tensorblock/pythia-1b-GGUF
# FILE: Q4_K_M (Standard, balanced quantization)
REPO_ID = "tensorblock/pythia-1b-GGUF"
FILENAME = "pythia-1b-Q4_K_M.gguf"

print(f"--- STARTUP: Downloading {REPO_ID} ---")
try:
    # 1. Download the specific file using HuggingFace Hub
    # This caches the model so it doesn't re-download every restart
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    print(f"Model downloaded to: {model_path}")

    # 2. Load the model using llama-cpp-python
    # n_ctx=1024 is standard for Pythia 1B
    # verbose=True helps debug if it gets stuck
    llm = Llama(model_path=model_path, n_ctx=1024, verbose=True)
    
    print("--- STARTUP: Model Loaded Successfully ---")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model: {e}")
    llm = None

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "LLM API is running", "model": "Pythia-1B-TensorBlock"}

@app.post("/chat")
def generate_chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model is not loaded on server.")
    
    print(f"Received prompt: {request.prompt[:50]}...")

    # --- PROMPT STRATEGY FOR 1B BASE MODEL ---
    # Pythia 1B is a raw text predictor. 
    # We must format the prompt like a script so it knows to answer.
    
    formatted_prompt = (
        "The following is a conversation with an AI assistant.\n"
        f"User: {request.prompt}\n"
        "AI:"
    )

    try:
        # llama-cpp-python syntax
        output = llm(
            formatted_prompt, 
            max_tokens=128, 
            stop=["User:", "\nUser"], # Stop generating when it's the user's turn
            echo=False, # Return only the generated answer, not the prompt
            temperature=0.7
        )
        
        # Extract text from the response dictionary
        response_text = output['choices'][0]['text']
        
        print("Generation complete.")
        return {"response": response_text.strip()}
    except Exception as e:
        print(f"Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)