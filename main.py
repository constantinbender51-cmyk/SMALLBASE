import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# We use llama_cpp because it supports TensorBlock's GGUF version
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
# MODEL: Pythia 1B
# REPO: tensorblock/pythia-1b-GGUF
# FILE: Q2_K (Maximum compression, lowest RAM usage)
REPO_ID = "tensorblock/pythia-1b-GGUF"
FILENAME = "pythia-1b-Q2_K.gguf"

print(f"--- STARTUP: Downloading {FILENAME} from {REPO_ID} ---")
try:
    # 1. Download the specific file using HuggingFace Hub
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    print(f"Model downloaded to: {model_path}")

    # 2. Load the model using llama-cpp-python
    # n_ctx=1024 is standard for Pythia
    llm = Llama(model_path=model_path, n_ctx=1024, verbose=True)
    
    print("--- STARTUP: Model Loaded Successfully ---")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model: {e}")
    llm = None

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "LLM API is running", "model": "Pythia-1B-Q2-TensorBlock"}

@app.post("/chat")
def generate_chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model is not loaded on server.")
    
    print(f"Received prompt: {request.prompt[:50]}...")

    # --- PROMPT STRATEGY ---
    # Pythia 1B is a base model. We format it as a script.
    formatted_prompt = (
        "The following is a Q&A session with an AI.\n"
        f"User: {request.prompt}\n"
        "AI:"
    )

    try:
        # llama-cpp-python generation call
        output = llm(
            formatted_prompt, 
            max_tokens=128, 
            stop=["User:", "\nUser"], # Stop generating when the AI finishes its turn
            echo=False, # Return only the new text
            temperature=0.7
        )
        
        # Parse the output
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