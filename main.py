import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ctransformers import AutoModelForCausalLM
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
# MODEL: Pythia 2.8B (Deduped)
# QUANTIZATION: Q2_K (Aggressive compression for low RAM usage)
# ARCHITECTURE: GPT-NeoX
MODEL_REPO = "tensorblock/pythia-2.8b-deduped-GGUF"
MODEL_FILE = "pythia-2.8b-deduped-Q2_K.gguf"

print(f"--- STARTUP: Loading {MODEL_REPO} ---")
try:
    # Pythia models use 'gpt_neox' architecture
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        model_file=MODEL_FILE,
        model_type="gpt_neox", 
        gpu_layers=0,
        context_length=1024 
    )
    print("--- STARTUP: Model Loaded Successfully ---")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model: {e}")
    llm = None

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "LLM API is running", "model": "Pythia-2.8B-Q2"}

@app.post("/chat")
def generate_chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model is not loaded on server.")
    
    print(f"Received prompt: {request.prompt[:50]}...")

    # --- PROMPT STRATEGY FOR 2.8B BASE MODEL ---
    # Since this model is smarter (2.8B), we can try to force a conversation format.
    # Otherwise, it might just ramble.
    
    formatted_prompt = (
        "The following is a conversation with an AI.\n"
        f"User: {request.prompt}\n"
        "AI:"
    )
    
    try:
        response_text = llm(
            formatted_prompt, 
            max_new_tokens=128, 
            temperature=0.7,
            repetition_penalty=1.1,
            stop=["User:", "\nUser"] # Stop it from generating the user's side
        )
        
        print("Generation complete.")
        return {"response": response_text}
    except Exception as e:
        print(f"Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)