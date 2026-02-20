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
# MODEL: Pythia 31M (Micro Base)
# SOURCE: tensorblock
# SIZE: ~25MB (Tiny!)
MODEL_REPO = "tensorblock/pythia-31m-GGUF"
# Tensorblock usually names files with the quantization in the name
MODEL_FILE = "pythia-31m-Q4_K_M.gguf"

print(f"--- STARTUP: Loading {MODEL_REPO} ---")
try:
    # Pythia models use 'gpt_neox' architecture
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        model_file=MODEL_FILE,
        model_type="gpt_neox", 
        gpu_layers=0,
        context_length=512 # Reduced context for the micro model
    )
    print("--- STARTUP: Model Loaded Successfully ---")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model: {e}")
    llm = None

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "LLM API is running", "model": "Pythia-31m-Micro"}

@app.post("/chat")
def generate_chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model is not loaded on server.")
    
    print(f"Received prompt: {request.prompt[:50]}...")

    # BASE MODEL STRATEGY:
    # Just feed the prompt. The model will try to predict what comes next.
    
    try:
        response_text = llm(
            request.prompt, 
            max_new_tokens=64, # Keep generation short
            temperature=0.8,   # Slightly higher creativity for such a small model
            repetition_penalty=1.1
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