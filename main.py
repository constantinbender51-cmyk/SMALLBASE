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
# MODEL: Pythia 1B
# REPO: mav23 (Community upload)
# FILE: Q4_K_M (Balanced quality and speed)
# ARCHITECTURE: GPT-NeoX
MODEL_REPO = "mav23/pythia-1b-GGUF"
MODEL_FILE = "pythia-1b.Q4_K_M.gguf"

print(f"--- STARTUP: Loading {MODEL_REPO} ---")
try:
    # Pythia is GPT-NeoX architecture
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
    return {"status": "LLM API is running", "model": "Pythia-1B-mav23"}

@app.post("/chat")
def generate_chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model is not loaded on server.")
    
    print(f"Received prompt: {request.prompt[:50]}...")

    # --- PROMPT STRATEGY FOR 1B BASE MODEL ---
    # 1B is smart enough to understand a basic script format.
    # We use User/AI format to keep it on track.
    
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
            stop=["User:", "\nUser"] # Stop the model from talking to itself
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