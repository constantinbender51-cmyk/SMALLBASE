from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
REPO_ID = "mav23/pythia-1.4b-GGUF"
FILENAME = "pythia-1.4b.Q5_K_M.gguf"

print(f"--- STARTUP: Downloading {FILENAME} ---")
try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=True)
    print("--- STARTUP: Model Loaded Successfully ---")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model: {e}")
    llm = None

class CompletionRequest(BaseModel):
    text: str # Just one big string

@app.post("/complete")
def complete_text(request: CompletionRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # 1. SLICING (Memory Safety)
    # The model can only read ~2048 tokens (approx 6000-8000 chars).
    # If the text is longer, we only feed it the ending to continue from.
    prompt = request.text
    if len(prompt) > 6000:
        prompt = prompt[-6000:]

    print(f"--- GENERATING FROM --- \n{prompt[-50:]}...") # Print last 50 chars for debug

    try:
        output = llm(
            prompt, 
            max_tokens=48, # Generate a sentence or two
            stop=[],       # Don't stop, just flow
            echo=False,    # Only return the NEW text
            temperature=0.8
        )
        
        new_text = output['choices'][0]['text']
        return {"new_text": new_text}
    except Exception as e:
        print(f"Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
