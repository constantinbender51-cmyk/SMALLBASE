from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
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
    
    # CRITICAL: Set context to 1024 to prevent RAM Crash.
    # If this still crashes, lower it to 512.
    llm = Llama(model_path=model_path, n_ctx=1024, verbose=True)
    print("--- STARTUP: Model Loaded Successfully ---")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model: {e}")
    llm = None

class Message(BaseModel):
    role: str # We receive this but we will IGNORE it
    text: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/chat")
def generate_chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # 1. FLATTEN HISTORY
    # Combine all previous inputs and outputs into one raw string.
    # We use a space separator so sentences flow together.
    # If you prefer line-by-line (like code or poems), change " " to "\n".
    full_context = ""
    for msg in request.messages:
        full_context += msg.text + " "

    # 2. SLIDING WINDOW (Prevents Crash)
    # Pythia has a limit. We must cut the beginning of the text
    # if it gets too long, or the server will crash/error.
    # We keep the last 3000 characters (approx 750 tokens).
    if len(full_context) > 3000:
        full_context = full_context[-3000:]

    print(f"--- CONTEXT INPUT ({len(full_context)} chars) ---\n{full_context}\n--------------------------")

    try:
        output = llm(
            full_context, 
            max_tokens=64, # Generate a short burst
            stop=[],       # No stop tokens. Just generate until max_tokens.
            echo=False, 
            temperature=0.8
        )
        
        response_text = output['choices'][0]['text']
        return {"response": response_text} # Return raw text chunk
    except Exception as e:
        print(f"Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))