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
    # n_ctx=2048: The "Memory Limit" (in tokens)
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=True)
    print("--- STARTUP: Model Loaded Successfully ---")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model: {e}")
    llm = None

class Message(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.get("/")
def home():
    return {"status": "LLM API is running", "mode": "Raw Completion / Memory"}

@app.post("/chat")
def generate_chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # 1. Build Raw Context
    # We simply join all previous texts with a newline. 
    # The model sees this as one continuous document.
    raw_prompt = ""
    
    # We limit to last 15 messages to keep it fresh, 
    # but you can increase this up to the context limit.
    for msg in request.messages[-15:]:
        # We add a newline to separate inputs, treating it like a list or log
        raw_prompt += f"{msg.text}\n"

    print(f"--- RAW INPUT TO MODEL ---\n{raw_prompt}\n--------------------------")

    try:
        output = llm(
            raw_prompt, 
            max_tokens=64,  # Generate a short burst of continuation
            # We REMOVED specific stop tokens like "User:" 
            # It will stop when it finishes a thought or hits max_tokens
            stop=[], 
            echo=False, 
            temperature=0.8 # Slightly higher creativity
        )
        
        response_text = output['choices'][0]['text']
        return {"response": response_text.strip()}
    except Exception as e:
        print(f"Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))