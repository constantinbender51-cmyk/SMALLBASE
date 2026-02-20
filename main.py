from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List  # <--- Make sure this is imported
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
    # n_ctx=2048 gives the model memory space
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=True)
    print("--- STARTUP: Model Loaded Successfully ---")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model: {e}")
    llm = None

# --- DATA MODELS ---
class Message(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    messages: List[Message] # The backend now expects a LIST, not a string

@app.post("/chat")
def generate_chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Take the last 6 messages to prevent memory overflow
    recent_messages = request.messages[-6:] 
    
    # Construct the script for the Base Model
    formatted_prompt = "The following is a conversation between a human User and an AI Assistant.\n\n"
    
    for msg in recent_messages:
        label = "User" if msg.role == "user" else "AI"
        formatted_prompt += f"{label}: {msg.text}\n"
            
    formatted_prompt += "AI:"

    print(f"--- PROMPT ---\n{formatted_prompt}")

    try:
        output = llm(
            formatted_prompt, 
            max_tokens=200, 
            stop=["User:", "\nUser"], 
            echo=False, 
            temperature=0.7
        )
        
        response_text = output['choices'][0]['text']
        return {"response": response_text.strip()}
    except Exception as e:
        print(f"Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
