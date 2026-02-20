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
    # n_ctx=2048 is crucial here to hold the memory
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=True) 
    print("--- STARTUP: Model Loaded Successfully ---")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model: {e}")
    llm = None

class Message(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    messages: List[Message]  # Receiving the whole history

@app.get("/")
def home():
    return {"status": "LLM API is running", "model": "Pythia-1.4B-Q5-Memory"}

@app.post("/chat")
def generate_chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # 1. Build the Conversation History Prompt
    # We take the last 10 messages to ensure we don't overflow the model's memory (context window)
    recent_messages = request.messages[-10:] 
    
    formatted_prompt = "The following is a conversation between a human User and an AI Assistant.\n\n"
    
    for msg in recent_messages:
        if msg.role == "user":
            formatted_prompt += f"User: {msg.text}\n"
        else:
            formatted_prompt += f"AI: {msg.text}\n"
            
    # Add the prompt for the AI to start generating
    formatted_prompt += "AI:"

    print(f"--- PROMPT SENT TO MODEL ---\n{formatted_prompt}\n----------------------------")

    try:
        output = llm(
            formatted_prompt, 
            max_tokens=200, 
            stop=["User:", "\nUser", "User "], # Stop before hallucinating a user reply
            echo=False, 
            temperature=0.7
        )
        
        response_text = output['choices'][0]['text']
        return {"response": response_text.strip()}
    except Exception as e:
        print(f"Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))