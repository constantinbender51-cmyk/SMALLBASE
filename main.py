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
# MODEL: Pythia 1.4B (GPT-NeoX Architecture)
# This is a RAW BASE MODEL. It is not tuned for chat.
MODEL_REPO = "TheBloke/pythia-1.4b-GGUF"
MODEL_FILE = "pythia-1.4b.Q4_K_M.gguf"

print("--- STARTUP: Loading Pythia Base Model ---")
try:
    # model_type must be 'gpt_neox' for Pythia models
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
    return {"status": "LLM API is running", "model": "Pythia-1.4B-Base"}

@app.post("/chat")
def generate_chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model is not loaded on server.")
    
    print(f"Received prompt: {request.prompt[:50]}...")

    # --- PROMPT ENGINEERING FOR BASE MODELS ---
    # Base models just complete text. They don't know they are assistants.
    # We must structure the prompt to look like a script of a conversation.
    # If we don't do this, the model might just continue your sentence instead of answering.
    
    formatted_prompt = (
        "The following is a conversation between a human and an AI.\n"
        f"Human: {request.prompt}\n"
        "AI:"
    )
    
    try:
        # stop=["Human:"] prevents the AI from generating the Human's next turn
        response_text = llm(
            formatted_prompt, 
            max_new_tokens=128, 
            temperature=0.7,
            repetition_penalty=1.1,
            stop=["Human:", "\nHuman"] 
        )
        
        # Clean up response (sometimes base models add extra newlines)
        clean_response = response_text.strip()
        
        print("Generation complete.")
        return {"response": clean_response}
    except Exception as e:
        print(f"Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)