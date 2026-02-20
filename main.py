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
MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

print("--- STARTUP: Loading Model ---")
try:
    # Reduced context_length to 1024 to save RAM and speed up processing
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        model_file=MODEL_FILE,
        model_type="llama",
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
    return {"status": "LLM API is running", "model": "TinyLlama-1.1B"}

# IMPORTANT: Removed 'async'. 
# Using 'def' instead of 'async def' runs this in a threadpool, 
# preventing the AI from blocking the server network connection.
@app.post("/chat")
def generate_chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model is not loaded on server.")
    
    print(f"Received prompt: {request.prompt[:50]}...") # Log what we received

    # Standard TinyLlama Chat format
    formatted_prompt = f"<|system|>\nYou are a helpful assistant.\n</s>\n<|user|>\n{request.prompt}\n</s>\n<|assistant|>\n"
    
    try:
        # Reduced max_new_tokens to 128 for faster responses during testing
        response_text = llm(
            formatted_prompt, 
            max_new_tokens=128, 
            temperature=0.7,
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
