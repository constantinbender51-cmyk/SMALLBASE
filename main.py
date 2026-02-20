import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ctransformers import AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS (allows your mobile app to talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for the model
# We use TinyLlama 1.1B Chat (Quantized) because it fits in low RAM environments
MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

print("Loading Model... this may take a moment on first startup.")
try:
    # gpu_layers=0 forces CPU usage. context_length limits memory usage.
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        model_file=MODEL_FILE,
        model_type="llama",
        gpu_layers=0,
        context_length=2048
    )
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "LLM API is running"}

@app.post("/chat")
async def generate_chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # TinyLlama Chat Format
    formatted_prompt = f"<|system|>\nYou are a helpful assistant.\n</s>\n<|user|>\n{request.prompt}\n</s>\n<|assistant|>\n"
    
    try:
        # Run inference
        response_text = llm(formatted_prompt, max_new_tokens=256, temperature=0.7)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
