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
# MODEL: Pythia 1B (TensorBlock)
# FILE: Q2_K (Compressed)
REPO_ID = "mav23/pythia-1.4b-GGUF"
FILENAME = "pythia-1.4b.Q5_K_M.gguf"

print(f"--- STARTUP: Downloading {FILENAME} ---")
try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    print(f"Model downloaded to: {model_path}")

    # Load Model
    llm = Llama(model_path=model_path, n_ctx=1024, verbose=True)
    print("--- STARTUP: Model Loaded Successfully ---")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model: {e}")
    llm = None

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "LLM API is running", "model": "Pythia-1B-TensorBlock"}

@app.post("/chat")
def generate_chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    print(f"Received prompt: {request.prompt[:50]}...")

    formatted_prompt = (
        "The following is a Q&A session with an AI.\n"
        f"User: {request.prompt}\n"
        "AI:"
    )

    try:
        output = llm(
            formatted_prompt, 
            max_tokens=128, 
            stop=["User:", "\nUser"], 
            echo=False, 
            temperature=0.7
        )
        
        response_text = output['choices'][0]['text']
        return {"response": response_text.strip()}
    except Exception as e:
        print(f"Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))