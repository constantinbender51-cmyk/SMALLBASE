import os
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
STORY_FILE = "story.txt" # The "Database"

print(f"--- STARTUP: Downloading {FILENAME} ---")
try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=True)
    print("--- STARTUP: Model Loaded ---")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    llm = None

# Ensure story file exists on startup
if not os.path.exists(STORY_FILE):
    with open(STORY_FILE, "w", encoding="utf-8") as f:
        f.write("Once upon a time, in a land of digital dreams...")

class TextPayload(BaseModel):
    text: str

# --- ENDPOINTS ---

@app.get("/story")
def get_story():
    """Read the file and send it to the frontend"""
    try:
        with open(STORY_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        return {"text": content}
    except Exception as e:
        return {"text": "Error loading story."}

@app.post("/update")
def update_story(payload: TextPayload):
    """User manually edited the text, so we overwrite the file"""
    try:
        with open(STORY_FILE, "w", encoding="utf-8") as f:
            f.write(payload.text)
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate_story(payload: TextPayload):
    """
    1. Save user's current view (payload.text)
    2. Generate new text
    3. Append to file
    4. Return full updated story
    """
    if not llm:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # 1. Save current state first (Sync)
    with open(STORY_FILE, "w", encoding="utf-8") as f:
        f.write(payload.text)

    # 2. Prepare Prompt (Context Limit Check)
    prompt = payload.text
    if len(prompt) > 6000:
        prompt = prompt[-6000:]

    print(f"--- GENERATING... ---")

    try:
        output = llm(
            prompt, 
            max_tokens=64, 
            stop=[], 
            echo=False, 
            temperature=0.8
        )
        
        new_content = output['choices'][0]['text']
        
        # 3. Append to file
        updated_full_text = payload.text + new_content
        with open(STORY_FILE, "w", encoding="utf-8") as f:
            f.write(updated_full_text)

        # 4. Return everything so frontend is perfectly synced
        return {"text": updated_full_text}
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))