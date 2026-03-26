"""backend/app.py

FastAPI inference server for sanskritGPT.
Loads checkpoint once at startup, serves generation requests.

Usage:
    pip install fastapi uvicorn transformers torch
    uvicorn app:app --host 0.0.0.0 --port 8000

Environment variables:
    CHECKPOINT_PATH  path to ckpt.pt  (default: ../out-sanskrit/ckpt.pt)
    DEVICE           cuda or cpu      (default: auto)
"""

import os
import sys
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "../out-sanskrit/ckpt.pt")
DEVICE          = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_ID    = "sarvamai/sarvam-1"
MAX_TOKENS_CAP  = 500

model     = None
tokenizer = None
ready     = False

# ---------------------------------------------------------------------------
# Lifespan: load model + tokenizer at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, ready

    print(f"Loading checkpoint from {CHECKPOINT_PATH} on {DEVICE} ...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    gptconf    = GPTConfig(**checkpoint['model_args'])
    model      = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print(f"Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")

    print(f"Loading tokenizer: {TOKENIZER_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    ready = True
    print("Ready.")
    yield

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="sanskritGPT", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 150
    temperature: float = 0.8
    top_k: int = 100

class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    full_text: str
    tokens_generated: int

@app.get("/")
def root():
    if not ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "model": "sanskritGPT", "device": DEVICE}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not ready:
        raise HTTPException(status_code=503, detail="Model is still loading, please retry in a moment")
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    max_tokens = min(req.max_new_tokens, MAX_TOKENS_CAP)
    input_ids  = tokenizer.encode(req.prompt, add_special_tokens=False)
    x          = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)[None, ...]

    with torch.no_grad():
        y = model.generate(x, max_tokens, temperature=req.temperature, top_k=req.top_k)

    all_ids    = y[0].tolist()
    new_ids    = all_ids[len(input_ids):]
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    full_text  = tokenizer.decode(all_ids, skip_special_tokens=True)

    return GenerateResponse(
        prompt=req.prompt,
        completion=completion,
        full_text=full_text,
        tokens_generated=len(new_ids),
    )

@app.get("/presets")
def presets():
    return {"presets": [
        {"label": "Rigveda",       "text": "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्"},
        {"label": "Bhagavad Gita", "text": "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः"},
        {"label": "Upanishad",     "text": "अहं ब्रह्मास्मि"},
        {"label": "Ramayana",      "text": "रामो विग्रहवान् धर्मः साधुः सत्यपराक्रमः"},
    ]}
