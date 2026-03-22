"""
data/sanskrit/prepare.py

Downloads and combines two Sanskrit corpora:
  1. wikimedia/wikipedia (sa)                  - Sanskrit Wikipedia
  2. chronbmm/sanskrit-monolingual-pretraining - classical texts in IAST, converted to Devanagari

Tokenizes with Sarvam AI's tokenizer (sarvamai/sarvam-1) and saves
train.bin / val.bin as numpy uint32 arrays in nanoGPT's expected format.

Usage:
    python data/sanskrit/prepare.py

Requirements:
    pip install datasets transformers numpy tqdm indic-transliteration
"""

import os
import re
import json
import pickle

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VAL_FRACTION = 0.005
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_ID = "sarvamai/sarvam-1"
CACHE_PATH   = os.path.join(OUTPUT_DIR, "mono_converted.jsonl")

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print(f"Loading tokenizer: {TOKENIZER_ID}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
print(f"Vocab size: {tokenizer.vocab_size}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def iast_to_devanagari(text: str) -> str:
    return transliterate(text, sanscript.IAST, sanscript.DEVANAGARI)

def clean_text(text: str, min_devanagari_ratio=0.4, min_length=20, max_length=100_000) -> str | None:
    """Returns cleaned text, or None if the text should be discarded."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if not (min_length <= len(text) <= max_length):
        return None
    # remove HTML tags and URLs
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # require minimum Devanagari content
    if not text or len(DEVANAGARI_RE.findall(text)) / len(text) < min_devanagari_ratio:
        return None
    # reject excessive character repetition
    if re.search(r'(.)\1{9,}', text):
        return None
    return text

def tokenize_to_file(texts: list[str], path: str, desc: str = "Tokenizing"):
    """Tokenize texts and write directly to a binary file in chunks."""
    CHUNK = 100_000
    buffer = []
    total = 0
    # write empty file first
    open(path, 'wb').close()
    for text in tqdm(texts, desc=desc):
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        buffer.extend(ids)
        if len(buffer) >= CHUNK:
            np.array(buffer, dtype=np.uint32).tofile(open(path, 'ab'))
            total += len(buffer)
            buffer = []
    if buffer:
        np.array(buffer, dtype=np.uint32).tofile(open(path, 'ab'))
        total += len(buffer)
    print(f"  {total:,} tokens written to {path}")
    return total

# ---------------------------------------------------------------------------
# 1. Sanskrit Wikipedia
# ---------------------------------------------------------------------------
print("\n[1/2] Loading Sanskrit Wikipedia ...")
wiki = load_dataset("wikimedia/wikipedia", "20231101.sa", split="train")
wiki_texts = [t for row in wiki if (t := clean_text(row["text"])) is not None]
print(f"  {len(wiki_texts):,} articles kept")
wiki_path = os.path.join(OUTPUT_DIR, "wiki_tokens.bin")
wiki_count = tokenize_to_file(wiki_texts, wiki_path, desc="  Tokenizing Wikipedia")

# ---------------------------------------------------------------------------
# 2. Sanskrit monolingual pretraining (IAST → Devanagari)
#    Cached to CACHE_PATH so a crash during tokenization doesn't cost
#    another 30-minute transliteration pass.
# ---------------------------------------------------------------------------
print("\n[2/2] Loading chronbmm/sanskrit-monolingual-pretraining ...")
if os.path.exists(CACHE_PATH):
    print(f"  Found cache at {CACHE_PATH}, skipping transliteration ...")
    with open(CACHE_PATH, 'r', encoding='utf-8') as f:
        mono_texts = [json.loads(line)["text"] for line in f]
else:
    mono = load_dataset("chronbmm/sanskrit-monolingual-pretraining", split="train")
    mono_texts = []
    for row in tqdm(mono, desc="  Converting IAST → Devanagari"):
        raw = row.get("text", "").strip()
        if not raw:
            continue
        cleaned = clean_text(iast_to_devanagari(raw))
        if cleaned is not None:
            mono_texts.append(cleaned)
    print(f"  Saving cache to {CACHE_PATH} ...")
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        for text in mono_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

print(f"  {len(mono_texts):,} rows")
mono_path = os.path.join(OUTPUT_DIR, "mono_tokens.bin")
mono_count = tokenize_to_file(mono_texts, mono_path, desc="  Tokenizing monolingual pretraining")

# ---------------------------------------------------------------------------
# Combine + split into train/val by concatenating the two bin files
# ---------------------------------------------------------------------------
print("\nCombining and splitting ...")
total     = wiki_count + mono_count
split_idx = int(total * (1 - VAL_FRACTION))

print(f"  Wikipedia    : {wiki_count:,}  ({100*wiki_count/total:.1f}%)")
print(f"  Monolingual  : {mono_count:,}  ({100*mono_count/total:.1f}%)")
print(f"  Total        : {total:,}")
print(f"  Train        : {split_idx:,}")
print(f"  Val          : {total - split_idx:,}")

train_path = os.path.join(OUTPUT_DIR, "train.bin")
val_path   = os.path.join(OUTPUT_DIR, "val.bin")
meta_path  = os.path.join(OUTPUT_DIR, "meta.pkl")

# stream-copy wiki + mono into train/val without loading into RAM
print("  Writing train.bin and val.bin ...")
written = 0
with open(train_path, 'wb') as train_f, open(val_path, 'wb') as val_f:
    for src_path in [wiki_path, mono_path]:
        src = np.memmap(src_path, dtype=np.uint32, mode='r')
        for start in range(0, len(src), 100_000):
            chunk = src[start:start+100_000]
            boundary = max(0, split_idx - written)
            if boundary >= len(chunk):
                train_f.write(chunk.tobytes())
            elif boundary == 0:
                val_f.write(chunk.tobytes())
            else:
                train_f.write(chunk[:boundary].tobytes())
                val_f.write(chunk[boundary:].tobytes())
            written += len(chunk)

# clean up intermediate files
os.remove(wiki_path)
os.remove(mono_path)