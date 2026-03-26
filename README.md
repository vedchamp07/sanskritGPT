# sanskritGPT

I built this to understand how GPT-2 works - not by watching someone code it, but by reading every line of [nanoGPT](https://github.com/karpathy/nanoGPT), tracing tensor shapes on paper, and annotating each component until it clicked. Once I understood the architecture, I wanted to do something with it that hadn't been done before: train it on Sanskrit.

Sanskrit is one of the oldest documented languages in the world, with a rich corpus of classical texts — the Vedas, Upanishads, Mahabharata, Ramayana — almost none of which have ever been used to train a GPT-style model. This project is an attempt to change that.

---

## What's interesting here

**Sarvam tokenizer instead of GPT-2 BPE**
GPT-2's tokenizer was trained on English and is extremely inefficient on Devanagari — a single Sanskrit word often fragments into 10+ tokens. This project uses [`sarvamai/sarvam-1`](https://huggingface.co/sarvamai/sarvam-1), a tokenizer trained natively on all 22 scheduled Indian languages across 12 scripts, achieving 2–4x better fertility on Devanagari. Swapping the tokenizer also meant changing the vocab size from 50,257 to 68,096 — a small but meaningful architectural change.

**IAST → Devanagari transliteration pipeline**
The largest available Sanskrit corpus ([`chronbmm/sanskrit-monolingual-pretraining`](https://huggingface.co/datasets/chronbmm/sanskrit-monolingual-pretraining), 21M rows) is stored in IAST — a romanized transliteration scheme designed with a perfect one-to-one mapping to Devanagari. Rather than train on a mixed-script corpus, this project converts the entire dataset to Devanagari using `indic-transliteration` before tokenization, keeping the training distribution consistent.

**Custom data pipeline**
Combines Sanskrit Wikipedia and the monolingual pretraining corpus with cleaning filters for Devanagari ratio, fragment length, HTML artifacts, and character repetition. The transliteration output is cached to disk so the 30-minute conversion pass only runs once.

---

## Model

GPT-2 style decoder-only transformer, trained from scratch.

|                 |                       |
| --------------- | --------------------- |
| Parameters      | ~29M                  |
| Layers          | 8                     |
| Attention heads | 8                     |
| Embedding dim   | 512                   |
| Context length  | 256 tokens            |
| Vocab size      | 68,096                |
| Dropout         | 0.1                   |
| Trained on      | Kaggle P100, ~6 hours |

---

## Dataset

| Source                                      | Tokens | Notes                                      |
| ------------------------------------------- | ------ | ------------------------------------------ |
| Sanskrit Wikipedia                          | ~10M   | Clean encyclopedic prose in Devanagari     |
| `chronbmm/sanskrit-monolingual-pretraining` | ~110M  | Classical + Vedic texts, IAST → Devanagari |

~120M tokens total after cleaning and subsampling.

---

## Inference

The trained checkpoint is hosted on HuggingFace. To download it locally:

```bash
huggingface-cli download vedchamp07/sanskritGPT ckpt.pt --local-dir ./out-sanskrit
python sample.py --out_dir=out-sanskrit --start="नमस्ते"
```

Or run the web UI (requires the checkpoint to be at `out-sanskrit/ckpt.pt`):

```bash
cd backend && uvicorn app:app --host 0.0.0.0 --port 8000
# then open frontend/index.html in a browser
```

---

## Reproduce

```bash
# 1. install
pip install torch transformers datasets numpy tqdm indic-transliteration

# 2. prepare data (~35 mins, cached after first run)
python data/sanskrit/prepare.py

# 3. train
python train.py config/train_sanskrit.py

# 4. sample
python sample.py --out_dir=out-sanskrit --start="नमस्ते"
```

---

## Repo structure

```
sanskritGPT/
├── model.py                       # GPT-2 architecture
├── train.py                       # training loop
├── sample.py                      # inference
├── configurator.py                # config override utility
├── sanskrit-gpt.ipynb             # training notebook (Kaggle)
├── config/
│   ├── train_sanskrit.py          # hyperparameters for this run
│   └── train_shakespeare_char.py  # Shakespeare baseline
├── data/sanskrit/
│   └── prepare.py                 # data pipeline
├── backend/
│   ├── app.py                     # FastAPI inference server
│   └── requirements.txt
├── frontend/
│   └── index.html                 # web UI
└── ARCHITECTURE.md                # notes on how GPT-2 works
```

---

## Architecture notes

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a walkthrough of the GPT-2 implementation — causal self-attention, the residual stream, weight tying, KV cache, and the generate loop — written as I was learning it.

---

## Acknowledgements

Forked from [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (MIT License).
Tokenizer by [Sarvam AI](https://www.sarvam.ai/).
Data: [chronbmm](https://huggingface.co/chronbmm) and the Wikimedia Foundation.
