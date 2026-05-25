# Local Training Architect

A desktop web app that turns your local hardware specs, model choice, and dataset
into a runnable LLM fine-tuning blueprint — **without ever leaving your machine**.

It outputs four things:

1. **VRAM feasibility report** — exact breakdown of base / gradients / optimizer /
   activations, with a green/yellow/red risk light and auto-suggested
   `per_device_batch_size` + `gradient_accumulation_steps`.
2. **Pre-tokenization script** — applies the right prompt template (Alpaca, ChatML,
   Llama-3, Mistral, Gemma, ShareGPT, Llama-2) and caches tokenized tensors so the
   GPU never starves on CPU I/O.
3. **Executable training config** — either an **Axolotl YAML** or a **Hugging Face
   `TrainingArguments` JSON + `train.py`**, with FlashAttention-2, gradient
   checkpointing, 4-bit/8-bit quant, and 8-bit Adam selected appropriately.
4. **Launch commands** — copy-pasteable shell commands to install deps, tokenize,
   and start training.

All math is rules-based and deterministic. No LLM calls, no telemetry, no cloud.

For the full dependency / compatibility matrix, see [REQUIREMENTS.md](./REQUIREMENTS.md).

---

## Install

Requires **Python 3.10+** on Windows, macOS, or Linux.

```bash
git clone <this-repo>   # or download the folder
cd local-training-architect
```

Then on first launch the included script creates a `.venv`, installs Streamlit
and PyYAML, and opens the app:

| OS                  | Command       |
| ------------------- | ------------- |
| Windows             | `run.bat`     |
| macOS / Linux       | `./run.sh`    |
| Any (manual)        | `pip install -r requirements.txt && streamlit run app.py` |

The app opens at <http://localhost:8501>.

---

## How to use

1. **Sidebar — Hardware.** The app auto-detects your GPU via `torch.cuda` or
   `nvidia-smi` when available. Override the dropdown / VRAM field if the
   detection is wrong, or if you're planning a build on a different machine.
2. **Model.** Pick from the curated list (Llama 3.x, Mistral, Qwen 2.5, Gemma 2,
   Phi-3, DeepSeek, Yi, TinyLlama) or select `[custom]` and supply your own
   architectural details.
3. **Dataset.** Point at a local file (`jsonl`, `json`, `csv`, `parquet`, `txt`)
   or a HuggingFace Hub repo id. Pick the prompt template — the app suggests one
   based on the model's family.
4. **Plan.** Method (QLoRA / LoRA / Full), optimizer, target global batch size,
   LoRA rank, sequence length, learning rate, epochs.
5. **Click _Generate Blueprint_.** Review the VRAM math, download the four
   artifacts, and follow the launch commands.

If your configuration exceeds available VRAM, the app **denies the request**,
shows the math, and tells you the smallest downscale that fits (drop full→qlora,
halve sequence length, swap to a paged 8-bit optimizer, etc.).

---

## Worked example — RTX 4090, QLoRA, Llama-3.1-8B at 4 K context

Inputs:

| Field                 | Value                              |
| --------------------- | ---------------------------------- |
| GPU                   | RTX 4090 (24 GB, CC 8.9)           |
| Model                 | `meta-llama/Llama-3.1-8B-Instruct` |
| Method                | QLoRA                              |
| Sequence length       | 4096                               |
| Target global batch   | 32                                 |
| LoRA r / alpha        | 16 / 32                            |
| Optimizer             | `paged_adamw_8bit`                 |

The blueprint engine returns:

| Component         |   GB   |
| ----------------- | -----: |
| Base model (4-bit) |  3.79 |
| Gradients (adapters) | 0.05 |
| Optimizer states   |  0.05 |
| Activations (peak) |  9.60 |
| CUDA context       |  1.00 |
| Safety pad (7%)    |  1.68 |
| **Total**          | **16.18** |

→ **GREEN risk, +7.8 GB headroom.** Resolved `per_device_train_batch_size=4`,
`gradient_accumulation_steps=8` → effective global batch of 32. FlashAttention-2
and bfloat16 are enabled automatically (Ada Lovelace, CC ≥ 8.0).

---

## Project layout

```
local-training-architect/
├── app.py                       # Streamlit UI (entry point)
├── architect/
│   ├── hardware.py              # GPU auto-detect + curated dropdown
│   ├── models.py                # Model parameter database
│   ├── templates.py             # Prompt template library
│   ├── vram.py                  # Memory math + risk assessment + auto batching
│   ├── blueprint.py             # Top-level orchestrator
│   └── generators/
│       ├── tokenizer.py         # Generates the pre-tokenization script
│       ├── axolotl.py           # Generates Axolotl YAML
│       └── hf.py                # Generates HF TrainingArguments JSON + train.py
├── requirements.txt             # App-side runtime deps (Streamlit, PyYAML)
├── REQUIREMENTS.md              # Full dependency / compatibility matrix
├── README.md
├── run.bat                      # Windows launcher (creates venv + installs + launches)
└── run.sh                       # macOS/Linux launcher
```

---

## Troubleshooting

| Symptom                                              | Cause / fix                                                                                         |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `python: command not found` from launcher           | Install Python 3.10+ from <https://python.org> and ensure it's on PATH. On Windows use `py -3.12`.  |
| Sidebar says "No GPU detected"                      | Either CUDA isn't installed or PyTorch is missing. Pick your GPU manually from the dropdown.        |
| FlashAttention-2 checkbox is greyed out             | GPU compute capability < 8.0 (pre-Ampere). The app silently falls back to xformers / SDPA.          |
| Risk light goes RED                                 | Follow the listed downscale path — full→qlora, halve seq_len, switch to paged 8-bit optimizer.      |
| `nvidia-smi: command not found`                     | Driver not installed — detection falls back to manual dropdown, no harm.                            |
| Streamlit opens but the page is blank               | Hard-refresh (Ctrl+Shift+R). If still blank, check the terminal log for a Python traceback.         |
| Port 8501 already in use                            | `streamlit run app.py --server.port 8765` to pick a different port.                                 |

---

## What it does NOT do

- It does **not** install Torch, transformers, peft, bitsandbytes, or Axolotl
  for you. Those go into your training environment, not the architect.
  See [REQUIREMENTS.md](./REQUIREMENTS.md) for the full list.
- It does **not** run training itself — by design. The generated configs are
  yours to inspect, modify, and run when you're ready.
- It does **not** call any external service. All math is local.
