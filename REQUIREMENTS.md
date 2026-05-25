# Requirements

Two environments are involved: the **architect app** (which you run to generate
blueprints) and the **training environment** (where you actually run the
generated configs against a GPU). They are intentionally separated so you can
run the architect on a laptop and the training job on a workstation, server,
or rented box.

---

## 1. System requirements (architect app)

| Item               | Minimum                          | Recommended                       |
| ------------------ | -------------------------------- | --------------------------------- |
| Python             | 3.10                             | 3.11 or 3.12                      |
| OS                 | Windows 10, macOS 11, Ubuntu 20  | Windows 11, macOS 13+, Ubuntu 22+ |
| RAM                | 1 GB free                        | 4 GB free                         |
| Disk               | 200 MB for `.venv` + deps        | 500 MB                            |
| Browser            | Any modern Chromium / Firefox    | Same                              |
| Network            | None required (offline-capable)  | Outbound HTTPS only for pip       |

The app itself is pure Python + Streamlit. It does **not** need a GPU to run —
the GPU you target lives in the inputs, not the host.

---

## 2. Runtime dependencies (architect app)

Pinned in [`requirements.txt`](./requirements.txt). Installed automatically by
the launcher scripts.

| Package    | Version  | Why                                                           |
| ---------- | -------- | ------------------------------------------------------------- |
| `streamlit` | ≥ 1.31.0 | UI framework                                                  |
| `pyyaml`    | ≥ 6.0    | Emits the Axolotl training config                             |

That's it. No PyTorch, no transformers, no CUDA — the architect is a generator,
not a trainer.

---

## 3. Optional dependencies (architect app)

These improve auto-detection. The app degrades gracefully if they're missing.

| Package   | Provides                                                       | If missing                                                |
| --------- | -------------------------------------------------------------- | --------------------------------------------------------- |
| `torch`   | First-choice GPU detection via `torch.cuda.get_device_properties` | Falls back to `nvidia-smi` parsing                        |
| `psutil`  | System RAM readout in the sidebar                              | RAM field shows `0.0 GB`; doesn't affect blueprint math   |
| `nvidia-smi` (binary in PATH) | Fallback GPU detection                          | Falls back to manual GPU dropdown                         |

Install any/all of these globally or into the architect venv if you want
auto-detection to "just work":

```bash
pip install torch psutil
```

---

## 4. Training environment dependencies

These are what you install **on the training machine** — not in the architect's
venv — to actually run the generated configs. The architect prints the full
install command in the *Section 4 — Launch Commands* panel of the UI; this
table explains why each is needed.

### Common to both paths (HF Trainer and Axolotl)

| Package         | Version (tested)     | Why                                                       |
| --------------- | -------------------- | --------------------------------------------------------- |
| `torch`         | ≥ 2.1 (CUDA build)   | Tensor backbone                                           |
| `transformers`  | ≥ 4.40               | Model + tokenizer loading, `Trainer`                      |
| `datasets`      | ≥ 2.18               | Loads + caches the tokenized dataset                      |
| `accelerate`    | ≥ 0.28               | Distributed launch + mixed precision                      |
| `sentencepiece` | ≥ 0.1.99             | Required by many tokenizers (Llama, Mistral, Gemma)       |

### For LoRA / QLoRA paths

| Package         | Version (tested)     | Why                                                       |
| --------------- | -------------------- | --------------------------------------------------------- |
| `peft`          | ≥ 0.10               | LoRA adapter wrapper                                      |
| `bitsandbytes`  | ≥ 0.43 (Linux/WSL2)  | 4-bit NF4 quant + 8-bit Adam optimizer                    |

> **Note on `bitsandbytes` on Windows.** Pre-built wheels are Linux-only. On
> native Windows use WSL2 with CUDA, or build from source. Most Windows users
> just run the training step inside WSL2 even if they generate the blueprint
> in native Windows.

### For FlashAttention-2

| Package         | Version (tested)     | Why                                                       |
| --------------- | -------------------- | --------------------------------------------------------- |
| `flash-attn`    | ≥ 2.5                | Removes O(seq²) attention memory; required by long ctx    |

> Requires CUDA ≥ 11.7 and an Ampere-or-newer GPU (compute capability ≥ 8.0).
> The architect auto-disables this in the UI and the config if your GPU is
> older. Install with `pip install flash-attn --no-build-isolation`.

### If you chose the **Axolotl** path

| Package         | Version (tested)     | Why                                                       |
| --------------- | -------------------- | --------------------------------------------------------- |
| `axolotl`       | ≥ 0.4.1              | Wraps all of the above with one YAML config               |

Install:
```bash
pip install -e 'git+https://github.com/OpenAccess-AI-Collective/axolotl.git@main#egg=axolotl[flash-attn,deepspeed]'
```

### If you chose the **Hugging Face** path

Use the `pip install transformers datasets accelerate peft bitsandbytes
sentencepiece flash-attn` line printed in the UI.

---

## 5. GPU support matrix

The architect computes feasibility for any NVIDIA GPU with a declared VRAM
size. The table shows what extra features unlock at each compute capability
tier. AMD ROCm and Apple Silicon are **not currently modelled** — see Known
Limitations below.

| Generation   | Cards (examples)           | CC        | bf16 native | FlashAttention-2 |
| ------------ | -------------------------- | --------- | ----------- | ---------------- |
| Turing       | RTX 2080 Ti, T4            | 7.5       | no          | no (uses xformers / SDPA) |
| Ampere       | RTX 3090, A100, A6000      | 8.0 / 8.6 | yes         | yes              |
| Ada Lovelace | RTX 4090, 4080, L40S, L4   | 8.9       | yes         | yes              |
| Hopper       | H100, H200                 | 9.0       | yes         | yes              |
| Blackwell    | RTX 5090                   | 12.0      | yes         | yes              |

Curated dropdown options (all auto-populate VRAM + CC):

> RTX 5090 · RTX 4090 · 4080 Super · 4080 · 4070 Ti Super · 4070 Ti · 4070 Super
> · 4070 · 4060 Ti 16 GB · 4060 Ti · 4060 · 3090 Ti · 3090 · 3080 Ti · 3080 ·
> 3070 Ti · 3070 · 3060 12 GB · 2080 Ti · A100 80 GB · A100 40 GB · A6000 ·
> A5000 · L40S · L4 · H100 80 GB · H200

Anything not on the list: choose any card to seed defaults, then override the
**VRAM (GB)** field in the sidebar.

---

## 6. Dataset format requirements

The architect's pre-tokenization script handles five file formats:

| Format    | Recognised by                                                 | Notes                                        |
| --------- | ------------------------------------------------------------- | -------------------------------------------- |
| `jsonl`   | `load_dataset('json', data_files=...)`                        | One JSON object per line. Most common.       |
| `json`    | Same as above (array of objects)                              | Single JSON array file.                      |
| `csv`     | `load_dataset('csv', data_files=...)`                         | First row = column headers.                  |
| `parquet` | `load_dataset('parquet', data_files=...)`                     | Columnar, fastest for huge datasets.         |
| `txt`     | `load_dataset('text', data_files=...)`                        | Each line → one `text` field. For raw CPT.   |
| `hf_hub`  | Direct repo id, e.g. `tatsu-lab/alpaca`                       | Falls through to `load_dataset(repo_id)`.    |

Expected schema per prompt template:

| Template   | Expected columns                                                  |
| ---------- | ----------------------------------------------------------------- |
| `alpaca`   | `instruction`, `input` (optional), `output`                       |
| `chatml`   | `messages: [{role, content}]` *or* `conversations: [{from,value}]` |
| `llama3`   | Same as `chatml`                                                  |
| `mistral`  | Same as `chatml`; falls back to `{instruction,output}`            |
| `gemma`    | Same as `chatml`                                                  |
| `sharegpt` | `conversations: [{from, value}]` — passed through `tokenizer.apply_chat_template` |
| `llama2`   | Same as `chatml`; falls back to `{instruction,output}`            |

Sample sizes are not enforced — the script will tokenize anything from 10 rows
to tens of millions. For datasets > 100 K rows, bump the *Tokenization workers*
slider in the UI to match your CPU core count.

---

## 7. Tested model coverage

Curated entries (used for dropdown + architectural defaults):

| Family   | Models                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------- |
| Llama 3  | 3.2-1B · 3.2-3B · 3.1-8B-Instruct · 3.1-70B-Instruct                                            |
| Llama 2  | 7B · 13B                                                                                        |
| Mistral  | 7B-v0.3 · 7B-Instruct-v0.3 · Nemo-Base-2407 (12B) · Mixtral-8x7B                                |
| Qwen 2.5 | 0.5B · 1.5B · 3B · 7B · 14B · 32B · Coder-7B                                                    |
| Gemma 2  | 2B · 9B · 27B                                                                                   |
| Phi-3    | mini-4k · medium-4k · 3.5-mini (128 K)                                                          |
| DeepSeek | coder-6.7B-base · V2-Lite                                                                       |
| Yi 1.5   | 6B · 9B · 34B                                                                                   |
| Other    | TinyLlama 1.1B (for smoke testing)                                                              |

Unlisted model? Select **`[custom]`** in the Model section and supply:
parameter count (billions), hidden dim, layer count, attention-head count,
KV-head count (= heads for non-GQA), native max context, and family.

---

## 8. Known limitations

- **NVIDIA only.** AMD ROCm GPUs and Apple Silicon MPS aren't in the GPU
  database. You can still get a VRAM estimate by selecting any NVIDIA card and
  overriding the VRAM, but FA2/bf16 detection won't be meaningful.
- **Single-GPU VRAM math.** The estimator assumes one GPU. Multi-GPU (DDP /
  FSDP / DeepSpeed ZeRO-3) sharding is **not** modelled — for those, treat the
  reported total as per-replica memory and configure DeepSpeed manually.
- **MoE models.** Mixtral-8x7B and DeepSeek-V2-Lite use total (not active)
  param counts for training memory, which matches reality. Inference-time
  active-param estimates are out of scope.
- **Activation estimate is a heuristic.** Real activation memory varies with
  Flash kernel version, dropout placement, and tensor-parallel degree. The 7%
  safety pad + the 20% empirical fudge inside the estimator absorb most of
  the slop, but always leave ≥ 1 GB headroom before declaring a config safe.
- **No multi-node planning.** Single host only.
