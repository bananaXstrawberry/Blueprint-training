"""Local Training Architect — Streamlit UI.

Run with: streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from architect import hardware as hw
from architect import models, templates
from architect.blueprint import TrainingHyperparams, build
from architect.generators.tokenizer import DatasetSpec
from architect.vram import TrainingPlan


st.set_page_config(
    page_title="Local Training Architect",
    page_icon="🛠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Local Training Architect")
st.caption(
    "Deterministic VRAM math, pre-tokenization pipelines, and ready-to-run training configs "
    "for your local hardware. No cloud, no LLM calls — pure rules-based blueprint generation."
)

# ----------------------------------------------------------------------------
# Sidebar: Hardware
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Hardware")

    if "detected" not in st.session_state:
        st.session_state.detected = hw.detect()
    detected: hw.HardwareSpec = st.session_state.detected

    if st.button("Re-detect hardware"):
        st.session_state.detected = hw.detect()
        st.rerun()

    src_label = {
        "torch.cuda": "✓ Detected via torch.cuda",
        "nvidia-smi": "✓ Detected via nvidia-smi",
        "manual": "Manual selection",
        "none": "✗ No GPU detected — pick from dropdown",
    }[detected.detection_source]
    st.caption(src_label)
    if detected.gpu_name:
        st.code(
            f"{detected.gpu_name}\n"
            f"VRAM: {detected.vram_gb} GB | "
            f"CC: {detected.compute_capability[0]}.{detected.compute_capability[1]}\n"
            f"GPUs: {detected.gpu_count} | RAM: {detected.system_ram_gb} GB | "
            f"CPU: {detected.cpu_cores} cores",
            language="text",
        )

    gpu_pick = st.selectbox(
        "Override / pick GPU",
        ["(use detected)"] + hw.gpu_choices(),
        help="Skip if auto-detection is correct. Lists curated consumer + datacenter cards.",
    )

    if gpu_pick != "(use detected)":
        hardware = hw.spec_from_choice(gpu_pick)
    else:
        hardware = detected

    vram_override = st.number_input(
        "VRAM (GB)",
        value=float(hardware.vram_gb or 24.0),
        min_value=2.0,
        max_value=200.0,
        step=1.0,
        help="Per-GPU VRAM. Override if dropdown / detection is wrong.",
    )
    hardware.vram_gb = vram_override

    for w in hardware.warnings:
        st.warning(w)

# ----------------------------------------------------------------------------
# Main: Model + Dataset + Plan
# ----------------------------------------------------------------------------
col_model, col_dataset = st.columns(2)

with col_model:
    st.header("2. Model")

    model_pick = st.selectbox(
        "Base model",
        models.list_models() + ["[custom]"],
        index=models.list_models().index("meta-llama/Llama-3.1-8B-Instruct")
        if "meta-llama/Llama-3.1-8B-Instruct" in models.list_models()
        else 0,
    )

    if model_pick == "[custom]":
        st.caption("Enter architectural details for an unlisted model.")
        custom_id = st.text_input("Repo ID", "my-org/my-model-7b")
        custom_params = st.number_input("Parameters (billion)", 0.1, 1000.0, 7.0, 0.1)
        c1, c2 = st.columns(2)
        with c1:
            custom_hidden = st.number_input("Hidden dim", 256, 16384, 4096, 128)
            custom_layers = st.number_input("Layers", 4, 200, 32, 1)
            custom_heads = st.number_input("Attention heads", 1, 256, 32, 1)
        with c2:
            custom_kv = st.number_input("KV heads (GQA)", 1, 256, 8, 1)
            custom_ctx = st.number_input("Native max context", 512, 1_000_000, 8192, 512)
            custom_family = st.selectbox(
                "Family / template default",
                ["llama3", "llama2", "mistral", "chatml", "gemma", "alpaca"],
            )
        model = models.custom(
            custom_id,
            custom_params,
            custom_hidden,
            custom_layers,
            custom_heads,
            custom_kv,
            custom_ctx,
            custom_family,
        )
    else:
        model = models.get(model_pick)
        assert model is not None
        with st.expander("Model details", expanded=False):
            st.write(
                {
                    "Parameters (B)": round(model.params / 1e9, 2),
                    "Hidden dim": model.hidden,
                    "Layers": model.layers,
                    "Attention heads": model.heads,
                    "KV heads": model.kv_heads,
                    "Native max context": model.max_context,
                    "Family": model.family,
                    "Notes": model.notes or "—",
                }
            )

with col_dataset:
    st.header("3. Dataset")

    ds_path = st.text_input(
        "Dataset path",
        value=r"./data/train.jsonl",
        help="Absolute or relative path. Can also be a HuggingFace Hub repo id.",
    )
    ds_format = st.selectbox("Format", ["jsonl", "json", "csv", "parquet", "txt", "hf_hub"])
    template_default = templates.default_for_family(model.family)
    template_name = st.selectbox(
        "Prompt template",
        templates.list_templates(),
        index=templates.list_templates().index(template_default),
        help="The template determines how rows are stitched into a single training string.",
    )
    st.caption(templates.get(template_name).description)

    seq_len = st.slider(
        "Sequence length (tokens)",
        128,
        min(model.max_context, 32768),
        min(2048, model.max_context),
        128,
        help="Longer sequences = much more activation memory. Halving roughly halves activation RAM.",
    )
    eval_pct = st.slider(
        "Eval split (%)",
        0,
        20,
        0,
        1,
        help="Fraction of dataset held out for eval. Set to 0 to disable eval.",
    )
    num_proc = st.slider(
        "Tokenization workers",
        1,
        max(hardware.cpu_cores or 8, 8),
        min(4, max(hardware.cpu_cores or 4, 1)),
        1,
    )

# ----------------------------------------------------------------------------
# Plan
# ----------------------------------------------------------------------------
st.header("4. Training plan")

plan_c1, plan_c2, plan_c3 = st.columns(3)
with plan_c1:
    method = st.radio(
        "Method",
        ["qlora", "lora", "full"],
        index=0,
        horizontal=True,
        help="QLoRA: 4-bit base, LoRA adapters. LoRA: fp16 base, LoRA adapters. Full: train all weights.",
    )
    optimizer = st.selectbox(
        "Optimizer",
        ["paged_adamw_8bit", "adamw_8bit", "adamw_torch"],
        index=0,
        help="8-bit / paged optimizers save ~6 bytes/param vs 32-bit Adam.",
    )
with plan_c2:
    target_global_batch = st.number_input(
        "Target global batch size",
        1,
        2048,
        32,
        1,
        help="Effective batch (micro × grad_accum × GPUs). 32–64 is typical for instruction tuning.",
    )
    lora_r = st.select_slider(
        "LoRA rank (r)",
        options=[4, 8, 16, 32, 64, 128, 256],
        value=16,
        help="Higher r = more capacity, more memory. 16–32 is the sweet spot for most adapter tuning.",
    )
    lora_alpha = st.number_input("LoRA alpha", 4, 1024, lora_r * 2, 4)
with plan_c3:
    grad_ckpt = st.checkbox(
        "Gradient checkpointing",
        value=True,
        help="Recompute activations during backward. ~30% slower, ~4–10× less activation memory.",
    )
    use_flash = st.checkbox(
        "FlashAttention-2",
        value=hardware.supports_flash_attention_2,
        disabled=not hardware.supports_flash_attention_2,
        help=(
            "Requires Ampere (SM 8.0+). Eliminates O(seq²) attention memory."
            if hardware.supports_flash_attention_2
            else "GPU compute capability < 8.0 — FA2 unavailable, falls back to SDPA."
        ),
    )
    use_bf16 = st.checkbox(
        "bfloat16",
        value=hardware.supports_bf16,
        disabled=not hardware.supports_bf16,
        help="bf16 preferred on Ampere+. Otherwise fp16 mixed precision.",
    )

hp_c1, hp_c2, hp_c3, hp_c4 = st.columns(4)
with hp_c1:
    learning_rate = st.number_input(
        "Learning rate",
        1e-6,
        1e-2,
        2e-4 if method != "full" else 2e-5,
        format="%.6f",
    )
with hp_c2:
    epochs = st.number_input("Epochs", 1, 50, 3, 1)
with hp_c3:
    warmup_steps = st.number_input("Warmup steps", 0, 2000, 50, 10)
with hp_c4:
    output_dir = st.text_input("Output dir", "./outputs")

# ----------------------------------------------------------------------------
# Generate
# ----------------------------------------------------------------------------
st.markdown("---")
if st.button("Generate Blueprint", type="primary", use_container_width=True):
    plan = TrainingPlan(
        method=method,
        optimizer=optimizer,
        sequence_len=seq_len,
        target_global_batch=target_global_batch,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target_count=7,
        gradient_checkpointing=grad_ckpt,
        flash_attention=use_flash,
        use_bf16=use_bf16,
    )
    dataset_spec = DatasetSpec(
        path=ds_path,
        format=ds_format,
        sequence_len=seq_len,
        output_dir="./tokenized_dataset",
        num_proc=num_proc,
        eval_split_pct=eval_pct / 100.0,
    )
    hp = TrainingHyperparams(
        learning_rate=learning_rate,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_dir=output_dir,
    )

    blueprint = build(hardware, model, plan, dataset_spec, template_name, hp)
    st.session_state.blueprint = blueprint
    st.session_state.plan = plan
    st.session_state.dataset_spec = dataset_spec
    st.session_state.model = model

# ----------------------------------------------------------------------------
# Render blueprint
# ----------------------------------------------------------------------------
if "blueprint" in st.session_state:
    bp = st.session_state.blueprint

    st.markdown("## Section 1 — Hardware Feasibility & Risk")
    risk_color = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[bp.report.risk]
    st.subheader(f"{risk_color} Risk: {bp.report.risk.upper()}")

    left, right = st.columns([2, 1])
    with left:
        st.write("**VRAM Breakdown** (GB)")
        st.table(bp.report.breakdown.as_dict())
    with right:
        st.metric("Available VRAM", f"{bp.report.available_vram_gb:.1f} GB")
        st.metric("Estimated peak", f"{bp.report.breakdown.total_gb:.1f} GB")
        st.metric("Headroom", f"{bp.report.headroom_gb:+.1f} GB")
        st.write(f"**per_device_train_batch_size:** `{bp.report.per_device_batch_size}`")
        st.write(f"**gradient_accumulation_steps:** `{bp.report.gradient_accumulation_steps}`")
        st.write(
            f"**effective global batch:** "
            f"`{bp.report.per_device_batch_size * bp.report.gradient_accumulation_steps}`"
        )

    for w in bp.report.warnings:
        st.warning(w)
    for n in bp.report.notes:
        st.info(n)

    st.markdown("## Section 2 — Pre-Tokenization Script")
    st.caption(
        "Run this ONCE before training. It applies the prompt template, tokenizes everything, "
        "and caches tensors to disk so the GPU is never CPU-bound."
    )
    st.code(bp.tokenize_script, language="python")
    st.download_button(
        "Download tokenize_dataset.py",
        bp.tokenize_script,
        file_name="tokenize_dataset.py",
        mime="text/x-python",
    )

    st.markdown("## Section 3 — Executable Configuration")
    tab_axolotl, tab_hf = st.tabs(["Axolotl YAML", "HuggingFace Trainer"])
    with tab_axolotl:
        st.caption("Drop-in config for `axolotl` (https://github.com/OpenAccess-AI-Collective/axolotl).")
        st.code(bp.axolotl_yaml, language="yaml")
        st.download_button(
            "Download training_config.yaml",
            bp.axolotl_yaml,
            file_name="training_config.yaml",
            mime="text/yaml",
        )
    with tab_hf:
        st.caption("Vanilla HuggingFace Trainer — JSON args + matching train script.")
        st.write("**training_args.json**")
        st.code(bp.hf_args_json, language="json")
        st.download_button(
            "Download training_args.json",
            bp.hf_args_json,
            file_name="training_args.json",
            mime="application/json",
        )
        st.write("**train.py**")
        st.code(bp.hf_train_script, language="python")
        st.download_button(
            "Download train.py",
            bp.hf_train_script,
            file_name="train.py",
            mime="text/x-python",
        )

    st.markdown("## Section 4 — Launch Commands")
    st.code(
        f"# 1) Install training deps (one-time)\n"
        f"pip install transformers datasets accelerate peft bitsandbytes "
        f"{'flash-attn ' if bp.report.method != 'full' or True else ''}sentencepiece\n\n"
        f"# 2) Pre-tokenize dataset\n"
        f"python tokenize_dataset.py\n\n"
        f"# 3a) Launch with Axolotl\n"
        f"{bp.launch_commands['axolotl']}\n\n"
        f"# 3b) Or launch with HuggingFace Trainer\n"
        f"python train.py",
        language="bash",
    )

else:
    st.info(
        "Configure hardware (sidebar), model, dataset, and plan above, then click "
        "**Generate Blueprint**."
    )
