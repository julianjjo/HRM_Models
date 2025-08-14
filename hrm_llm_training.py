"""
HRM-Text1 Training Script

Inspiration taken from [SofiTesfay2010's script](https://colab.research.google.com/drive/1xZNYC-yhwdJxzbpwRekE_rDjTki5CvEv?usp=sharing)
"""

import os, shutil, pathlib, random, json, datetime, math
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset
from transformers import T5Tokenizer
from tqdm.auto import tqdm

from huggingface_hub import HfApi, HfFolder, hf_hub_download

# ----------------------------
# Training Parameters
HF_REPO_ID = "qingy2024/HRM-Text1"
SEED = 42
NUM_EPOCHS = 2
BLOCK_SIZE = 512
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4 # Effective batch size = 8 * 4 = 32
LEARNING_RATE_MAX = 2e-4
LEARNING_RATE_MIN = 1e-6
WEIGHT_DECAY = 0.01
MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 2 # Stop if validation loss doesn't improve for 2 epochs

SAVE_STEPS = 500  # Save a checkpoint every 500 global steps

# HRM Model Hyperparameters
MODEL_CONFIG = {"n_embd": 512, "n_head": 8, "n_inner": 2048, "dropout": 0.1}
MAX_HALT_STEPS = 8
PONDER_WEIGHT = 1e-2
PONDER_WEIGHT_DECAY = 0.98 # Decay ponder weight each epoch to focus on LM loss later
HALT_BIAS_INIT = -2.2 # Halt bias encourages more steps early on (sigmoid(-2.2) approx 0.1)

T5_TOKENIZER_REPO = "t5-small"
LOCAL_CHECKPOINT_PATH = "local_training_state.pt"
LOCAL_WEIGHTS_PATH = "pytorch_model.bin"
BEST_MODEL_PATH = "best_model.bin"

WANDB_PROJECT = "HRM-Text1"
UPDATE_README = True

# ----------------------------
# Selección del campo para entrenamiento
# Opciones: "conversations" (por defecto), "input_answer"
TRAIN_FIELD_MODE = "conversations"

# ----------------------------
# Utilities & Initialization
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# HuggingFace & W&B Authentication
try:
    HF_TOKEN = os.environ['HF_TOKEN']
    WANDB_API_KEY = os.environ['WANDB_API_KEY']

    print("Hugging Face & W&B tokens loaded.")
    HfFolder.save_token(HF_TOKEN)

    print("Login to Hugging Face Hub and W&B successful.")
except Exception as e:
    print("HF_TOKEN or WANDB_API_KEY secret not found.")
    HF_TOKEN, WANDB_API_KEY = None, None

# ----------------------------
# Tokenizer
print("Loading tokenizer (T5 slow)...")
os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"
tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_REPO, use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.padding_side = "left" # Important for causal modeling during generation
print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}; eos={tokenizer.eos_token}; pad={tokenizer.pad_token}")

# Import model classes
from hrm_text1_modeling import HRMText1

# -----------------------------------
# Data Loading and Preprocessing
print("Cargando y preparando dataset sanjay920/goat-sharegpt...")

# Cargar los splits del dataset Goat-ShareGPT
raw_datasets = load_dataset(
    "sanjay920/goat-sharegpt",
    split=None
)

def tokenize_function(examples):
    """
    Preprocesa los datos según TRAIN_FIELD_MODE:
    - "conversations": concatena los mensajes como antes.
    - "input_answer": concatena 'input' y 'answer' como texto de entrada.
    """
    texts = []
    if TRAIN_FIELD_MODE == "conversations":
        for conv_list in examples["conversations"]:
            # Concatenar los mensajes usando 'value' si existe, si no 'content'
            msg_texts = []
            for msg in conv_list:
                if "value" in msg:
                    msg_texts.append(str(msg["value"]))
                elif "content" in msg:
                    msg_texts.append(str(msg["content"]))
            full_text = " ".join(msg_texts) + tokenizer.eos_token
            texts.append(full_text)
    elif TRAIN_FIELD_MODE == "input_answer":
        # Procesa cada ejemplo concatenando 'input' y 'answer'
        inputs = examples.get("input", [])
        answers = examples.get("answer", [])
        for inp, ans in zip(inputs, answers):
            full_text = f"{str(inp)} {str(ans)}{tokenizer.eos_token}"
            texts.append(full_text)
    else:
        raise ValueError(f"Modo de campo de entrenamiento no soportado: {TRAIN_FIELD_MODE}")
    return tokenizer(
        texts,
        truncation=True,
        max_length=BLOCK_SIZE,
        padding="max_length",
        add_special_tokens=False, # EOS se añade manualmente
    )

# Tokenizar los splits relevantes ('train' y 'validation')
tokenized_splits = {}

if "validation" in raw_datasets:
    # Si existe el split validation, tokenizar normalmente
    for split_name in ["train", "validation"]:
        if split_name in raw_datasets:
            tokenized_splits[split_name] = raw_datasets[split_name].map(
                tokenize_function,
                batched=True,
                num_proc=os.cpu_count(),
                remove_columns=raw_datasets[split_name].column_names,
            )
            tokenized_splits[split_name].set_format("torch")
    train_loader = DataLoader(tokenized_splits["train"], batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(tokenized_splits["validation"], batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
else:
    # Si no existe validation, crear partición del split train
    print("No se encontró split 'validation'. Creando partición de validación (10%) desde 'train'...")
    train_dataset = raw_datasets["train"]
    num_samples = len(train_dataset)
    split_idx = int(num_samples * 0.9)
    train_split = train_dataset.select(range(0, split_idx))
    val_split = train_dataset.select(range(split_idx, num_samples))

    tokenized_splits["train"] = train_split.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=train_split.column_names,
    )
    tokenized_splits["train"].set_format("torch")

    tokenized_splits["validation"] = val_split.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=val_split.column_names,
    )
    tokenized_splits["validation"].set_format("torch")

    train_loader = DataLoader(tokenized_splits["train"], batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(tokenized_splits["validation"], batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# --------------------------------
# Model, Optimizer, Scheduler
from types import SimpleNamespace
print("Estructura de MODEL_CONFIG:", MODEL_CONFIG)
print("Claves disponibles en MODEL_CONFIG:", list(MODEL_CONFIG.keys()))
config = SimpleNamespace(
    vocab_size=len(tokenizer),
    block_size=BLOCK_SIZE,
    n_embd=MODEL_CONFIG["n_embd"],
    n_head=MODEL_CONFIG["n_head"],
    n_inner=MODEL_CONFIG["n_inner"],
    dropout=MODEL_CONFIG["dropout"],
    halt_max_steps=MAX_HALT_STEPS,
    ponder_loss_weight=PONDER_WEIGHT,
    halt_bias_init=HALT_BIAS_INIT
)
model = HRMText1(config).to(device)

decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    if p.dim() == 1 or any(k in n.lower() for k in ["bias", "norm"]):
        no_decay.append(p)
    else: decay.append(p)

optimizer = AdamW(
    [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
    lr=LEARNING_RATE_MAX, betas=(0.9, 0.95), eps=1e-8
)

try:
    print(f"Downloading latest model from '{HF_REPO_ID}'...")
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=LOCAL_WEIGHTS_PATH)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)
    print("Successfully loaded model from the Hub. Continuing training.")
except Exception as e:
    print(f"Could not download model from Hub. Starting fresh. Error: {e}")

start_epoch, global_step = 0, 0
if os.path.exists(LOCAL_CHECKPOINT_PATH):
    try:
        chk = torch.load(LOCAL_CHECKPOINT_PATH, map_location="cpu")
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk.get("epoch", -1) + 1
        global_step = chk.get("global_step", 0)
        print(f"Resuming from Epoch {start_epoch}, global_step {global_step}.")
    except Exception as e:
        print(f"Warning: failed to load local training state: {e}")

steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS

num_training_steps = NUM_EPOCHS * steps_per_epoch
scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=LEARNING_RATE_MIN)

scaler = torch.cuda.amp.GradScaler(enabled=(MIXED_PRECISION and device.type == "cuda"))

# ----------------------------
# Training Loop

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    # Update ponder weight for this epoch
    current_ponder_weight = PONDER_WEIGHT * (PONDER_WEIGHT_DECAY ** epoch)
    model.ponder_loss_weight = current_ponder_weight

    progress = tqdm(train_loader, desc=f"Epoch {epoch} | Ponder Weight: {current_ponder_weight:.4f}")
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids

        with torch.amp.autocast("cuda", enabled=(MIXED_PRECISION and device.type == "cuda")):
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs["loss"]
            lm_loss = outputs.get("lm_loss", torch.tensor(0.0))
            ponder_loss = outputs.get("ponder_loss", torch.tensor(0.0))

        if loss is None or not torch.isfinite(loss):
            print("Non-finite loss, skipping batch.")
            optimizer.zero_grad(set_to_none=True); continue

        loss_to_backprop = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss_to_backprop).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1


            if global_step > 0 and global_step % SAVE_STEPS == 0:
                print(f"\nSaving checkpoint at global_step {global_step}")
                # Create a dedicated directory for this checkpoint
                checkpoint_dir = f"checkpoint-{global_step}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Save model weights and full training state for resuming
                model_path = os.path.join(checkpoint_dir, LOCAL_WEIGHTS_PATH)
                state_path = os.path.join(checkpoint_dir, LOCAL_CHECKPOINT_PATH)

                torch.save(model.state_dict(), model_path)
                torch.save({
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step
                }, state_path)

                # Upload the checkpoint folder to Hugging Face Hub
                try:
                    api = HfApi()
                    commit_msg = f"Checkpoint at step {global_step} (Epoch {epoch})"
                    api.upload_folder(
                        folder_path=checkpoint_dir,
                        repo_id=HF_REPO_ID,
                        repo_type="model",
                        commit_message=commit_msg,
                    )
                    print(f"Successfully uploaded checkpoint {global_step} to {HF_REPO_ID}")
                    # Clean up the local folder after a successful upload
                    shutil.rmtree(checkpoint_dir)
                except Exception as e:
                    print(f"Upload failed for checkpoint {global_step}: {e}")

        progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.inference_mode():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids, labels=input_ids, attention_mask=attention_mask)
            if out["loss"] is not None and torch.isfinite(out["loss"]):
                total_val_loss += out["loss"].item()

    avg_val_loss = total_val_loss / max(1, len(val_loader))
    val_perplexity = torch.exp(torch.tensor(avg_val_loss))
    print(f"\nEpoch {epoch} | Validation Loss: {avg_val_loss:.4f} | Validation Perplexity: {val_perplexity:.2f}")

    # Save Model & Check for Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print(f"New best validation loss: {best_val_loss:.4f}. Saving best model to '{BEST_MODEL_PATH}'")
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        patience_counter += 1
        print(f"Validation loss did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

    # Save full checkpoint for resuming (this one is for local recovery)
    torch.save({"epoch": epoch, "optimizer_state_dict": optimizer.state_dict(), "global_step": global_step}, LOCAL_CHECKPOINT_PATH)

    print("\nUploading best model from epoch to HuggingFace Hub...")
    try:
        api = HfApi()
        # Upload the best model to the root of the repo
        api.upload_file(
            path_or_fileobj=BEST_MODEL_PATH,
            path_in_repo=LOCAL_WEIGHTS_PATH, # Overwrites the main model file
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"End of Epoch {epoch}: Val Loss {avg_val_loss:.4f}, Perplexity {val_perplexity:.2f}",
            token=HF_TOKEN
        )

        if UPDATE_README:
            card_text = f"""---
base_model: {T5_TOKENIZER_REPO}
tags: [- hrm, - act, - wikitext]
metrics: [- loss, - perplexity]
---
# HRM-Text1

**HRM-Text1** is an experimental text generation model based on the **Hierarchical Recurrent Memory (HRM)** architecture. It was trained from scratch on the `roneneldan/TinyStories` dataset, designed to produce simple, coherent, and child-appropriate stories.

The model utilizes the HRM structure, consisting of a "Specialist" module for low-level processing and a "Manager" module for high-level abstraction and planning. This architecture aims to handle long-range dependencies more effectively by summarizing information at different temporal scales.

## Model Description

- **Architecture:** Hierarchical Recurrent Memory (HRM)
- **Training Data:** [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- **Original Paper:** [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734)
- **Tokenizer:** `{T5_TOKENIZER_REPO}` (slow T5 SentencePiece)
- **Vocab Size**: {len(tokenizer)}
- **Objective:** Causal Language Modeling

### Latest Performance (Epoch {epoch})
- **Validation Loss**: `{avg_val_loss:.4f}`
- **Validation Perplexity**: `{val_perplexity:.2f}`
"""
            with open("README.md", "w") as f:
                f.write(card_text)

            api.upload_file(
                path_or_fileobj="README.md",
                path_in_repo="README.md",
                repo_id=HF_REPO_ID,
                repo_type="model",
                commit_message=f"Model card updated after epoch {epoch}",
                token=HF_TOKEN
            )

        print("Finished upload to repo!")
    except Exception as e:
        print(f"Upload failed: {e}")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

print("Training run finished.")
# Exportar modelo y tokenizer en formato Hugging Face
model.save_pretrained("output_model")
tokenizer.save_pretrained("output_model")

# ----------------------------
# Chatting!
def chat_with_model(prompt_text, model, tokenizer, max_new_tokens=60, temperature=0.7, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(input_ids, attention_mask=attention_mask)
            next_token_logits = out["logits"][:, -1, :] / max(temperature, 1e-6)

            # Top-K filtering
            topk_vals, topk_idx = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
            mask = torch.full_like(next_token_logits, float("-inf"))
            mask.scatter_(1, topk_idx, topk_vals)

            probs = F.softmax(mask, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1,1), device=device, dtype=torch.long)], dim=1)

            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

print("\nSampling Generation...")

if os.path.exists(BEST_MODEL_PATH):
    print("Loading best model for generation...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    try:
        print(chat_with_model("Hello, how are you?", model, tokenizer, max_new_tokens=30))
        print(chat_with_model("The meaning of life is", model, tokenizer, max_new_tokens=50))
    except Exception as e:
        print(f"Generation test failed: {e}")
else:
    print("Best model file not found. Could not run generation test.")
