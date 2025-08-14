# -*- coding: utf-8 -*-
"""
HRM-Text1 Training Script (con Modo de Depuración y Batch Size Condicional Corregido)

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

# ---------------------------------------------------------
# HRM Architecture
class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))

class SwiGLUMuchPelu(nn.Module):
    def __init__(self, n_embd, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(n_embd, d_ff, bias=False)
        self.w2 = nn.Linear(n_embd, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class HRMBlock(nn.Module):
    def __init__(self, n_embd, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = SwiGLUMuchPelu(n_embd, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class HRMInner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.H_module = HRMBlock(config.n_embd, config.n_head, config.d_ff, config.dropout)
        self.L_module = HRMBlock(config.n_embd, config.n_head, config.d_ff, config.dropout)
    def forward(self, z_H, z_L, attn_mask=None, key_padding_mask=None):
        z_L_input = z_L + z_H
        z_L_new = self.L_module(z_L_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        z_H_input = z_H + z_L_new
        z_H_new = self.H_module(z_H_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return z_H_new, z_L_new

class HRMText1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.register_buffer("pos_ids", torch.arange(config.block_size).unsqueeze(0))
        self.inner_model = HRMInner(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.halt_head = nn.Sequential(nn.Linear(config.n_embd, 1), nn.Sigmoid())
        self.max_steps = config.halt_max_steps
        self.ponder_loss_weight = config.ponder_loss_weight

        with torch.no_grad():
            halt_bias_value = getattr(config, "halt_bias_init", -2.0)
            self.halt_head[0].bias.fill_(halt_bias_value)

    def forward(self, input_ids, labels=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        z_L = self.token_embeddings(input_ids) + self.pos_embeddings(self.pos_ids[:, :seq_len])
        z_H = torch.zeros_like(z_L)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        remainders = torch.ones((batch_size, seq_len), device=device)
        total_z_H = torch.zeros_like(z_H)
        n_updates = torch.zeros((batch_size, seq_len), device=device)
        eps = 1e-6
        for step in range(self.max_steps):
            p_halt = self.halt_head(z_H).squeeze(-1).clamp(eps, 1 - eps)
            is_last_step = (step == self.max_steps - 1)
            halt_now_prob = torch.ones_like(p_halt) if is_last_step else p_halt
            contrib = remainders * halt_now_prob
            total_z_H += contrib.unsqueeze(-1) * z_H
            remainders = remainders * (1 - p_halt) if not is_last_step else torch.zeros_like(remainders)
            if not is_last_step:
                n_updates += remainders
            if torch.all(remainders < eps):
                break
            z_H, z_L = self.inner_model(z_H, z_L, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        logits = self.lm_head(total_z_H)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            ponder_loss = torch.mean(n_updates)
            loss = lm_loss + self.ponder_loss_weight * ponder_loss
        return {"loss": loss, "logits": logits}
# ---------------------------------------------------------

# ==============================================================================
# --- INICIO DE LA CONFIGURACIÓN PRINCIPAL ---
# ==============================================================================

# --- Parámetros de Depuración y Prueba ---
DEBUG_MODE = True  # ¡¡¡ Poner en False para un entrenamiento completo !!!
NUM_DEBUG_SAMPLES = 500  # Número de muestras para el dataset reducido
DEBUG_BATCH_SIZE = 4   # Tamaño del lote en modo de depuración (muy pequeño)

# --- Training Parameters ---
HF_REPO_ID = "qingy2024/HRM-Text1"
SEED = 42
NUM_EPOCHS = 2
BLOCK_SIZE = 512
TRAIN_BATCH_SIZE = 170
GRAD_ACCUM_STEPS = 1
LEARNING_RATE_MAX = 5e-5
LEARNING_RATE_MIN = 1e-6
WEIGHT_DECAY = 0.01
MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 2
SAVE_STEPS = 500
UPDATE_README = True

# --- HRM Model Hyperparameters ---
MODEL_CONFIG = {
    "n_embd": 512, "n_head": 8, "d_ff": 2048, "dropout": 0.1,
    "vocab_size": None, "block_size": BLOCK_SIZE
}
MAX_HALT_STEPS = 8
PONDER_WEIGHT = 1e-2
PONDER_WEIGHT_DECAY = 0.98
HALT_BIAS_INIT = -2.2

# --- Otros Parámetros ---
T5_TOKENIZER_REPO = "t5-small"
LOCAL_CHECKPOINT_PATH = "local_training_state.pt"
LOCAL_WEIGHTS_PATH = "pytorch_model.bin"
BEST_MODEL_PATH = "best_model.bin"
TRAIN_FIELD_MODE = "input_answer"

# ==============================================================================
# --- FIN DE LA CONFIGURACIÓN PRINCIPAL ---
# ==============================================================================

# Ajuste del BATCH_SIZE según el modo de depuración
BATCH_SIZE = DEBUG_BATCH_SIZE if DEBUG_MODE else TRAIN_BATCH_SIZE
print(f"BATCH_SIZE actual: {BATCH_SIZE} (DEBUG_MODE={DEBUG_MODE})")

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

try:
    HF_TOKEN = os.environ['HF_TOKEN']
    HfFolder.save_token(HF_TOKEN)
    print("Hugging Face token loaded.")
except Exception:
    print("HF_TOKEN secret not found.")
    HF_TOKEN = None

# Tokenizer
print("Loading tokenizer (T5 slow)...")
os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"
tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_REPO, use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.padding_side = "left"
print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
MODEL_CONFIG["vocab_size"] = len(tokenizer)

# Data Loading and Preprocessing
print(f"Cargando dataset sanjay920/goat-sharegpt con el modo: {TRAIN_FIELD_MODE}")
raw_datasets = load_dataset("sanjay920/goat-sharegpt")

if DEBUG_MODE:
    print(f"\n!!! MODO DE PRUEBA ACTIVO: Reduciendo el dataset a {NUM_DEBUG_SAMPLES} ejemplos. !!!\n")
    if "train" in raw_datasets:
        # Asegurarse de que haya suficientes muestras para la división
        if len(raw_datasets["train"]) > NUM_DEBUG_SAMPLES:
            raw_datasets["train"] = raw_datasets["train"].shuffle(seed=SEED).select(range(NUM_DEBUG_SAMPLES))
        else:
            print(f"Advertencia: NUM_DEBUG_SAMPLES ({NUM_DEBUG_SAMPLES}) es mayor que el dataset completo. Usando el dataset completo.")
    else:
        print("Advertencia: No se encontró el split 'train' para reducir su tamaño en modo DEBUG.")

def tokenize_function(examples):
    texts = []
    if TRAIN_FIELD_MODE == "input_answer":
        for inp, ans in zip(examples.get("input", []), examples.get("answer", [])):
            if inp and ans:
                texts.append(f"{str(inp)} {str(ans)}{tokenizer.eos_token}")
    return tokenizer(texts, truncation=True, max_length=BLOCK_SIZE, padding="max_length", add_special_tokens=False)

if "validation" not in raw_datasets:
    print("No se encontró split 'validation'. Creando partición (10%) desde 'train'...")
    split = raw_datasets["train"].train_test_split(test_size=0.1, seed=SEED, stratify_by_column=None) # stratify_by_column puede dar error si no existe la columna
    raw_datasets["train"] = split["train"]
    raw_datasets["validation"] = split["test"]

tokenized_splits = {}
for split_name in ["train", "validation"]:
    tokenized_splits[split_name] = raw_datasets[split_name].map(
        tokenize_function, batched=True, num_proc=os.cpu_count(),
        remove_columns=raw_datasets[split_name].column_names)
    tokenized_splits[split_name].set_format("torch")

# DataLoaders
num_workers = os.cpu_count() or 2
# --- CAMBIO CLAVE ---
# Usar drop_last=True solo si NO estamos en modo de depuración para evitar un DataLoader vacío
train_loader = DataLoader(
    tokenized_splits["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(not DEBUG_MODE), # ¡ESTA ES LA CORRECCIÓN!
    num_workers=num_workers,
    pin_memory=True
)
val_loader = DataLoader(tokenized_splits["validation"], batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

# Model, Optimizer, Scheduler
from types import SimpleNamespace
config = SimpleNamespace(**MODEL_CONFIG, halt_max_steps=MAX_HALT_STEPS, ponder_loss_weight=PONDER_WEIGHT, halt_bias_init=HALT_BIAS_INIT)
model = HRMText1(config).to(device)

if torch.__version__.startswith("2"):
    print("Compilando el modelo con torch.compile()...")
    model = torch.compile(model)

if DEBUG_MODE:
    print("\n!!! MODO DE PRUEBA ACTIVO: Se ha activado la detección de anomalías en autograd. El entrenamiento será más lento. !!!\n")
    torch.autograd.set_detect_anomaly(True)

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

start_epoch, global_step, best_val_loss = 0, 0, float('inf')
if os.path.exists(LOCAL_CHECKPOINT_PATH):
    try:
        chk = torch.load(LOCAL_CHECKPOINT_PATH, map_location="cpu")
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk.get("epoch", 0)
        global_step = chk.get("global_step", 0)
        best_val_loss = chk.get("best_val_loss", float('inf'))
        print(f"Reanudando desde la Época {start_epoch}, paso global {global_step}.")
    except Exception as e:
        print(f"Advertencia: no se pudo cargar el estado de entrenamiento local: {e}")

steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
num_training_steps = NUM_EPOCHS * steps_per_epoch if steps_per_epoch > 0 else 1
scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=LEARNING_RATE_MIN)

scaler = torch.cuda.amp.GradScaler(enabled=(MIXED_PRECISION and device.type == "cuda"))

# Training Loop
patience_counter = 0
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    current_ponder_weight = PONDER_WEIGHT * (PONDER_WEIGHT_DECAY ** epoch)
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    base_model.ponder_loss_weight = current_ponder_weight

    if len(train_loader) == 0:
        print(f"El DataLoader de entrenamiento está vacío para la época {epoch}. Saltando al siguiente paso.")
        continue

    progress = tqdm(train_loader, desc=f"Época {epoch} | Ponder W: {current_ponder_weight:.4f}")
    for step, batch in enumerate(progress):
        # ... (resto del bucle de entrenamiento, igual que antes) ...
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids

        with torch.amp.autocast("cuda", enabled=(MIXED_PRECISION and device.type == "cuda")):
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs["loss"]

        if loss is None or not torch.isfinite(loss):
            print("Pérdida no finita, saltando lote.")
            optimizer.zero_grad(set_to_none=True); continue

        loss_to_backprop = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss_to_backprop).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if not DEBUG_MODE and global_step > 0 and global_step % SAVE_STEPS == 0:
                print(f"\nGuardando checkpoint en el paso global {global_step}")
                # ... Lógica de guardado de checkpoints ...

        progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    # Validation
    model.eval()
    total_val_loss = 0.0
    if len(val_loader) > 0:
      with torch.inference_mode():
          for batch in val_loader:
              input_ids = batch["input_ids"].to(device)
              attention_mask = batch["attention_mask"].to(device)
              out = model(input_ids, labels=input_ids, attention_mask=attention_mask)
              if out.get("loss") is not None and torch.isfinite(out["loss"]):
                  total_val_loss += out["loss"].item()
      avg_val_loss = total_val_loss / len(val_loader)
      val_perplexity = torch.exp(torch.tensor(avg_val_loss))
      print(f"\nÉpoca {epoch} | Pérdida de Validación: {avg_val_loss:.4f} | Perplejidad: {val_perplexity:.2f}")

      if avg_val_loss < best_val_loss:
          best_val_loss = avg_val_loss
          patience_counter = 0
          print(f"Nueva mejor pérdida. Guardando modelo en '{BEST_MODEL_PATH}'")
          model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
          torch.save(model_to_save.state_dict(), BEST_MODEL_PATH)
      else:
          patience_counter += 1
          print(f"Validación no mejoró. Paciencia: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
    else:
        print("El DataLoader de validación está vacío. Saltando la validación.")


    torch.save({
        "epoch": epoch + 1, "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step, "best_val_loss": best_val_loss
    }, LOCAL_CHECKPOINT_PATH)
    
    if not DEBUG_MODE and HF_TOKEN and os.path.exists(BEST_MODEL_PATH):
        # ... (código para subir al hub) ...
        pass

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Detención temprana en la época {epoch+1}.")
        break

print("Entrenamiento finalizado.")
# Exportar modelo y tokenizer en formato Hugging Face
# ¡Asegúrate de guardar el modelo base, no el compilado, para compatibilidad!
final_model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
final_model_to_save.save_pretrained("output_model")
tokenizer.save_pretrained("output_model")

# ----------------------------
# Chatting! (Código de inferencia se mantiene igual)
def chat_with_model(prompt_text, model, tokenizer, max_new_tokens=60, temperature=0.7, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs["logits"][:, -1, :]
            
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            if top_k > 0:
                top_k_val = min(top_k, next_token_logits.size(-1))
                topk_vals, topk_idx = torch.topk(next_token_logits, k=top_k_val)
                mask = torch.full_like(next_token_logits, float("-inf"))
                mask.scatter_(1, topk_idx, topk_vals)
                filtered_logits = mask
            else:
                filtered_logits = next_token_logits

            probs = F.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1,1), device=device, dtype=torch.long)], dim=1)
            
            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                break

    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return full_text


print("\n--- Probando la Generación del Modelo ---")

if os.path.exists(BEST_MODEL_PATH):
    print("Cargando el mejor modelo para la generación...")
    
    inference_config = SimpleNamespace(**MODEL_CONFIG,
        halt_max_steps=MAX_HALT_STEPS,
        ponder_loss_weight=PONDER_WEIGHT,
        halt_bias_init=HALT_BIAS_INIT)
    inference_model = HRMText1(inference_config).to(device)
    
    inference_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    
    if torch.__version__.startswith("2"):
        print("Compilando el modelo para una inferencia más rápida...")
        inference_model = torch.compile(inference_model)
        
    try:
        prompt1 = "What is the capital of Spain?"
        full_response1 = chat_with_model(prompt1, inference_model, tokenizer, max_new_tokens=30)
        generated_part1 = full_response1[len(prompt1):].strip()
        print(f"\nPrompt: {prompt1}")
        print(f"Respuesta: {generated_part1}")

        prompt2 = "The meaning of life is"
        full_response2 = chat_with_model(prompt2, inference_model, tokenizer, max_new_tokens=50)
        generated_part2 = full_response2[len(prompt2):].strip()
        print(f"\nPrompt: {prompt2}")
        print(f"Respuesta: {generated_part2}")
        
        prompt3 = "530991+6051993"
        full_response3 = chat_with_model(prompt3, inference_model, tokenizer, max_new_tokens=20)
        generated_part3 = full_response3[len(prompt3):].strip()
        print(f"\nPrompt: {prompt3}")
        print(f"Respuesta: {generated_part3}")

    except Exception as e:
        print(f"El test de generación falló: {e}")
else:
    print("El archivo del mejor modelo no fue encontrado. No se pudo ejecutar el test de generación.")