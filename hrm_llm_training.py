# -*- coding: utf-8 -*-
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

# Asumiendo que hrm_text1_modeling.py está en el mismo directorio
# from hrm_text1_modeling import HRMText1

# ---------------------------------------------------------
# HRM Architecture (w/ Positional Embeddings for CausalLM)
# I shall refer to it as HRM-Text1
# He copiado el código del modelo aquí para que el script sea autocontenido.
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
        # Corregido config.d_ff
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

        halting_probs = torch.zeros((batch_size, seq_len, self.max_steps), device=device)
        remainders = torch.ones((batch_size, seq_len), device=device)
        total_z_H = torch.zeros_like(z_H)
        n_updates = torch.zeros((batch_size, seq_len), device=device)

        eps = 1e-6
        for step in range(self.max_steps):
            p_halt = self.halt_head(z_H).squeeze(-1)
            p_halt = p_halt.clamp(eps, 1 - eps)
            is_last_step = (step == self.max_steps - 1)

            halt_now_prob = torch.ones_like(p_halt) if is_last_step else p_halt
            contrib = remainders * halt_now_prob

            halting_probs[:, :, step] = contrib
            total_z_H += contrib.unsqueeze(-1) * z_H

            remainders = remainders * (1 - p_halt) if not is_last_step else torch.zeros_like(remainders)

            if not is_last_step:
                n_updates += remainders

            if torch.all(remainders < eps):
                break

            z_H, z_L = self.inner_model(z_H, z_L, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        logits = self.lm_head(total_z_H)
        loss, ponder_loss, lm_loss = None, None, None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            ponder_loss = torch.mean(n_updates)
            loss = lm_loss + self.ponder_loss_weight * ponder_loss

        return {"loss": loss, "logits": logits, "ponder_loss": ponder_loss, "lm_loss": lm_loss}
# ---------------------------------------------------------


# ----------------------------
# Training Parameters
HF_REPO_ID = "qingy2024/HRM-Text1"
SEED = 42
NUM_EPOCHS = 2
BLOCK_SIZE = 512
BATCH_SIZE = 170
GRAD_ACCUM_STEPS = 1 # Effective batch size = 140 * 1 = 140
LEARNING_RATE_MAX = 2e-4
LEARNING_RATE_MIN = 1e-6
WEIGHT_DECAY = 0.01
MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 2 # Stop if validation loss doesn't improve for 2 epochs

SAVE_STEPS = 500  # Save a checkpoint every 500 global steps

# HRM Model Hyperparameters
MODEL_CONFIG = {
    "n_embd": 512,
    "n_head": 8,
    "d_ff": 2048, # Renombrado de n_inner a d_ff para claridad
    "dropout": 0.1,
    "vocab_size": None,
    "block_size": BLOCK_SIZE
}
MAX_HALT_STEPS = 8
PONDER_WEIGHT = 1e-2
PONDER_WEIGHT_DECAY = 0.98
HALT_BIAS_INIT = -2.2

T5_TOKENIZER_REPO = "t5-small"
LOCAL_CHECKPOINT_PATH = "local_training_state.pt"
LOCAL_WEIGHTS_PATH = "pytorch_model.bin"
BEST_MODEL_PATH = "best_model.bin"

WANDB_PROJECT = "HRM-Text1"
UPDATE_README = True

TRAIN_FIELD_MODE = "input_answer"

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

try:
    HF_TOKEN = os.environ['HF_TOKEN']
    HfFolder.save_token(HF_TOKEN)
    print("Hugging Face token loaded.")
except Exception as e:
    print("HF_TOKEN secret not found.")
    HF_TOKEN = None

# ----------------------------
# Tokenizer
print("Loading tokenizer (T5 slow)...")
os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"
tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_REPO, use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.padding_side = "left"
print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}; eos={tokenizer.eos_token}; pad={tokenizer.pad_token}")
MODEL_CONFIG["vocab_size"] = len(tokenizer)


# -----------------------------------
# Data Loading and Preprocessing
print(f"Cargando y preparando dataset sanjay920/goat-sharegpt con el modo: {TRAIN_FIELD_MODE}")

raw_datasets = load_dataset("sanjay920/goat-sharegpt")

def tokenize_function(examples):
    texts = []
    if TRAIN_FIELD_MODE == "conversations":
        for conv_json_str in examples.get("conversations", []):
            try:
                conv_list = json.loads(conv_json_str)
                msg_texts = [str(msg.get("value", msg.get("content", ""))) for msg in conv_list]
                full_text = " ".join(filter(None, msg_texts)) + tokenizer.eos_token
                texts.append(full_text)
            except (json.JSONDecodeError, TypeError):
                continue
    elif TRAIN_FIELD_MODE == "input_answer":
        for inp, ans in zip(examples.get("input", []), examples.get("answer", [])):
            if inp and ans:
                full_text = f"{str(inp)} {str(ans)}{tokenizer.eos_token}"
                texts.append(full_text)
    else:
        raise ValueError(f"Modo de campo de entrenamiento no soportado: {TRAIN_FIELD_MODE}")
    return tokenizer(
        texts,
        truncation=True,
        max_length=BLOCK_SIZE,
        padding="max_length",
        add_special_tokens=False,
    )

tokenized_splits = {}
dataset_splits = raw_datasets.keys()

if "validation" not in dataset_splits:
    print("No se encontró split 'validation'. Creando partición (10%) desde 'train'...")
    train_val_split = raw_datasets["train"].train_test_split(test_size=0.1, seed=SEED)
    raw_datasets["train"] = train_val_split["train"]
    raw_datasets["validation"] = train_val_split["test"]
    dataset_splits = raw_datasets.keys()

for split_name in dataset_splits:
    if split_name in ["train", "validation"]:
        tokenized_splits[split_name] = raw_datasets[split_name].map(
            tokenize_function,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=raw_datasets[split_name].column_names,
        )
        tokenized_splits[split_name].set_format("torch")

num_workers = os.cpu_count() or 2
train_loader = DataLoader(tokenized_splits["train"], batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(tokenized_splits["validation"], batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)


# --------------------------------
# Model, Optimizer, Scheduler
from types import SimpleNamespace
config = SimpleNamespace(**MODEL_CONFIG,
    halt_max_steps=MAX_HALT_STEPS,
    ponder_loss_weight=PONDER_WEIGHT,
    halt_bias_init=HALT_BIAS_INIT)

model = HRMText1(config).to(device)

if torch.__version__.startswith("2"):
    print("Compilando el modelo con torch.compile()...")
    model = torch.compile(model)

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
    print(f"Intentando descargar el modelo desde '{HF_REPO_ID}'...")
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=LOCAL_WEIGHTS_PATH)
    state = torch.load(weights_path, map_location=device)
    # Cargar los pesos compilados en un modelo no compilado o viceversa requiere manejo
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    base_model.load_state_dict(state, strict=False)
    print("Pesos del modelo cargados exitosamente desde el Hub.")
except Exception as e:
    print(f"No se pudo descargar el modelo del Hub. Empezando de cero. Error: {e}")

start_epoch, global_step = 0, 0
if os.path.exists(LOCAL_CHECKPOINT_PATH):
    try:
        chk = torch.load(LOCAL_CHECKPOINT_PATH, map_location="cpu")
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk.get("epoch", 0)
        global_step = chk.get("global_step", 0)
        print(f"Reanudando desde la Época {start_epoch}, paso global {global_step}.")
    except Exception as e:
        print(f"Advertencia: no se pudo cargar el estado de entrenamiento local: {e}")

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
    current_ponder_weight = PONDER_WEIGHT * (PONDER_WEIGHT_DECAY ** epoch)
    
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    base_model.ponder_loss_weight = current_ponder_weight

    progress = tqdm(train_loader, desc=f"Época {epoch} | Ponder W: {current_ponder_weight:.4f}")
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress):
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

            if global_step > 0 and global_step % SAVE_STEPS == 0:
                print(f"\nGuardando checkpoint en el paso global {global_step}")
                checkpoint_dir = f"checkpoint-{global_step}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
                torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, LOCAL_WEIGHTS_PATH))
                # Guardar otros estados si es necesario

        progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.inference_mode():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids, labels=input_ids, attention_mask=attention_mask)
            if out.get("loss") is not None and torch.isfinite(out["loss"]):
                total_val_loss += out["loss"].item()

    avg_val_loss = total_val_loss / max(1, len(val_loader))
    val_perplexity = torch.exp(torch.tensor(avg_val_loss))
    print(f"\nÉpoca {epoch} | Pérdida de Validación: {avg_val_loss:.4f} | Perplejidad de Validación: {val_perplexity:.2f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print(f"Nueva mejor pérdida de validación: {best_val_loss:.4f}. Guardando el mejor modelo en '{BEST_MODEL_PATH}'")
        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save(model_to_save.state_dict(), BEST_MODEL_PATH)
    else:
        patience_counter += 1
        print(f"La pérdida de validación no mejoró. Paciencia: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

    # Guardar checkpoint local para reanudar
    torch.save({
        "epoch": epoch + 1,
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step
    }, LOCAL_CHECKPOINT_PATH)
    
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Detención temprana activada después de {epoch+1} épocas.")
        break


# ----------------------------
# Guardado Final y Subida a Hub
print("Entrenamiento finalizado.")
if os.path.exists(BEST_MODEL_PATH):
    print("\nSubiendo el mejor modelo al HuggingFace Hub...")
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=BEST_MODEL_PATH,
            path_in_repo=LOCAL_WEIGHTS_PATH,
            repo_id=HF_REPO_ID, repo_type="model",
            commit_message=f"Final de entrenamiento: Val Loss {best_val_loss:.4f}, Perplejidad {torch.exp(torch.tensor(best_val_loss)):.2f}",
            token=HF_TOKEN
        )
        print("Subida del modelo al Hub completada.")
    except Exception as e:
        print(f"La subida falló: {e}")


# ----------------------------
# Chatting!
def chat_with_model(prompt_text, model, tokenizer, max_new_tokens=60, temperature=0.7, top_k=50):
    """Genera texto a partir de un prompt usando el modelo entrenado."""
    model.eval()
    
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Asegúrate de que el modelo reciba la máscara de atención
            outputs = model(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs["logits"][:, -1, :]

            # Aplicar temperatura
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Top-K filtering
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                topk_vals, topk_idx = torch.topk(next_token_logits, k=top_k)
                mask = torch.full_like(next_token_logits, float("-inf"))
                mask.scatter_(1, topk_idx, topk_vals)
                filtered_logits = mask
            else:
                filtered_logits = next_token_logits

            probs = F.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            # Actualizar la máscara de atención para el nuevo token
            attention_mask = torch.cat([attention_mask, torch.ones((1,1), device=device, dtype=torch.long)], dim=1)
            
            # Detener si se genera el token EOS
            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                break

    # Decodificar toda la secuencia
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return full_text


print("\n--- Probando la Generación del Modelo ---")

if os.path.exists(BEST_MODEL_PATH):
    print("Cargando el mejor modelo para la generación...")
    
    # Crea una nueva instancia del modelo para la inferencia
    inference_config = SimpleNamespace(**MODEL_CONFIG,
        halt_max_steps=MAX_HALT_STEPS,
        ponder_loss_weight=PONDER_WEIGHT,
        halt_bias_init=HALT_BIAS_INIT)
    inference_model = HRMText1(inference_config).to(device)
    
    # Carga los pesos del mejor modelo
    inference_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    
    # Compila el modelo de inferencia si es posible
    if torch.__version__.startswith("2"):
        print("Compilando el modelo para una inferencia más rápida...")
        inference_model = torch.compile(inference_model)
        
    try:
        # Ejemplo 1: Una pregunta simple
        prompt1 = "What is the capital of Spain?"
        full_response1 = chat_with_model(prompt1, inference_model, tokenizer, max_new_tokens=30)
        # Extrae solo la respuesta generada
        generated_part1 = full_response1[len(prompt1):].strip()
        print(f"\nPrompt: {prompt1}")
        print(f"Respuesta: {generated_part1}")

        # Ejemplo 2: Un prompt que invita a continuar una frase
        prompt2 = "The meaning of life is"
        full_response2 = chat_with_model(prompt2, inference_model, tokenizer, max_new_tokens=50)
        generated_part2 = full_response2[len(prompt2):].strip()
        print(f"\nPrompt: {prompt2}")
        print(f"Respuesta: {generated_part2}")
        
        # Ejemplo 3: Un prompt matemático, similar a los datos de entrenamiento
        prompt3 = "530991+6051993"
        full_response3 = chat_with_model(prompt3, inference_model, tokenizer, max_new_tokens=20)
        generated_part3 = full_response3[len(prompt3):].strip()
        print(f"\nPrompt: {prompt3}")
        print(f"Respuesta: {generated_part3}")

    except Exception as e:
        print(f"El test de generación falló: {e}")
else:
    print("El archivo del mejor modelo no fue encontrado. No se pudo ejecutar el test de generación.")