# -*- coding: utf-8 -*-
"""
HRM-Text1 Training Script con Aproximación de Gradiente de 1 Paso (sin BPTT)
VERSIÓN FINAL: Compatible con generación de texto y validación robusta.
"""

import os, shutil, pathlib, random, json, datetime, math, contextlib
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset
from transformers import T5Tokenizer, PreTrainedModel, PretrainedConfig, GenerationMixin
from tqdm.auto import tqdm

from huggingface_hub import HfApi, HfFolder, hf_hub_download

# Optimización específica para NVIDIA Ampere+
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    print("GPU NVIDIA compatible con TF32 detectada. Activando la precisión de matmul 'high'.")
    torch.set_float32_matmul_precision('high')

# ==============================================================================
# --- DEFINICIÓN DEL MODELO ---
# ==============================================================================

class HRMText1Config(PretrainedConfig):
    model_type = "hrm_text1"
    def __init__(self, vocab_size=32100, block_size=512, n_embd=512, n_head=8, d_ff=2048, dropout=0.1, halt_max_steps=8, ponder_loss_weight=1e-2, halt_bias_init=-2.2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size, self.block_size, self.n_embd, self.n_head, self.d_ff, self.dropout, self.halt_max_steps, self.ponder_loss_weight, self.halt_bias_init = vocab_size, block_size, n_embd, n_head, d_ff, dropout, halt_max_steps, ponder_loss_weight, halt_bias_init

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-6):
        super().__init__(); self.eps, self.weight = eps, nn.Parameter(torch.ones(n_embd))
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))

class SwiGLUMuchPelu(nn.Module):
    def __init__(self, n_embd, d_ff, dropout=0.1):
        super().__init__(); self.w1, self.w2, self.w3, self.dropout = nn.Linear(n_embd, d_ff, bias=False), nn.Linear(n_embd, d_ff, bias=False), nn.Linear(d_ff, n_embd, bias=False), nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class HRMBlock(nn.Module):
    def __init__(self, n_embd, n_head, d_ff, dropout=0.1):
        super().__init__(); self.norm1, self.attn, self.norm2, self.mlp, self.dropout = RMSNorm(n_embd), nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True), RMSNorm(n_embd), SwiGLUMuchPelu(n_embd, d_ff, dropout), nn.Dropout(dropout)
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x_norm = self.norm1(x); attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False); x = x + self.dropout(attn_out); x = x + self.dropout(self.mlp(self.norm2(x))); return x

class HRMInner(nn.Module):
    def __init__(self, config):
        super().__init__(); self.H_module, self.L_module = HRMBlock(config.n_embd, config.n_head, config.d_ff, config.dropout), HRMBlock(config.n_embd, config.n_head, config.d_ff, config.dropout)
    def forward(self, z_H, z_L, attn_mask=None, key_padding_mask=None):
        z_L_input = z_L + z_H; z_L_new = self.L_module(z_L_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask); z_H_input = z_H + z_L_new; z_H_new = self.H_module(z_H_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask); return z_H_new, z_L_new

# ===================== CLASE DE MODELO FINAL Y CORREGIDA =====================
class HRMText1(PreTrainedModel, GenerationMixin):
    config_class = HRMText1Config
    main_input_name = "input_ids"

    def __init__(self, config: HRMText1Config):
        super().__init__(config)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.register_buffer("pos_ids", torch.arange(config.block_size).unsqueeze(0))
        self.inner_model = HRMInner(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.halt_head = nn.Sequential(nn.Linear(config.n_embd, 1), nn.Sigmoid())
        with torch.no_grad():
            self.halt_head[0].bias.fill_(config.halt_bias_init)

    def forward(self, input_ids, labels=None, attention_mask=None, past_key_values=None):
        batch_size, seq_len = input_ids.shape; device = input_ids.device
        z_L = self.token_embeddings(input_ids) + self.pos_embeddings(self.pos_ids[:, :seq_len])
        z_H = torch.zeros_like(z_L)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

        remainders = torch.ones((batch_size, seq_len), device=device)
        total_z_H = torch.zeros_like(z_H)
        n_updates = torch.zeros((batch_size, seq_len), device=device)
        eps = 1e-6

        context_manager = torch.no_grad() if self.training else contextlib.nullcontext()

        with context_manager:
            for step in range(self.config.halt_max_steps - 1):
                z_H, z_L = z_H.detach(), z_L.detach()
                p_halt = self.halt_head(z_H).squeeze(-1).clamp(eps, 1 - eps)
                contrib = remainders * p_halt
                total_z_H += contrib.unsqueeze(-1) * z_H
                remainders = remainders * (1 - p_halt)
                n_updates += remainders
                if torch.all(remainders < eps):
                    remainders.fill_(0.0)
                    break
                z_H, z_L = self.inner_model(z_H, z_L, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        z_H, z_L = z_H.detach(), z_L.detach()
        contrib = remainders
        total_z_H += contrib.unsqueeze(-1) * z_H
        
        logits = self.lm_head(total_z_H)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            ponder_loss = torch.mean(n_updates + remainders)
            loss = lm_loss + self.config.ponder_loss_weight * ponder_loss
                
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

# ==============================================================================
# --- CONFIGURACIÓN DEL SCRIPT ---
# ==============================================================================
DEBUG_MODE, NUM_DEBUG_SAMPLES, DEBUG_BATCH_SIZE = True, 1000, 4
HF_REPO_ID, SEED, NUM_EPOCHS, BLOCK_SIZE, TRAIN_BATCH_SIZE, GRAD_ACCUM_STEPS = "qingy2024/HRM-Text1", 42, 2, 512, 185, 1
LEARNING_RATE_MAX, LEARNING_RATE_MIN, WEIGHT_DECAY = 2e-4, 1e-6, 0.01
MIXED_PRECISION, EARLY_STOPPING_PATIENCE, SAVE_STEPS, UPDATE_README = True, 2, 500, True
MODEL_PARAMS = {"n_embd": 512, "n_head": 8, "d_ff": 2048, "dropout": 0.1, "halt_max_steps": 8, "ponder_loss_weight": 1e-2, "halt_bias_init": -2.2}
T5_TOKENIZER_REPO, LOCAL_CHECKPOINT_PATH, LOCAL_WEIGHTS_PATH = "t5-small", "local_training_state.pt", "pytorch_model.bin"
BEST_MODEL_PATH, TRAIN_FIELD_MODE = "best_model.bin", "input_answer"
PONDER_WEIGHT, PONDER_WEIGHT_DECAY = 1e-2, 0.9

# ==============================================================================
# --- FIN DE LA CONFIGURACIÓN ---
# ==============================================================================

BATCH_SIZE = DEBUG_BATCH_SIZE if DEBUG_MODE else TRAIN_BATCH_SIZE
print(f"BATCH_SIZE actual: {BATCH_SIZE} (DEBUG_MODE={DEBUG_MODE})")

def set_seed(seed: int):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU NVIDIA (CUDA) detectada. Usando 'cuda'.")
elif hasattr(torch.backends, "hip") and torch.backends.hip.is_available():
    device = torch.device("rocm")
    print("GPU AMD (ROCm) detectada. Usando 'rocm'.")
else:
    device = torch.device("cpu")
    print("No se detectó GPU compatible. Usando CPU.")
print(f"Dispositivo final en uso: {device}")

try:
    HF_TOKEN = os.environ['HF_TOKEN']; HfFolder.save_token(HF_TOKEN); print("Hugging Face token loaded.")
except Exception:
    print("HF_TOKEN secret not found."); HF_TOKEN = None

print("Loading tokenizer (T5 slow)...")
os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"
tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_REPO, use_fast=False, trust_remote_code=True, legacy=False)
if tokenizer.pad_token is None: tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.padding_side = "left"
print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

print(f"Cargando dataset sanjay920/goat-sharegpt con el modo: {TRAIN_FIELD_MODE}")
raw_datasets = load_dataset("sanjay920/goat-sharegpt")

if DEBUG_MODE:
    print(f"\n!!! MODO DE PRUEBA ACTIVO: Reduciendo el dataset a {NUM_DEBUG_SAMPLES} ejemplos. !!!\n")
    if "train" in raw_datasets and len(raw_datasets["train"]) > NUM_DEBUG_SAMPLES:
        raw_datasets["train"] = raw_datasets["train"].shuffle(seed=SEED).select(range(NUM_DEBUG_SAMPLES))

def tokenize_function(examples):
    texts = []
    if TRAIN_FIELD_MODE == "input_answer":
        for inp, ans in zip(examples.get("input", []), examples.get("answer", [])):
            if inp and ans: texts.append(f"{str(inp)} {str(ans)}{tokenizer.eos_token}")
    return tokenizer(texts, truncation=True, max_length=BLOCK_SIZE, padding="max_length", add_special_tokens=False)

if "validation" not in raw_datasets:
    print("No se encontró split 'validation'. Creando partición (10%) desde 'train'..."); split = raw_datasets["train"].train_test_split(test_size=0.1, seed=SEED); raw_datasets["train"], raw_datasets["validation"] = split["train"], split["test"]

tokenized_splits = {}
for split_name in ["train", "validation"]:
    tokenized_splits[split_name] = raw_datasets[split_name].map(tokenize_function, batched=True, num_proc=os.cpu_count(), remove_columns=raw_datasets[split_name].column_names)
    tokenized_splits[split_name].set_format("torch")

num_workers = os.cpu_count() or 2
train_loader = DataLoader(tokenized_splits["train"], batch_size=BATCH_SIZE, shuffle=True, drop_last=(not DEBUG_MODE), num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(tokenized_splits["validation"], batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

config = HRMText1Config(vocab_size=len(tokenizer), block_size=BLOCK_SIZE, **MODEL_PARAMS)
model = HRMText1(config).to(device)

if torch.__version__.startswith("2"):
    print("Compilando el modelo con torch.compile()...")
    model = torch.compile(model)

if DEBUG_MODE:
    print("\n!!! MODO DE PRUEBA ACTIVO: Se ha activado la detección de anomalías. !!!\n")
    torch.autograd.set_detect_anomaly(True)

decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    if p.dim() == 1 or any(k in n.lower() for k in ["bias", "norm"]): no_decay.append(p)
    else: decay.append(p)

optimizer = AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}], lr=LEARNING_RATE_MAX, betas=(0.9, 0.95), eps=1e-8)

start_epoch, global_step, best_val_loss = 0, 0, float('inf')
if os.path.exists(LOCAL_CHECKPOINT_PATH):
    try:
        chk = torch.load(LOCAL_CHECKPOINT_PATH, map_location="cpu")
        optimizer.load_state_dict(chk["optimizer_state_dict"]); start_epoch, global_step, best_val_loss = chk.get("epoch", 0), chk.get("global_step", 0), chk.get("best_val_loss", float('inf'))
        print(f"Reanudando desde la Época {start_epoch}, paso global {global_step}.")
    except Exception as e:
        print(f"Advertencia: no se pudo cargar el estado de entrenamiento local: {e}")

steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
num_training_steps = NUM_EPOCHS * steps_per_epoch if steps_per_epoch > 0 else 1
scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=LEARNING_RATE_MIN)

scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and device.type != 'cpu'))

patience_counter = 0
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train(); base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    base_model.config.ponder_loss_weight = PONDER_WEIGHT * (PONDER_WEIGHT_DECAY ** epoch)

    progress = tqdm(train_loader, desc=f"Época {epoch} | Ponder W: {base_model.config.ponder_loss_weight:.4f}")
    for step, batch in enumerate(progress):
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["input_ids"].to(device)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16 if device.type != 'cpu' else torch.float32, enabled=(MIXED_PRECISION and device.type != 'cpu')):
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask); loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            print("Pérdida no finita, saltando lote."); optimizer.zero_grad(set_to_none=True); continue
        scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step(); global_step += 1
        progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    model.eval()
    total_val_loss = 0.0
    val_batches = 0
    if len(val_loader) > 0:
        print("\n--- Iniciando Validación ---")
        with torch.inference_mode():
            for i, batch in enumerate(tqdm(val_loader, desc="Validando...")):
                input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["input_ids"].to(device)
                outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
                
                if outputs.loss is not None and torch.isfinite(outputs.loss):
                    batch_loss = outputs.loss.item()
                    total_val_loss += batch_loss
                    val_batches += 1
                else:
                    print(f"Advertencia: Pérdida no válida o nula en el lote de validación {i}")

        if val_batches > 0:
            avg_val_loss = total_val_loss / val_batches
            val_perplexity = torch.exp(torch.tensor(avg_val_loss))
            print(f"\n--- Resultados de Validación (Época {epoch}) ---")
            print(f"Pérdida de Validación Promedio: {avg_val_loss:.4f} | Perplejidad: {val_perplexity:.2f}")
            print(f"Pérdida total: {total_val_loss:.4f}, Lotes procesados: {val_batches}")

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
            print("\nAdvertencia: No se procesó ningún lote de validación válido. Saltando la lógica de validación.")
    else:
        print("El DataLoader de validación está vacío. Saltando la validación.")

    torch.save({"epoch": epoch + 1, "optimizer_state_dict": optimizer.state_dict(), "global_step": global_step, "best_val_loss": best_val_loss}, LOCAL_CHECKPOINT_PATH)
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Detención temprana en la época {epoch+1}.")
        break

print("Entrenamiento finalizado.")
model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
if os.path.exists(BEST_MODEL_PATH):
    print(f"Cargando el mejor modelo guardado desde '{BEST_MODEL_PATH}' para el guardado final.")
    model_to_save.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

print("Iniciando guardado en la carpeta 'output_model'...")
model_to_save.save_pretrained("output_model")
tokenizer.save_pretrained("output_model")
print("Guardado en 'output_model' completado.")

def chat_with_model(prompt_text, model, tokenizer, max_new_tokens=60, temperature=0.7, top_k=50):
    model.eval()
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n--- Probando la Generación del Modelo ---")
output_model_path = "output_model"
model_file_path = os.path.join(output_model_path, "model.safetensors")

if os.path.exists(model_file_path):
    print(f"Archivo '{model_file_path}' encontrado. Cargando el modelo para la generación...")
    inference_model = HRMText1.from_pretrained(output_model_path).to(device)
    if torch.__version__.startswith("2"): 
        print("Compilando el modelo de inferencia para una mayor rapidez...")
        inference_model = torch.compile(inference_model)
    try:
        prompts = ["42/6", "1000*2345", "530991+6051993"]
        for prompt in prompts:
            full_response = chat_with_model(prompt, inference_model, tokenizer, max_new_tokens=50)
            generated_part = full_response[len(prompt):].strip()
            print(f"\nPrompt: {prompt}\nRespuesta: {generated_part}")
    except Exception as e:
        print(f"El test de generación falló: {e}")
else:
    print(f"La carpeta '{output_model_path}' o el archivo '{model_file_path}' no fueron encontrados. No se pudo ejecutar el test de generación.")