# -*- coding: utf-8 -*-
"""
HRM-Text1 Training Script - ADAPTADO PARA EL DATASET C4
VERSIÓN FINAL: Optimizado con streaming, subconjuntos de datos y generación.
"""

import os, random, contextlib
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset
from transformers import T5Tokenizer, PreTrainedModel, PretrainedConfig, GenerationMixin
from tqdm.auto import tqdm

from huggingface_hub import HfFolder

# Optimización específica para NVIDIA Ampere+
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    print("GPU NVIDIA compatible con TF32 detectada. Activando la precisión de matmul 'high'.")
    torch.set_float32_matmul_precision('high')

# ==============================================================================
# --- DEFINICIÓN DEL MODELO ---
# ==============================================================================

class HRMText1Config(PretrainedConfig):
    model_type = "hrm_text1"
    def __init__(self, vocab_size=32128, block_size=512, n_embd=512, n_head=8, d_ff=2048, dropout=0.1, halt_max_steps=8, ponder_loss_weight=1e-2, halt_bias_init=-2.2, **kwargs):
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
        with torch.no_grad(): self.halt_head[0].bias.fill_(config.halt_bias_init)

    def forward(self, input_ids, labels=None, attention_mask=None, past_key_values=None, **kwargs):
        batch_size, seq_len = input_ids.shape; device = input_ids.device
        z_L = self.token_embeddings(input_ids) + self.pos_embeddings(self.pos_ids[:, :seq_len])
        z_H = torch.zeros_like(z_L)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        remainders, total_z_H, n_updates = torch.ones((batch_size, seq_len), device=device), torch.zeros_like(z_H), torch.zeros((batch_size, seq_len), device=device)
        eps = 1e-6
        
        for step in range(self.config.halt_max_steps):
            p_halt = self.halt_head(z_H).squeeze(-1).clamp(eps, 1 - eps)
            is_last_step = step == (self.config.halt_max_steps - 1)
            halt_now_prob = p_halt if not is_last_step else torch.ones_like(p_halt)
            
            contrib = remainders * halt_now_prob
            total_z_H += contrib.unsqueeze(-1) * z_H
            
            if is_last_step: break
                
            remainders *= (1 - p_halt)
            n_updates += 1 # Contamos cada paso de "ponderación"
            
            if torch.all(remainders < eps): break
            
            z_H, z_L = self.inner_model(z_H, z_L, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        logits, loss = self.lm_head(total_z_H), None
        if labels is not None:
            shift_logits, shift_labels = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            ponder_loss = torch.mean(n_updates * (1-remainders)) # Penaliza por más pasos
            loss = lm_loss + self.config.ponder_loss_weight * ponder_loss
                
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", torch.ones_like(input_ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

# ==============================================================================
# --- CONFIGURACIÓN DEL SCRIPT ---
# ==============================================================================
# MEJORA: Define el porcentaje del dataset a utilizar (1.0 para prueba rápida, 100.0 para completo)
DATASET_SUBSET_PERCENT = 30.0

DATASET_NAME = "allenai/c4"
DATASET_CONFIG = "en.noblocklist"

HF_REPO_ID = "dreamwar/HRM-Text1-C4"
SEED = 42
NUM_EPOCHS = 2
BLOCK_SIZE = 512
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2 # Grad_accum * batch_size = effective_batch_size (ej. 2 * 64 = 128)

LEARNING_RATE_MAX, LEARNING_RATE_MIN, WEIGHT_DECAY = 3e-4, 1e-5, 0.05
MIXED_PRECISION, EARLY_STOPPING_PATIENCE = True, 2
MODEL_PARAMS = {"n_embd": 512, "n_head": 8, "d_ff": 2048, "dropout": 0.1, "halt_max_steps": 8, "ponder_loss_weight": 1e-2, "halt_bias_init": -2.2}

T5_TOKENIZER_REPO = "t5-small"
BEST_MODEL_PATH = "best_model.bin"

# ==============================================================================
# --- INICIO DEL SCRIPT ---
# ==============================================================================

def set_seed(seed: int):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo detectado: {device}")

try:
    HF_TOKEN = os.environ['HF_TOKEN']; HfFolder.save_token(HF_TOKEN); print("Hugging Face token loaded.")
except Exception:
    print("HF_TOKEN secret not found."); HF_TOKEN = None

print("Loading tokenizer (T5 slow)...")
tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_REPO, use_fast=False, legacy=False)
if tokenizer.pad_token is None: tokenizer.add_special_tokens({"pad_token": "<pad>"})
print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

# --- MEJORA: Carga de datos optimizada con streaming y subconjuntos ---
print(f"Loading dataset '{DATASET_NAME}' in streaming mode.")
raw_datasets = load_dataset(DATASET_NAME, DATASET_CONFIG, streaming=True)

TOTAL_TRAIN_SAMPLES, TOTAL_VAL_SAMPLES = 364_868_892, 364_608
num_train_samples = int(TOTAL_TRAIN_SAMPLES * (DATASET_SUBSET_PERCENT / 100.0))
num_val_samples = int(TOTAL_VAL_SAMPLES * (DATASET_SUBSET_PERCENT / 100.0))

print(f"\n!!! USANDO UN SUBCONJUNTO DEL {DATASET_SUBSET_PERCENT}% DEL DATASET !!!")
print(f"Tomando aprox. {num_train_samples:,} ejemplos de entrenamiento.")
print(f"Tomando aprox. {num_val_samples:,} ejemplos de validación.\n")

raw_datasets["train"] = raw_datasets["train"].take(num_train_samples).shuffle(seed=SEED, buffer_size=10_000)
raw_datasets["validation"] = raw_datasets["validation"].take(num_val_samples)

def tokenize_function(examples):
    texts = [str(text) + tokenizer.eos_token for text in examples["text"] if isinstance(text, str) and len(text) > 50]
    return tokenizer(texts, truncation=True, max_length=BLOCK_SIZE, padding="max_length")

print("Applying tokenization function (on-the-fly)...")
tokenized_splits = {}
for split_name in ["train", "validation"]:
    tokenized_splits[split_name] = raw_datasets[split_name].map(
        tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"]
    ).with_format("torch")

num_workers = min(os.cpu_count() or 2, 8)
train_loader = DataLoader(tokenized_splits["train"], batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(tokenized_splits["validation"], batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True)

# --- Fin de la sección de datos optimizada ---

config = HRMText1Config(vocab_size=len(tokenizer), block_size=BLOCK_SIZE, **MODEL_PARAMS)
model = HRMText1(config).to(device)
print(f"Número de parámetros del modelo: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

if torch.__version__.startswith("2"):
    print("Compilando el modelo con torch.compile()...")
    model = torch.compile(model)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE_MAX, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

# MEJORA: Cálculo dinámico de pasos
num_training_steps = (num_train_samples // (BATCH_SIZE * GRAD_ACCUM_STEPS)) * NUM_EPOCHS
print(f"Total de pasos de entrenamiento calculados: {num_training_steps}")
    
scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=LEARNING_RATE_MIN)
scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and device.type == 'cuda'))

best_val_loss, patience_counter = float('inf'), 0
for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    progress = tqdm(train_loader, desc=f"Época {epoch+1}/{NUM_EPOCHS}")
    for i, batch in enumerate(progress):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = input_ids.clone()

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=MIXED_PRECISION):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
        
        if loss is not None and torch.isfinite(loss):
            scaler.scale(loss).backward()
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            progress.set_postfix({"loss": f"{loss.item()*GRAD_ACCUM_STEPS:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    model.eval()
    total_val_loss, val_batches = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validando..."):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = input_ids.clone()
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=MIXED_PRECISION):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            if outputs.loss is not None and torch.isfinite(outputs.loss):
                total_val_loss += outputs.loss.item()
                val_batches += 1
    
    if val_batches > 0:
        avg_val_loss = total_val_loss / val_batches
        print(f"Época {epoch+1}: Pérdida de Validación = {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Nueva mejor pérdida de validación. Guardando modelo en {BEST_MODEL_PATH}")
            # Desempaquetar el modelo si está compilado
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(model_to_save.state_dict(), BEST_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
    
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("Detención temprana por falta de mejora en la validación.")
        break

print("Entrenamiento finalizado.")

# --- Guardado y Generación ---
output_dir = "hrm_text1_c4_output"
os.makedirs(output_dir, exist_ok=True)
model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

if os.path.exists(BEST_MODEL_PATH):
    print(f"Cargando el mejor modelo desde '{BEST_MODEL_PATH}' para el guardado final.")
    model_to_save.load_state_dict(torch.load(BEST_MODEL_PATH))

model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Modelo y tokenizador guardados en '{output_dir}'")

def chat_with_model(prompt_text, model, tokenizer, max_new_tokens=60, temperature=0.7, top_k=50):
    model.eval()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=MIXED_PRECISION):
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_k=top_k, 
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n--- Probando la Generación del Modelo Final ---")
try:
    inference_model = HRMText1.from_pretrained(output_dir).to(device)
    if torch.__version__.startswith("2"): inference_model = torch.compile(inference_model)
    
    prompts = ["The cat sat on the", "Artificial intelligence is a field that", "To be, or not to be, that is the question:"]
    for prompt in prompts:
        response = chat_with_model(prompt, inference_model, tokenizer)
        print(f"\nPrompt: {prompt}\nRespuesta: {response}")
except Exception as e:
    print(f"El test de generación falló: {e}")