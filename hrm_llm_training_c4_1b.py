# -*- coding: utf-8 -*-
"""
HRM-Text1 Training Script - ESCALADO A ~1B PARÁMETROS
VERSIÓN AMPLIADA: Configuración para ~1B parámetros con contexto extendido (2048/4096)
- Arquitectura multi-capa HRM apilada (24 capas)
- Rotary Position Embeddings (RoPE) para mejor extrapolación
- Optimizaciones de memoria y velocidad
- Configuración optimizada para modelos grandes
"""

import os, random, contextlib, math
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset
from transformers import T5Tokenizer, PreTrainedModel, PretrainedConfig, GenerationMixin, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from huggingface_hub import HfFolder

# Optimización específica para NVIDIA Ampere+
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    print("GPU NVIDIA compatible con TF32 detectada. Activando la precisión de matmul 'high'.")
    torch.set_float32_matmul_precision('high')

# Verificar si Flash Attention está disponible
try:
    import flash_attn
    HAS_FLASH_ATTN = True
    print("Flash Attention detectado. Se usará para optimización de velocidad.")
except ImportError:
    HAS_FLASH_ATTN = False
    print("Flash Attention no disponible. Usando atención estándar.")

# ==============================================================================
# --- ROTARY POSITION EMBEDDINGS (RoPE) ---
# ==============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding para mejor extrapolación de secuencias largas"""
    def __init__(self, dim, max_position_embeddings=4096, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1. / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for common sequence lengths
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        
    def _update_cos_sin_cache(self, seq_len, device):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]
    
    def forward(self, x, seq_len):
        self._update_cos_sin_cache(seq_len, x.device)
        return self._cos_cached[:, :seq_len, :, :], self._sin_cached[:, :seq_len, :, :]

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ==============================================================================
# --- DEFINICIÓN DEL MODELO ESCALADO ---
# ==============================================================================

class HRMText1Config(PretrainedConfig):
    model_type = "hrm_text1"
    
    def __init__(self, 
                 vocab_size=32128, 
                 block_size=2048,           # Aumentado para contexto extendido
                 n_embd=1536,               # Para ~1B params
                 n_head=24,                 # Más cabezas de atención
                 n_layers=24,               # NUEVO: múltiples capas HRM
                 d_ff=6144,                 # 4 * n_embd
                 dropout=0.1,
                 halt_max_steps=12,         # Más pasos para secuencias largas
                 ponder_loss_weight=1e-2,
                 halt_bias_init=-2.2,
                 use_rotary_embeddings=True, # NUEVO: RoPE
                 rotary_embedding_base=10000,
                 use_flash_attention=True,   # NUEVO: Flash Attention
                 gradient_checkpointing=True, # NUEVO: Para ahorrar memoria
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.halt_max_steps = halt_max_steps
        self.ponder_loss_weight = ponder_loss_weight
        self.halt_bias_init = halt_bias_init
        self.use_rotary_embeddings = use_rotary_embeddings
        self.rotary_embedding_base = rotary_embedding_base
        self.use_flash_attention = use_flash_attention
        self.gradient_checkpointing = gradient_checkpointing

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-6):
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

class OptimizedMultiHeadAttention(nn.Module):
    """Atención multi-cabeza optimizada con RoPE y Flash Attention opcional"""
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        self.use_flash_attention = config.use_flash_attention and HAS_FLASH_ATTN
        
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim, 
                max_position_embeddings=config.block_size,
                base=config.rotary_embedding_base
            )
        else:
            self.rotary_emb = None
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Proyecciones lineales
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Aplicar RoPE si está habilitado
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(x, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Usar Flash Attention si está disponible
        if self.use_flash_attention and x.device.type == 'cuda':
            # Para Flash Attention necesitamos reorganizar las dimensiones
            q = q.transpose(1, 2).contiguous()  # (batch, seq_len, n_head, head_dim)
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            try:
                from flash_attn import flash_attn_func
                attn_output = flash_attn_func(q, k, v, dropout_p=self.dropout.p if self.training else 0.0, causal=True)
            except:
                # Fallback a atención estándar
                attn_output = self._standard_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask, key_padding_mask)
                attn_output = attn_output.transpose(1, 2)
        else:
            attn_output = self._standard_attention(q, k, v, attn_mask, key_padding_mask)
            attn_output = attn_output.transpose(1, 2)  # (batch, seq_len, n_head, head_dim)
        
        # Reshape y proyección de salida
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.n_embd)
        return self.out_proj(attn_output)
    
    def _standard_attention(self, q, k, v, attn_mask=None, key_padding_mask=None):
        """Atención estándar escalada por productos punto"""
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        return torch.matmul(attn_weights, v)

class HRMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = OptimizedMultiHeadAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLUMuchPelu(config.n_embd, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pre-norm architecture
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        
        # MLP block
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class HRMInner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.H_module = HRMBlock(config)
        self.L_module = HRMBlock(config)
    
    def forward(self, z_H, z_L, attn_mask=None, key_padding_mask=None):
        z_L_input = z_L + z_H
        z_L_new = self.L_module(z_L_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        z_H_input = z_H + z_L_new
        z_H_new = self.H_module(z_H_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return z_H_new, z_L_new

class HRMText1(PreTrainedModel, GenerationMixin):
    config_class = HRMText1Config
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: HRMText1Config):
        super().__init__(config)
        self.config = config
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Usar RoPE en lugar de embeddings posicionales aprendidos
        if not config.use_rotary_embeddings:
            self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
            self.register_buffer("pos_ids", torch.arange(config.block_size).unsqueeze(0))
        else:
            self.pos_embeddings = None
            self.pos_ids = None
        
        # Apilar múltiples capas HRM
        self.layers = nn.ModuleList([
            HRMInner(config) for _ in range(config.n_layers)
        ])
        
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Un halt_head por capa para control más fino
        self.halt_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, 1), 
                nn.Sigmoid()
            ) for _ in range(config.n_layers)
        ])
        
        # Inicializar bias de halt
        with torch.no_grad():
            for halt_head in self.halt_heads:
                halt_head[0].bias.fill_(config.halt_bias_init)
        
        # Compartir pesos entre token embeddings y lm_head
        self.lm_head.weight = self.token_embeddings.weight
        
        # Habilitar gradient checkpointing si está configurado
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()
    
    def _set_gradient_checkpointing(self, module, value=False):
        """Para compatibilidad con transformers"""
        if isinstance(module, HRMText1):
            module.gradient_checkpointing = value
    
    def forward(self, input_ids, labels=None, attention_mask=None, past_key_values=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        z_L = self.token_embeddings(input_ids)
        
        # Position embeddings (solo si no usamos RoPE)
        if self.pos_embeddings is not None:
            z_L = z_L + self.pos_embeddings(self.pos_ids[:, :seq_len])
        
        # Inicializar z_H
        z_H = torch.zeros_like(z_L)
        
        # Máscaras de atención
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        
        # Variables para el mecanismo de halt adaptativo
        remainders = torch.ones((batch_size, seq_len), device=device)
        total_z_H = torch.zeros_like(z_H)
        n_updates = torch.zeros((batch_size, seq_len), device=device)
        eps = 1e-6
        
        # Procesamiento por capas con halt adaptativo
        for layer_idx, (layer, halt_head) in enumerate(zip(self.layers, self.halt_heads)):
            for step in range(self.config.halt_max_steps):
                # Calcular probabilidad de halt para esta capa
                p_halt = halt_head(z_H).squeeze(-1).clamp(eps, 1 - eps)
                is_last_step = step == (self.config.halt_max_steps - 1)
                halt_now_prob = p_halt if not is_last_step else torch.ones_like(p_halt)
                
                # Contribución ponderada
                contrib = remainders * halt_now_prob
                total_z_H = total_z_H + contrib.unsqueeze(-1) * z_H
                
                if is_last_step:
                    break
                
                # Actualizar remainders y contar updates
                remainders = remainders * (1 - p_halt)
                n_updates = n_updates + 1
                
                # Early stopping si todos los tokens han decidido parar
                if torch.all(remainders < eps):
                    break
                
                # Aplicar la capa HRM con gradient checkpointing si está habilitado
                if self.gradient_checkpointing and self.training:
                    z_H, z_L = torch.utils.checkpoint.checkpoint(
                        layer, z_H, z_L, causal_mask, key_padding_mask, use_reentrant=False
                    )
                else:
                    z_H, z_L = layer(z_H, z_L, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
            
            # Reiniciar para la siguiente capa (opcional, dependiendo del diseño)
            remainders = torch.ones((batch_size, seq_len), device=device)
        
        # Normalización final y proyección
        total_z_H = self.final_norm(total_z_H)
        logits = self.lm_head(total_z_H)
        
        loss = None
        if labels is not None:
            # Calcular pérdida de lenguaje
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # Pérdida de ponderación (ponder loss)
            ponder_loss = torch.mean(n_updates)
            
            # Pérdida total
            loss = lm_loss + self.config.ponder_loss_weight * ponder_loss
        
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", torch.ones_like(input_ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

# ==============================================================================
# --- CONFIGURACIÓN DEL SCRIPT PARA ~1B PARÁMETROS ---
# ==============================================================================

# Dataset configuration
DATASET_SUBSET_PERCENT = 10  # Aumentado para más datos de entrenamiento
DATASET_NAME = "allenai/c4"
DATASET_CONFIG = "multilingual"

HF_REPO_ID = "dreamwar/HRM-Text1-C4-1B"
SEED = 42
NUM_EPOCHS = 3
BLOCK_SIZE = 8192  # Contexto extendido

# Configuración de entrenamiento para modelo grande
BATCH_SIZE = 8           # Reducido significativamente para manejar 1B parámetros
GRAD_ACCUM_STEPS = 32    # Aumentado para batch efectivo de 256
EVAL_STEPS = 1000        # Evaluar cada 1000 pasos

# Learning rate schedule optimizado para modelos grandes
LEARNING_RATE_MAX = 2e-4  # Reducido para estabilidad
LEARNING_RATE_MIN = 1e-6
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1        # 10% de warmup

# Optimizaciones
MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 3
USE_GRADIENT_CHECKPOINTING = True

# Configuración del modelo para ~1B parámetros
MODEL_PARAMS = {
    "n_embd": 1536,                    # Para ~1B params
    "n_head": 24,                      # Más cabezas de atención
    "n_layers": 24,                    # 24 capas HRM apiladas
    "d_ff": 6144,                      # 4 * n_embd
    "dropout": 0.1,
    "halt_max_steps": 12,              # Más pasos para secuencias largas
    "ponder_loss_weight": 1e-2,
    "halt_bias_init": -2.2,
    "use_rotary_embeddings": True,     # RoPE para mejor extrapolación
    "use_flash_attention": True,       # Flash Attention si está disponible
    "gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
}

T5_TOKENIZER_REPO = "t5-small"
OUTPUT_DIR = "hrm_text1_c4_1b_output"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.bin")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")

# ==============================================================================
# --- INICIO DEL SCRIPT ---
# ==============================================================================

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo detectado: {device}")

# Verificar memoria disponible
if torch.cuda.is_available():
    print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()

try:
    HF_TOKEN = os.environ['HF_TOKEN']
    HfFolder.save_token(HF_TOKEN)
    print("Hugging Face token loaded.")
except Exception:
    print("HF_TOKEN secret not found.")
    HF_TOKEN = None

print("Loading tokenizer (T5 slow)...")
tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_REPO, use_fast=False, legacy=False)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

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
    texts = [str(text) + tokenizer.eos_token for text in examples["text"] 
             if isinstance(text, str) and len(text) > 100]  # Filtro más estricto para calidad
    return tokenizer(texts, truncation=True, max_length=BLOCK_SIZE, padding="max_length", add_special_tokens=False)

print("Applying tokenization function (on-the-fly)...")
tokenized_splits = {}
for split_name in ["train", "validation"]:
    tokenized_splits[split_name] = raw_datasets[split_name].map(
        tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"]
    ).with_format("torch")

num_workers = min(os.cpu_count() or 2, 4)  # Reducido para ahorrar memoria
train_loader = DataLoader(tokenized_splits["train"], batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(tokenized_splits["validation"], batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True)

# Crear modelo
config = HRMText1Config(vocab_size=len(tokenizer), block_size=BLOCK_SIZE, **MODEL_PARAMS)
model = HRMText1(config).to(device)

# Contar parámetros
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Número de parámetros del modelo: {total_params:,}")
print(f"Estimación de VRAM necesaria: {total_params * 4 / 1e9:.1f} GB (solo parámetros)")

# Optimizador con configuración para modelos grandes
optimizer = AdamW(
    model.parameters(), 
    lr=LEARNING_RATE_MAX, 
    weight_decay=WEIGHT_DECAY, 
    betas=(0.9, 0.95),
    eps=1e-8
)

# Calcular pasos de entrenamiento
num_training_steps = (num_train_samples // (BATCH_SIZE * GRAD_ACCUM_STEPS)) * NUM_EPOCHS
num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
print(f"Total de pasos de entrenamiento: {num_training_steps}")
print(f"Pasos de warmup: {num_warmup_steps}")

# Scheduler con warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Mixed precision scaler
scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and device.type == 'cuda'))

# Checkpoint loading
start_epoch = 0
start_step = 0
best_val_loss = float('inf')
patience_counter = 0
CHECKPOINT_STEPS = 1000

if os.path.exists(CHECKPOINT_PATH):
    print(f"--- Reanudando entrenamiento desde el checkpoint: {CHECKPOINT_PATH} ---")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    start_step = checkpoint.get('step', 0)
    best_val_loss = checkpoint['best_val_loss']
    patience_counter = checkpoint.get('patience_counter', 0)
    
    # VERIFICAR SI EL DATASET CAMBIÓ Y REAJUSTAR SCHEDULER
    checkpoint_training_steps = checkpoint.get('num_training_steps', 0)
    if checkpoint_training_steps != num_training_steps:
        print(f"Dataset cambió. Reajustando scheduler: {checkpoint_training_steps} -> {num_training_steps}")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        # Ajustar el paso actual proporcionalmente
        current_progress = start_step / checkpoint_training_steps if checkpoint_training_steps > 0 else 0
        new_step = int(current_progress * num_training_steps)
        for _ in range(new_step):
            scheduler.step()
        print(f"Scheduler reajustado. Progreso: {current_progress:.2%}, nuevo paso: {new_step}")
    
    print(f"Checkpoint cargado. Reanudando desde la época {start_epoch + 1}, paso {start_step}.")
    print(f"Mejor pérdida de validación hasta ahora: {best_val_loss:.4f}")
else:
    print("--- No se encontró checkpoint. Empezando entrenamiento desde cero. ---")

# Compilar modelo si está disponible
if torch.__version__.startswith("2") and hasattr(torch, 'compile'):
    print("Compilando el modelo con torch.compile()...")
    model = torch.compile(model)

# ==============================================================================
# --- BUCLE DE ENTRENAMIENTO ---
# ==============================================================================

global_step = start_step

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    progress = tqdm(train_loader, desc=f"Época {epoch+1}/{NUM_EPOCHS}")
    
    for i, batch in enumerate(progress):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = input_ids.clone()

        # Mixed precision forward pass
        with torch.amp.autocast(
            device_type=device.type, 
            dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, 
            enabled=MIXED_PRECISION
        ):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
        
        # Backward pass
        if loss is not None and torch.isfinite(loss):
            scaler.scale(loss).backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                # Gradient clipping y update
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
                # Checkpoint periódico
                if global_step % CHECKPOINT_STEPS == 0:
                    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
                    print(f"\nGuardando checkpoint en paso {global_step}...")
                    torch.save({
                        'epoch': epoch,
                        'step': global_step,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'patience_counter': patience_counter,
                        'num_training_steps': num_training_steps,  # Guardar para verificar cambios
                    }, CHECKPOINT_PATH)
                    print(f"Checkpoint guardado en {CHECKPOINT_PATH}")
            
            # Actualizar progress bar
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else LEARNING_RATE_MAX
            progress.set_postfix({
                "loss": f"{loss.item()*GRAD_ACCUM_STEPS:.4f}", 
                "lr": f"{current_lr:.2e}", 
                "step": global_step
            })

    # Validación al final de cada época
    model.eval()
    total_val_loss, val_batches = 0.0, 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validando..."):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = input_ids.clone()
            
            with torch.amp.autocast(
                device_type=device.type, 
                dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, 
                enabled=MIXED_PRECISION
            ):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            if outputs.loss is not None and torch.isfinite(outputs.loss):
                total_val_loss += outputs.loss.item()
                val_batches += 1
    
    if val_batches > 0:
        avg_val_loss = total_val_loss / val_batches
        print(f"Época {epoch+1}: Pérdida de Validación = {avg_val_loss:.4f}")
        
        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        # Guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Nueva mejor pérdida de validación. Guardando modelo en {BEST_MODEL_PATH}")
            torch.save(model_to_save.state_dict(), BEST_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Checkpoint al final de época
        print(f"Guardando checkpoint al final de época {epoch+1}...")
        torch.save({
            'epoch': epoch + 1,
            'step': global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'num_training_steps': num_training_steps,  # Guardar para verificar cambios
        }, CHECKPOINT_PATH)
    
    # Early stopping
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("Detención temprana por falta de mejora en la validación.")
        break

print("Entrenamiento finalizado.")

# Guardar modelo final
model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

if os.path.exists(BEST_MODEL_PATH):
    print(f"Cargando el mejor modelo desde '{BEST_MODEL_PATH}' para el guardado final.")
    model_to_save.load_state_dict(torch.load(BEST_MODEL_PATH))

model_to_save.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Modelo y tokenizador guardados en '{OUTPUT_DIR}'")

# ==============================================================================
# --- FUNCIÓN DE CHAT Y PRUEBAS ---
# ==============================================================================

def chat_with_model(prompt_text, model, tokenizer, max_new_tokens=100, temperature=0.7, top_k=50):
    model.eval()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type, 
        dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, 
        enabled=MIXED_PRECISION
    ):
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_k=top_k, 
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False  # Deshabilitar cache para ahorrar memoria
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n--- Probando la Generación del Modelo Final ---")
try:
    inference_model = HRMText1.from_pretrained(OUTPUT_DIR).to(device)
    if torch.__version__.startswith("2") and hasattr(torch, 'compile'):
        inference_model = torch.compile(inference_model)
    
    prompts = [
        "The cat sat on the", 
        "Artificial intelligence is a field that", 
        "To be, or not to be, that is the question:",
        "In a world where technology advances rapidly,",
        "The future of humanity depends on"
    ]
    
    for prompt in prompts:
        response = chat_with_model(prompt, inference_model, tokenizer)
        print(f"\nPrompt: {prompt}\nRespuesta: {response}")
        
except Exception as e:
    print(f"El test de generación falló: {e}")

print(f"\n=== RESUMEN DEL ENTRENAMIENTO ===")
print(f"Parámetros del modelo: {total_params:,}")
print(f"Contexto máximo: {BLOCK_SIZE}")
print(f"Capas HRM: {MODEL_PARAMS['n_layers']}")
print(f"Dimensión del modelo: {MODEL_PARAMS['n_embd']}")
print(f"Cabezas de atención: {MODEL_PARAMS['n_head']}")
print(f"Mejor pérdida de validación: {best_val_loss:.4f}")
print(f"Modelo guardado en: {OUTPUT_DIR}")