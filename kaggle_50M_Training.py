# -*- coding: utf-8 -*-
"""
HRM-Models Training Script - OPTIMIZADO PARA KAGGLE
VERSIÓN SMALL: ~50M parámetros optimizado para GPUs Kaggle (T4/P100)
- Configuración automática para entorno Kaggle
- Gestión optimizada de memoria y datasets
- Checkpoints automáticos en /kaggle/working
- Compatible con time limits de Kaggle (12 horas)
- Dataset: carlmcbrideellis/llm-mistral-7b-instruct-texts
"""

import os, random, math, time, gc, json
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Imports adicionales necesarios
from tqdm import tqdm

# Configurar para Kaggle automáticamente
IN_KAGGLE = os.path.exists('/kaggle')
if IN_KAGGLE:
    os.environ['KAGGLE_KERNEL_RUN_TYPE'] = 'Interactive'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    print("🔍 Entorno Kaggle detectado - Configuración automática aplicada")
else:
    print("🔍 Entorno local detectado")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

# Verificar GPU disponible en Kaggle y configurar multi-GPU
if IN_KAGGLE and torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    total_vram = 0
    
    print(f"🔥 GPUs Kaggle detectadas: {num_gpus}")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        total_vram += memory_gb
        print(f"   GPU {i}: {gpu_name} ({memory_gb:.1f} GB VRAM)")
    
    print(f"💾 VRAM total: {total_vram:.1f} GB")
    
    # Optimizaciones específicas para GPUs de Kaggle
    if "T4" in torch.cuda.get_device_name(0):
        print("⚡ Optimizando para Tesla T4s...")
        torch.backends.cuda.matmul.allow_tf32 = True
        if num_gpus > 1:
            print(f"🚀 Multi-GPU T4 setup detectado ({num_gpus}x T4)")
    elif "P100" in torch.cuda.get_device_name(0):
        print("⚡ Optimizando para Tesla P100s...")
        if num_gpus > 1:
            print(f"🚀 Multi-GPU P100 setup detectado ({num_gpus}x P100)")
    
    # Limpiar cache de todas las GPUs
    for i in range(num_gpus):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()

# Clases base necesarias para compatibilidad (DEFINIR ANTES DE SU USO)
class SimpleConfig:
    """Configuración base simple para reemplazar PretrainedConfig"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class SimplePreTrainedModel(nn.Module):
    """Modelo base simple para reemplazar PreTrainedModel"""
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, do_sample=True, pad_token_id=0, **kwargs):
        """Generación simple de texto"""
        self.eval()
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids)
                logits = outputs.logits[:, -1, :] / temperature
                
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == pad_token_id:
                    break
        
        return input_ids

class SimpleModelOutput:
    """Output simple para reemplazar transformers ModelOutput"""
    def __init__(self, loss=None, logits=None, **kwargs):
        self.loss = loss
        self.logits = logits
        for key, value in kwargs.items():
            setattr(self, key, value)

# Tokenizer alternativo SIN Hugging Face para Kaggle
TRANSFORMERS_AVAILABLE = False
print("🔧 Usando implementación standalone para Kaggle")

# Tokenizer básico personalizado para Kaggle
import re
from collections import Counter

class SimpleTokenizer:
    """Tokenizer simple sin dependencias de HuggingFace para Kaggle"""
    
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<s>': 2,
            '</s>': 3,
            '<mask>': 4
        }
        
        # Inicializar con tokens especiales
        for token, token_id in self.special_tokens.items():
            self.word_to_id[token] = token_id
            self.id_to_word[token_id] = token
        
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.mask_token = '<mask>'
        
        self.pad_token_id = self.special_tokens['<pad>']
        self.unk_token_id = self.special_tokens['<unk>']
        self.bos_token_id = self.special_tokens['<s>']
        self.eos_token_id = self.special_tokens['</s>']
        self.mask_token_id = self.special_tokens['<mask>']
        
        self._built = False
    
    def _tokenize_text(self, text):
        """Tokenización básica por palabras y caracteres"""
        # Limpiar y normalizar
        text = text.lower().strip()
        # Separar por espacios y puntuación
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def build_vocab(self, texts):
        """Construir vocabulario desde textos"""
        print(f"🔧 Construyendo vocabulario desde {len(texts)} textos...")
        
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize_text(text)
            word_counts.update(tokens)
        
        # Tomar las palabras más frecuentes
        most_common = word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        # Agregar al vocabulario
        current_id = len(self.special_tokens)
        for word, count in most_common:
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
        
        self._built = True
        print(f"✅ Vocabulario construido: {len(self.word_to_id)} tokens")
        return self
    
    def encode(self, text, max_length=None, truncation=True, padding=False, return_tensors=None):
        """Codificar texto a tokens"""
        if not self._built:
            raise ValueError("Vocabulario no construido. Llama build_vocab() primero.")
        
        tokens = self._tokenize_text(text)
        token_ids = [self.word_to_id.get(token, self.unk_token_id) for token in tokens]
        
        # Truncar si es necesario
        if max_length and truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Padding si es necesario
        if padding and max_length:
            if len(token_ids) < max_length:
                token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
        
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
        
        return token_ids
    
    def __call__(self, text, **kwargs):
        """Hacer el tokenizer callable como Transformers"""
        return self.encode(text, **kwargs)
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decodificar tokens a texto"""
        if hasattr(token_ids, 'tolist'):  # Es un tensor
            if len(token_ids.shape) > 1:
                token_ids = token_ids.squeeze()
            token_ids = token_ids.tolist()
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                token = self.id_to_word[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        return " ".join(tokens)
    
    def __len__(self):
        return len(self.word_to_id)
    
    def save_pretrained(self, save_directory):
        """Guardar tokenizer"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w') as f:
            json.dump({
                'word_to_id': self.word_to_id,
                'id_to_word': {str(k): v for k, v in self.id_to_word.items()},
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens
            }, f, indent=2)
        
        print(f"💾 Tokenizer guardado en: {save_directory}")

# Scheduler simple sin Transformers
class SimpleCosineScheduler:
    """Scheduler coseno simple sin dependencias de Transformers"""
    
    def __init__(self, optimizer, num_warmup_steps, num_training_steps):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self):
        self.current_step += 1
        
        if self.current_step < self.num_warmup_steps:
            # Warmup lineal
            lr = self.base_lr * (self.current_step / self.num_warmup_steps)
        else:
            # Coseno decay
            progress = (self.current_step - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

# Install kagglehub if not available
try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    KAGGLEHUB_AVAILABLE = True
    print("✅ KaggleHub disponible")
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("❌ KaggleHub no disponible - instalando...")
    if IN_KAGGLE:
        os.system("pip install kagglehub[hf-datasets]")
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        KAGGLEHUB_AVAILABLE = True
    else:
        print("⚠️ En entorno local - instala: pip install kagglehub[hf-datasets]")

try:
    from huggingface_hub import HfFolder, HfApi, login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  huggingface_hub no disponible")

# Configuración específica para Kaggle multi-GPU
KAGGLE_CONFIG = {
    'max_train_hours': 11.5,
    'checkpoint_frequency': 150,  # Más frecuente para multi-GPU
    'memory_cleanup_steps': 30,   # Más frecuente para multi-GPU
    'dataset_cache_dir': '/kaggle/tmp/datasets' if IN_KAGGLE else './cache',
    'output_dir': '/kaggle/working' if IN_KAGGLE else './output',
    'multi_gpu_enabled': False,  # Se configurará automáticamente
    'force_single_gpu': True,    # Forzar single-GPU (multi-GPU tiene problemas de device)
}

# ==============================================================================
# --- CONFIGURACIÓN DEL MODELO OPTIMIZADA PARA KAGGLE ---
# ==============================================================================

SEED = 42
BLOCK_SIZE = 768  # Optimizado para T4/P100
BATCH_SIZE = 6    # Optimizado para single-GPU T4/P100 (aumentado de 4)
GRAD_ACCUM_STEPS = 6  # Batch efectivo de 36 (balanceado)
LEARNING_RATE_MAX = 8e-4
LEARNING_RATE_MIN = 1e-6
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
NUM_EPOCHS = 2  # Reducido para tiempo de Kaggle
MIXED_PRECISION = True
EVAL_STEPS = 100
CHECKPOINT_STEPS = 1000  # Guardar checkpoint cada 1000 pasos

# Configuración del modelo (50M parámetros)
MODEL_PARAMS = {
    "n_embd": 384,
    "n_head": 12,
    "n_layers": 8,
    "d_ff": 1536,
    "dropout": 0.1,
    "halt_max_steps": 6,               # Pasos optimizados para modelo 50M (desde original)
    "ponder_loss_weight": 1e-2,
    "halt_bias_init": -2.2,            # Bias inicial desde original
    "use_rotary_embeddings": True,     # RoPE para mejor extrapolación
    "use_flash_attention": False,      # Deshabilitado para compatibilidad Kaggle
    "gradient_checkpointing": True,
    "h_update_period": 3,              # H-module se actualiza cada 3 pasos para 50M (desde original)
}

# Dataset configuration - SIN KAGGLEHUB para evitar problemas de conexión
KAGGLE_DATASET_PATH = "/kaggle/input"  # Cambiar por el path real del dataset
DATASET_NAME = "mistral-7b-instruct-texts"  # Nombre del dataset
T5_TOKENIZER_REPO = "t5-small"  # Ya no se usa, mantenido por compatibilidad
HF_REPO_ID = "your-username/hrm-kaggle-50m"

# Configurar directorios
os.makedirs(KAGGLE_CONFIG['dataset_cache_dir'], exist_ok=True)
os.makedirs(KAGGLE_CONFIG['output_dir'], exist_ok=True)

OUTPUT_DIR = os.path.join(KAGGLE_CONFIG['output_dir'], 'hrm_model')
CHECKPOINT_PATH = os.path.join(KAGGLE_CONFIG['output_dir'], 'checkpoint.pth')
BEST_MODEL_PATH = os.path.join(KAGGLE_CONFIG['output_dir'], 'best_model.pth')

print(f"📁 Directorios configurados:")
print(f"   Output: {OUTPUT_DIR}")
print(f"   Checkpoints: {CHECKPOINT_PATH}")

# ==============================================================================
# --- FUNCIONES AUXILIARES PARA KAGGLE ---
# ==============================================================================

def kaggle_memory_cleanup():
    """Limpiar memoria específico para Kaggle"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def check_time_limit():
    """Verificar si se está acercando al límite de tiempo de Kaggle"""
    if not IN_KAGGLE:
        return False
    
    # En un entorno real, aquí verificarías el tiempo transcurrido
    # Por simplicidad, retornamos False
    return False

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ==============================================================================
# --- ROTARY POSITION EMBEDDINGS (RoPE) ---
# ==============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding optimizado para Kaggle y multi-GPU"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1. / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Pre-compute para evitar problemas multi-GPU
        self._precompute_cos_sin_cache(max_position_embeddings)
    
    def _precompute_cos_sin_cache(self, max_seq_len):
        """Pre-computar cos/sin para evitar problemas de device en multi-GPU"""
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()[None, :, None, :] 
        sin_cached = emb.sin()[None, :, None, :]
        
        # Registrar como buffers para multi-GPU
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)
    
    def forward(self, x, seq_len):
        # Usar cache pre-computado
        seq_len = min(seq_len, self.cos_cached.size(1))
        return (
            self.cos_cached[:, :seq_len, :, :].to(x.device),
            self.sin_cached[:, :seq_len, :, :].to(x.device)
        )

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ==============================================================================
# --- MODELO HRM OPTIMIZADO PARA KAGGLE ---
# ==============================================================================

class HRMKaggleConfig(SimpleConfig):
    model_type = "hrm_kaggle"
    
    def __init__(self, 
                 vocab_size=30000,  # Reducido para tokenizer personalizado
                 block_size=768,
                 n_embd=384,
                 n_head=12,
                 n_layers=8,
                 d_ff=1536,
                 dropout=0.1,
                 halt_max_steps=4,
                 ponder_loss_weight=1e-2,
                 halt_bias_init=-2.0,
                 use_rotary_embeddings=True,
                 gradient_checkpointing=True,
                 h_update_period=2,
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
        self.gradient_checkpointing = gradient_checkpointing
        self.h_update_period = h_update_period

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
    
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))

class SwiGLU(nn.Module):
    def __init__(self, n_embd, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(n_embd, d_ff, bias=False)
        self.w2 = nn.Linear(n_embd, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class OptimizedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        
        assert self.n_embd % self.n_head == 0
        
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim, 
                max_position_embeddings=config.block_size
            )
        else:
            self.rotary_emb = None
    
    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(x, seq_len)
            cos = cos.expand(q.shape[0], -1, q.shape[1], -1).transpose(1, 2)
            sin = sin.expand(q.shape[0], -1, q.shape[1], -1).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Atención estándar (más compatible con Kaggle)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        
        return self.out_proj(attn_output)

class HRMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = OptimizedAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config.n_embd, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attn_mask=None):
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class HRMInner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.H_module = HRMBlock(config)
        self.L_module = HRMBlock(config)
        self.config = config
        
        # Q-learning simplificado para Kaggle
        self.q_network = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.ReLU(),
            nn.Linear(config.n_embd // 4, 2)
        )
        
        self.convergence_threshold = 1e-3
        self.max_l_steps = config.halt_max_steps
        self.h_update_period = config.h_update_period
    
    def forward(self, z_H, z_L, step_count=0, attn_mask=None, training=True):
        batch_size, seq_len, d_model = z_H.shape
        
        is_h_update_step = (step_count % self.h_update_period) == 0
        
        if is_h_update_step:
            # Convergencia simplificada para Kaggle
            z_L_current = z_L
            for l_step in range(min(self.max_l_steps, 2)):  # Máximo 2 pasos para eficiencia
                z_L_input = z_L_current + z_H.detach()
                z_L_next = self.L_module(z_L_input, attn_mask=attn_mask)
                
                # Check simple de convergencia
                diff = torch.norm(z_L_next - z_L_current, p=2, dim=-1).mean()
                if diff < self.convergence_threshold:
                    break
                z_L_current = z_L_next
            
            # Actualizar H-module
            z_H_input = z_H + z_L_current
            z_H_new = self.H_module(z_H_input, attn_mask=attn_mask)
            z_L_new = torch.zeros_like(z_L)
            
            return z_H_new, z_L_new, {
                'h_updated': True,
                'l_steps': l_step + 1,
                'convergence_achieved': True
            }
        else:
            # Solo paso L
            z_L_input = z_L + z_H.detach()
            z_L_new = self.L_module(z_L_input, attn_mask=attn_mask)
            
            return z_H, z_L_new, {
                'h_updated': False,
                'l_steps': 1,
                'convergence_achieved': False
            }

class HRMKaggleModel(SimplePreTrainedModel):
    config_class = HRMKaggleConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: HRMKaggleConfig):
        super().__init__(config)
        self.config = config
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        
        if not config.use_rotary_embeddings:
            self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
            self.register_buffer("pos_ids", torch.arange(config.block_size).unsqueeze(0))
        else:
            self.pos_embeddings = None
            self.pos_ids = None
        
        self.layers = nn.ModuleList([
            HRMInner(config) for _ in range(config.n_layers)
        ])
        
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
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
        
        # Compartir pesos
        self.lm_head.weight = self.token_embeddings.weight
        
        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _enable_gradient_checkpointing(self):
        """Habilitar gradient checkpointing sin Transformers"""
        def make_checkpointed(module):
            def checkpointed_forward(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(
                    module._original_forward, *args, use_reentrant=False, **kwargs
                )
            
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
                module.forward = checkpointed_forward
        
        for layer in self.layers:
            make_checkpointed(layer)
    
    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        z_L = self.token_embeddings(input_ids)
        
        if self.pos_embeddings is not None:
            z_L = z_L + self.pos_embeddings(self.pos_ids[:, :seq_len])
        
        z_H = torch.zeros_like(z_L)
        
        # Máscara causal
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        
        # Variables ACT
        eps = 1e-6
        total_z_H = torch.zeros_like(z_H)
        n_updates = torch.zeros((batch_size, seq_len), device=device)
        
        step_count = 0
        
        for layer_idx, (layer, halt_head) in enumerate(zip(self.layers, self.halt_heads)):
            layer_remainders = torch.ones((batch_size, seq_len), device=device)
            layer_total_z_H = torch.zeros_like(z_H)
            layer_n_updates = torch.zeros((batch_size, seq_len), device=device)
            
            # Simplificado para Kaggle (máximo 3 pasos por capa)
            for step in range(min(self.config.halt_max_steps, 3)):
                z_H, z_L, hrm_info = layer(z_H, z_L, step_count=step_count,
                                         attn_mask=causal_mask, training=self.training)
                
                # ACT halt mechanism
                p_halt = halt_head(z_H).squeeze(-1).clamp(eps, 1 - eps)
                is_last_step = step == (min(self.config.halt_max_steps, 3) - 1)
                halt_now_prob = p_halt if not is_last_step else torch.ones_like(p_halt)
                
                contrib = layer_remainders * halt_now_prob
                layer_total_z_H = layer_total_z_H + contrib.unsqueeze(-1) * z_H
                layer_n_updates = layer_n_updates + contrib
                
                if is_last_step:
                    break
                
                layer_remainders = layer_remainders * (1 - p_halt)
                step_count += 1
                
                if torch.all(layer_remainders < eps):
                    break
            
            z_H = layer_total_z_H
            total_z_H = total_z_H + z_H
            n_updates = n_updates + layer_n_updates
        
        total_z_H = self.final_norm(total_z_H)
        logits = self.lm_head(total_z_H)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            ponder_loss = torch.mean(n_updates)
            loss = lm_loss + self.config.ponder_loss_weight * ponder_loss
        
        return SimpleModelOutput(loss=loss, logits=logits)

# ==============================================================================
# --- FUNCIONES DE CHECKPOINT PARA KAGGLE ---
# ==============================================================================

def save_checkpoint_kaggle(model, optimizer, scheduler, scaler, epoch, step, best_val_loss, 
                          checkpoint_path, tokenizer=None):
    """Guardar checkpoint optimizado para Kaggle"""
    try:
        model_to_save = model
        if hasattr(model, 'module'):
            model_to_save = model.module
        
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'model_config': model_to_save.config.to_dict() if hasattr(model_to_save.config, 'to_dict') else vars(model_to_save.config),
            'timestamp': time.time(),
        }
        
        # Guardar atómicamente
        temp_path = checkpoint_path + '.tmp'
        torch.save(checkpoint_data, temp_path)
        os.rename(temp_path, checkpoint_path)
        
        print(f"✅ Checkpoint guardado: paso {step}")
        return True
        
    except Exception as e:
        print(f"❌ Error guardando checkpoint: {e}")
        return False

def load_checkpoint_kaggle(checkpoint_path, model, optimizer, scheduler, scaler, device):
    """Cargar checkpoint en Kaggle"""
    if not os.path.exists(checkpoint_path):
        print("No se encontró checkpoint previo.")
        return False, 0, 0, float('inf')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model_to_load = model
        if hasattr(model, 'module'):
            model_to_load = model.module
        
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        best_val_loss = checkpoint['best_val_loss']
        
        print(f"✅ Checkpoint cargado: época {epoch+1}, paso {step}")
        return True, epoch, step, best_val_loss
        
    except Exception as e:
        print(f"❌ Error cargando checkpoint: {e}")
        return False, 0, 0, float('inf')

def save_final_model_kaggle(model, tokenizer, output_dir):
    """Guardar modelo final para Kaggle"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        model_to_save = model
        if hasattr(model, 'module'):
            model_to_save = model.module
        
        # Guardar config.json
        config_dict = model_to_save.config.to_dict() if hasattr(model_to_save.config, 'to_dict') else vars(model_to_save.config)
        import json
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Guardar modelo
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Guardar tokenizer
        if tokenizer:
            tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Modelo guardado en: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error guardando modelo: {e}")
        return False

# ==============================================================================
# --- DATASET KAGGLE CON HUGGING FACE ---
# ==============================================================================

class KaggleHFDataset(Dataset):
    """Dataset adapter para el dataset de Kaggle usando HuggingFace"""
    
    def __init__(self, hf_dataset, tokenizer, max_length=768, train_split=0.9, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"🔍 Procesando dataset HuggingFace...")
        print(f"📊 Dataset info: {hf_dataset}")
        
        # Acceder a los datos del dataset
        if hasattr(hf_dataset, 'data'):
            data = hf_dataset.data
        else:
            data = hf_dataset
            
        # Convertir a lista si es necesario
        if hasattr(data, 'to_pandas'):
            df = data.to_pandas()
            data_list = df.to_dict('records')
        elif hasattr(data, '__iter__'):
            data_list = list(data)
        else:
            raise ValueError(f"No se pudo procesar el dataset: {type(data)}")
        
        print(f"📊 Dataset cargado: {len(data_list)} registros")
        
        # Procesar el dataset - buscar campos específicos para Mistral-7B dataset
        self.processed_data = []
        sample_item = data_list[0] if data_list else {}
        print(f"📋 Campos disponibles: {list(sample_item.keys())}")
        
        # Campos específicos para el dataset carlmcbrideellis/llm-mistral-7b-instruct-texts
        prompt_field = 'prompt_name'
        response_field = 'generated'
        
        # Verificar si los campos específicos existen
        if prompt_field in sample_item and response_field in sample_item:
            print(f"🎯 Usando campos específicos del dataset Mistral-7B:")
            print(f"   Prompt: '{prompt_field}' -> Respuesta: '{response_field}'")
            
            for item in data_list:
                try:
                    prompt_text = str(item.get(prompt_field, '')).strip()
                    generated_text = str(item.get(response_field, '')).strip()
                    
                    if prompt_text and generated_text and len(prompt_text) > 5 and len(generated_text) > 5:
                        # Formato instrucción-respuesta optimizado para el dataset Mistral
                        combined_text = f"Instrucción: {prompt_text}\nRespuesta: {generated_text}"
                        
                        if len(combined_text.strip()) > 20:  # Filtrar textos muy cortos
                            self.processed_data.append(combined_text)
                    
                except Exception as e:
                    continue  # Saltar elementos problemáticos
        else:
            # Fallback para otros formatos de dataset
            print("⚠️ Campos específicos no encontrados, usando detección automática...")
            text_fields = ['text', 'content', 'instruction', 'input', 'question', 'query', 'prompt']
            response_fields = ['response', 'output', 'answer', 'completion', 'target', 'generated']
            
            # Encontrar los campos correctos
            text_field = None
            response_field = None
            
            for field in text_fields:
                if field in sample_item:
                    text_field = field
                    break
                    
            for field in response_fields:
                if field in sample_item:
                    response_field = field
                    break
            
            # Si no encontramos campos específicos, usar el primer campo de texto disponible
            if not text_field:
                for key in sample_item.keys():
                    if isinstance(sample_item[key], str) and len(sample_item[key]) > 10:
                        text_field = key
                        break
            
            if not response_field and text_field:
                # Si solo tenemos un campo de texto, usarlo para auto-predicción
                response_field = text_field
            
            print(f"🔍 Usando campos fallback: texto='{text_field}', respuesta='{response_field}'")
            
            for item in data_list:
                try:
                    if text_field and response_field:
                        if text_field == response_field:
                            # Auto-predicción: usar todo el texto
                            combined_text = str(item[text_field])
                        else:
                            # Formato instrucción-respuesta
                            text_part = str(item.get(text_field, ''))
                            response_part = str(item.get(response_field, ''))
                            combined_text = f"Instrucción: {text_part}\nRespuesta: {response_part}"
                        
                        if len(combined_text.strip()) > 10:  # Filtrar textos muy cortos
                            self.processed_data.append(combined_text)
                    
                except Exception as e:
                    continue  # Saltar elementos problemáticos
        
        print(f"📊 Datos procesados: {len(self.processed_data)} ejemplos válidos")
        
        # Dividir en train/validation
        split_idx = int(len(self.processed_data) * train_split)
        
        if split == 'train':
            self.data = self.processed_data[:split_idx]
        else:  # validation
            self.data = self.processed_data[split_idx:]
        
        print(f"📊 Split '{split}': {len(self.data)} ejemplos")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenizar
        tokens = self.tokenizer(
            text + self.tokenizer.eos_token,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'text': text[:100] + "..." if len(text) > 100 else text
        }
    
    def print_samples(self, n=3):
        """Imprime muestras del dataset para verificar"""
        print(f"\n📝 Muestras del dataset Mistral-7B (primeras {n}):")
        print("🎯 Formato: prompt_name + generated -> texto de entrenamiento")
        for i in range(min(n, len(self.data))):
            print(f"\n--- Muestra {i+1} ---")
            sample_text = self.data[i]
            # Mostrar más contexto para verificar el formato correcto
            display_text = sample_text[:400] + "..." if len(sample_text) > 400 else sample_text
            print(display_text)
            print(f"📏 Longitud total: {len(sample_text)} caracteres")

def load_kaggle_hf_dataset():
    """Cargar dataset de Kaggle usando HuggingFace"""
    if not KAGGLEHUB_AVAILABLE:
        raise ImportError("KaggleHub no disponible. Instala: pip install kagglehub[hf-datasets]")
    
    try:
        print(f"📦 Cargando dataset: {KAGGLE_DATASET_ID}")
        
        # Cargar usando kagglehub
        file_path = ""  # Vacío para cargar todo el dataset
        hf_dataset = kagglehub.load_dataset(
            KaggleDatasetAdapter.HUGGING_FACE,
            KAGGLE_DATASET_ID,
            file_path
        )
        
        print(f"✅ Dataset cargado exitosamente")
        print(f"📊 Tipo de dataset: {type(hf_dataset)}")
        
        # Inspeccionar estructura
        if hasattr(hf_dataset, 'info'):
            print(f"📋 Info del dataset: {hf_dataset.info}")
        
        if hasattr(hf_dataset, 'features'):
            print(f"📋 Features: {hf_dataset.features}")
        
        return hf_dataset
        
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        print("💡 Verifica que el dataset existe y es público")
        raise e

def custom_collate_fn(batch):
    """Collate function optimizada para Kaggle"""
    input_ids = [item['input_ids'] for item in batch]
    max_length = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    attention_mask = []
    
    for ids in input_ids:
        if len(ids) < max_length:
            padding_length = max_length - len(ids)
            padded_ids = torch.cat([ids, torch.full((padding_length,), 0, dtype=ids.dtype)])  # Pad con 0
            mask = torch.cat([torch.ones(len(ids)), torch.zeros(padding_length)])
        else:
            padded_ids = ids
            mask = torch.ones(len(ids))
        
        padded_input_ids.append(padded_ids)
        attention_mask.append(mask)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(attention_mask).long()
    }

def test_model_kaggle():
    """Prueba el modelo final y muestra generación de ejemplo"""
    print("\n🧪 Probando generación del modelo final...")
    
    # Variables globales necesarias
    global tokenizer, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Cargar modelo guardado
        model_path = KAGGLE_CONFIG['output_dir']
        if os.path.exists(model_path):
            # Cargar configuración
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = HRMKaggleConfig(**config_dict)
            else:
                config = HRMKaggleConfig(vocab_size=len(tokenizer), block_size=BLOCK_SIZE, **MODEL_PARAMS)
            
            # Cargar modelo
            test_model = HRMKaggleModel(config)
            model_file = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(model_file):
                test_model.load_state_dict(torch.load(model_file, map_location='cpu'))
                test_model = test_model.to(device)
                test_model.eval()
                
                # Probar generación simple
                prompts = [
                    "The future of AI is",
                    "In machine learning, we",
                    "Deep neural networks"
                ]
                
                print("📝 Ejemplos de generación:")
                for prompt in prompts[:2]:  # Solo 2 para Kaggle
                    try:
                        # Tokenizar entrada
                        inputs = tokenizer.encode(prompt, return_tensors="pt")
                        if hasattr(inputs, 'to'):
                            inputs = inputs.to(device)
                        elif isinstance(inputs, dict) and 'input_ids' in inputs:
                            inputs = inputs['input_ids']
                            if hasattr(inputs, 'to'):
                                inputs = inputs.to(device)
                        else:
                            # Fallback para tokenizer simple
                            inputs = torch.tensor([inputs], device=device)
                        
                        # Generar
                        with torch.no_grad():
                            outputs = test_model.generate(inputs, max_new_tokens=20, temperature=0.8)
                            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            print(f"   Prompt: {prompt}")
                            print(f"   Output: {response[:100]}...")
                    except Exception as gen_e:
                        print(f"   ⚠️ Error en prompt '{prompt}': {gen_e}")
                        continue
                        
                print("✅ Test de generación completado")
            else:
                print("⚠️ Archivo de modelo no encontrado para test")
        else:
            print("⚠️ Directorio del modelo no encontrado para test")
            
    except Exception as e:
        print(f"❌ Error en test del modelo: {e}")

def main_kaggle_training():
    """Función principal de entrenamiento SIN dependencias de HuggingFace
    
    NOTA: Multi-GPU deshabilitado por defecto debido a problemas de sincronización
    de device en RMSNorm y otros componentes con DataParallel.
    
    Para habilitar multi-GPU (experimental):
    KAGGLE_CONFIG['force_single_gpu'] = False
    """
    print("🚀 Iniciando entrenamiento HRM en Kaggle SIN HuggingFace")
    
    # Configurar dispositivo y detectar multi-GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device("cuda:0")
        
        if num_gpus > 1 and not KAGGLE_CONFIG['force_single_gpu']:
            print(f"🔥 Multi-GPU detectado: {num_gpus} GPUs disponibles")
            KAGGLE_CONFIG['multi_gpu_enabled'] = True
        else:
            if KAGGLE_CONFIG['force_single_gpu']:
                print(f"🔧 Forzando Single-GPU mode (debugging)")
            else:
                print(f"📱 Single-GPU mode: 1 GPU disponible")
            KAGGLE_CONFIG['multi_gpu_enabled'] = False
    else:
        device = torch.device("cpu")
        num_gpus = 0
        print("⚠️  Sin GPU detectada - usando CPU")
    
    print(f"Dispositivo principal: {device}")
    
    # Limpiar memoria inicial
    kaggle_memory_cleanup()
    
    # Crear tokenizer personalizado
    print("🔧 Creando tokenizer personalizado...")
    tokenizer = SimpleTokenizer(vocab_size=30000)
    print(f"Tokenizer creado (será construido con los datos)")
    
    # Cargar dataset local
    print(f"🔍 Cargando dataset local...")
    
    try:
        # Crear datasets de train y validation
        train_dataset = DirectDataset(
            tokenizer=tokenizer,
            max_length=BLOCK_SIZE,
            train_split=0.9,
            split='train'
        )
        
        val_dataset = DirectDataset(
            tokenizer=tokenizer,
            max_length=BLOCK_SIZE,
            train_split=0.9,
            split='val'
        )
        
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return
    
    # Mostrar muestras del dataset
    train_dataset.print_samples(3)
    
    print(f"📊 Datasets creados:")
    print(f"   Entrenamiento: {len(train_dataset)} ejemplos")
    print(f"   Validación: {len(val_dataset)} ejemplos")
    print(f"   Vocabulario: {len(tokenizer)} tokens")
    
    # Crear dataloaders
    dataloader_batch_size = BATCH_SIZE
    if KAGGLE_CONFIG['multi_gpu_enabled']:
        print(f"🔧 Configuración multi-GPU: {BATCH_SIZE} batch size por GPU")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader_batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=2 if IN_KAGGLE else 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataloader_batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=2 if IN_KAGGLE else 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    print(f"📦 DataLoaders creados:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Crear modelo
    print("Creando modelo...")
    config = HRMKaggleConfig(vocab_size=len(tokenizer), block_size=BLOCK_SIZE, **MODEL_PARAMS)
    model = HRMKaggleModel(config).to(device)
    
    # Envolver modelo para multi-GPU si es necesario
    if KAGGLE_CONFIG['multi_gpu_enabled']:
        print(f"🔗 Envolviendo modelo con DataParallel para {num_gpus} GPUs")
        model = nn.DataParallel(model)
        
        # Optimizaciones para DataParallel
        torch.backends.cudnn.benchmark = True
        print("⚡ Optimizaciones multi-GPU activadas")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros del modelo: {total_params:,}")
    
    # Optimizador y scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=LEARNING_RATE_MAX, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95)
    )
    
    # Calcular pasos de entrenamiento
    effective_batch_size = BATCH_SIZE * GRAD_ACCUM_STEPS
    if KAGGLE_CONFIG['multi_gpu_enabled']:
        effective_batch_size *= num_gpus
    
    num_training_steps = len(train_loader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
    num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
    
    # Usar nuestro scheduler personalizado
    scheduler = SimpleCosineScheduler(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    scaler = torch.amp.GradScaler(enabled=MIXED_PRECISION)
    
    print(f"📊 Configuración de entrenamiento:")
    print(f"   Pasos de entrenamiento: {num_training_steps}")
    print(f"   Pasos de warmup: {num_warmup_steps}")
    print(f"   Batch efectivo: {effective_batch_size}")
    
    # Cargar checkpoint si existe
    checkpoint_loaded, start_epoch, global_step, best_val_loss = load_checkpoint_kaggle(
        CHECKPOINT_PATH, model, optimizer, scheduler, scaler, device
    )
    
    # Entrenar
    training_start_time = time.time()
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n🚀 Época {epoch+1}/{NUM_EPOCHS}")
        model.train()
        optimizer.zero_grad()
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        progress = tqdm(train_loader, desc=f"Entrenamiento E{epoch+1}")
        
        for i, batch in enumerate(progress):
            # Verificar límite de tiempo
            if check_time_limit():
                print("⏰ Límite de tiempo alcanzado. Guardando checkpoint...")
                save_checkpoint_kaggle(model, optimizer, scheduler, scaler, 
                                     epoch, global_step, best_val_loss, CHECKPOINT_PATH, tokenizer)
                return
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device, non_blocking=True)
            labels = input_ids.clone()
            
            with torch.amp.autocast(device_type=device.type, enabled=MIXED_PRECISION):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Verificar si el loss es nan/inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Loss inválido detectado: {loss.item()}")
                    # Saltar este batch
                    continue
                
                # Para DataParallel, la loss puede venir como tensor con múltiples valores
                if KAGGLE_CONFIG['multi_gpu_enabled'] and hasattr(loss, 'dim') and loss.dim() > 0:
                    loss = loss.mean()
                
                loss = loss / GRAD_ACCUM_STEPS
            
            if loss is not None and torch.isfinite(loss):
                scaler.scale(loss).backward()
                epoch_loss += loss.item() * GRAD_ACCUM_STEPS
                epoch_steps += 1
                
                if (i + 1) % GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    # Clip más agresivo para prevenir explosión de gradientes
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    if grad_norm > 10.0:
                        print(f"⚠️ Gradientes grandes detectados: {grad_norm:.2f}")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    
                    # Guardar checkpoint cada CHECKPOINT_STEPS
                    if global_step % CHECKPOINT_STEPS == 0:
                        current_loss = loss.item() * GRAD_ACCUM_STEPS
                        current_lr = scheduler.get_last_lr()[0]
                        elapsed_time = time.time() - training_start_time
                        samples_per_sec = (global_step * effective_batch_size) / (elapsed_time + 1e-8)
                        
                        print(f"\n💾 Guardando checkpoint en paso {global_step}")
                        print(f"   📊 Loss actual: {current_loss:.4f}")
                        print(f"   🏆 Mejor loss: {best_val_loss:.4f}")
                        print(f"   📈 Learning rate: {current_lr:.2e}")
                        print(f"   🚀 Throughput: {samples_per_sec:.0f} samples/sec")
                        
                        save_checkpoint_kaggle(model, optimizer, scheduler, scaler, 
                                             epoch, global_step, best_val_loss, CHECKPOINT_PATH, tokenizer)
                        print(f"✅ Checkpoint guardado exitosamente")
                    
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Calcular throughput
                    elapsed_time = time.time() - training_start_time
                    samples_per_sec = (global_step * effective_batch_size) / (elapsed_time + 1e-8)
                    
                    # Actualizar barra de progreso con métricas en tiempo real
                    current_loss = loss.item() * GRAD_ACCUM_STEPS
                    progress.set_postfix({
                        "loss": f"{current_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "step": global_step,
                        "samp/s": f"{samples_per_sec:.0f}"
                    })
                    
                    # También actualizar el promedio de época
                    epoch_loss += current_loss
                    
                    # Checkpoint frecuente
                    if global_step % KAGGLE_CONFIG['checkpoint_frequency'] == 0:
                        save_checkpoint_kaggle(model, optimizer, scheduler, scaler,
                                             epoch, global_step, best_val_loss, CHECKPOINT_PATH, tokenizer)
                    
                    # Limpieza de memoria
                    if global_step % KAGGLE_CONFIG['memory_cleanup_steps'] == 0:
                        kaggle_memory_cleanup()
        
        # Validación
        print(f"\n📊 Validación época {epoch+1}")
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="Validación", leave=False)
            for i, batch in enumerate(val_progress):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
                
                with torch.amp.autocast(device_type=device.type, enabled=MIXED_PRECISION):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss
                    
                    if KAGGLE_CONFIG['multi_gpu_enabled'] and hasattr(loss, 'dim') and loss.dim() > 0:
                        loss = loss.mean()
                
                val_loss += loss.item()
                val_steps += 1
                
                val_progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_val_loss = val_loss / max(val_steps, 1)
        print(f"Pérdida de validación: {avg_val_loss:.4f}")
        
        # Guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Para DataParallel, guardar el modelo sin el wrapper
            model_to_save = model.module if KAGGLE_CONFIG['multi_gpu_enabled'] else model
            torch.save(model_to_save.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 Nuevo mejor modelo! Pérdida: {best_val_loss:.4f}")
    
    # Guardar modelo final
    print("\n💾 Guardando modelo final...")
    save_final_model_kaggle(model, tokenizer, OUTPUT_DIR)
    
    print("✅ Entrenamiento completado!")
    
    # Estadísticas finales
    total_time = time.time() - training_start_time
    total_samples = len(train_dataset) * NUM_EPOCHS
    final_samples_per_sec = total_samples / total_time if total_time > 0 else 0
    
    print(f"\n📊 Estadísticas finales:")
    print(f"   Tiempo total: {total_time/3600:.2f} horas")
    print(f"   Total de muestras procesadas: {total_samples:,}")
    print(f"   Throughput promedio: {final_samples_per_sec:.0f} muestras/seg")
    if KAGGLE_CONFIG['multi_gpu_enabled']:
        print(f"   GPUs utilizadas: {num_gpus}x Tesla T4")
        print(f"   Throughput por GPU: {final_samples_per_sec/num_gpus:.0f} muestras/seg/GPU")
    
    # Mostrar ejemplo de generación
    print(f"\n🧪 Prueba rápida de generación (formato Mistral-7B):")
    model.eval()
    test_prompts = [
        "Instrucción: Explain quantum computing in simple terms",
        "Instrucción: Write a Python function to calculate fibonacci numbers",
        "Instrucción: What are the benefits of renewable energy?",
    ]
    
    for prompt in test_prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"\n{prompt}")
                print(f"Respuesta: {response}")
        except Exception as e:
            print(f"Error en generación: {e}")

# ==============================================================================
# --- CLASE DIRECTDATASET PARA DATASETS LOCALES ---
# ==============================================================================

class DirectDataset(Dataset):
    """Dataset que carga texto directamente desde archivos locales de Kaggle"""
    
    def __init__(self, tokenizer, max_length=768, train_split=0.9, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Buscar archivos de texto en el directorio de input de Kaggle
        input_paths = [
            "/kaggle/input",
            "/kaggle/input/mistral-7b-instruct-texts",
            "/kaggle/input/llm-mistral-7b-instruct-texts"
        ]
        
        found_files = []
        for path in input_paths:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith(('.txt', '.json', '.csv')):
                            found_files.append(os.path.join(root, file))
        
        print(f"🔍 Archivos encontrados: {len(found_files)}")
        
        # Cargar datos desde archivos encontrados
        sample_texts = []
        for file_path in found_files[:10]:  # Limitar para evitar problemas de memoria
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 100:  # Filtrar contenido muy corto
                        sample_texts.append(content[:1000])  # Truncar textos largos
            except Exception as e:
                print(f"⚠️ Error leyendo {file_path}: {e}")
                continue
        
        # Si no encontramos archivos, usar textos de ejemplo
        if not sample_texts:
            print("⚠️ No se encontraron archivos de texto. Usando ejemplos de prueba.")
            sample_texts = [
                "Instrucción: Explain machine learning in simple terms.\nRespuesta: Machine learning is a method of data analysis that automates analytical model building.",
                "Instrucción: Write a Python function to calculate factorial.\nRespuesta: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                "Instrucción: What is artificial intelligence?\nRespuesta: Artificial intelligence is the simulation of human intelligence in machines.",
                "Instrucción: Describe deep learning.\nRespuesta: Deep learning uses neural networks with multiple layers to learn complex patterns.",
                "Instrucción: How does natural language processing work?\nRespuesta: NLP combines computational linguistics with machine learning to help computers understand human language."
            ]
        
        # Construir vocabulario del tokenizer si no está construido
        if not self.tokenizer._built:
            print("🔧 Construyendo vocabulario del tokenizer...")
            self.tokenizer.build_vocab(sample_texts)
        
        # Procesar y dividir datos
        processed_data = []
        for text in sample_texts:
            if len(text.strip()) > 20:  # Filtrar textos muy cortos
                processed_data.append(text.strip())
        
        # Duplicar datos si tenemos muy pocos para entrenamiento
        while len(processed_data) < 100:
            processed_data.extend(processed_data[:10])
        
        # Dividir en train/validation
        split_idx = int(len(processed_data) * train_split)
        
        if split == 'train':
            self.data = processed_data[:split_idx]
        else:  # validation
            self.data = processed_data[split_idx:]
        
        print(f"📊 Dataset '{split}': {len(self.data)} ejemplos")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenizar
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'text': text[:100] + "..." if len(text) > 100 else text
        }
    
    def print_samples(self, n=3):
        """Imprime muestras del dataset para verificar"""
        print(f"\n📝 Muestras del dataset (primeras {n}):")
        for i in range(min(n, len(self.data))):
            print(f"\n--- Muestra {i+1} ---")
            sample_text = self.data[i]
            display_text = sample_text[:400] + "..." if len(sample_text) > 400 else sample_text
            print(display_text)
            print(f"📏 Longitud total: {len(sample_text)} caracteres")

# ==============================================================================
# --- EJECUTAR EN KAGGLE ---
# ==============================================================================

if __name__ == "__main__":
    try:
        print("🎯 HRM Training - Versión Kaggle Migrada")
        print("📋 Migrado desde hrm_training_small_50m.py")
        print("🔧 Sin dependencias problemáticas")
        
        # Entrenamiento principal
        main_kaggle_training()
        
        # Probar modelo
        test_model_kaggle()
        
        print("\n🎉 Entrenamiento completado en Kaggle!")
        
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
        import traceback
        traceback.print_exc()
