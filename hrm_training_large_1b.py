# -*- coding: utf-8 -*-
"""
HRM-Text1 Training Script - ESCALADO A ~1B PARÁMETROS
VERSIÓN AMPLIADA: Configuración para ~1B parámetros con contexto extendido (2048/4096)
- Arquitectura multi-capa HRM apilada (24 capas)
- Rotary Position Embeddings (RoPE) para mejor extrapolación
- Optimizaciones de memoria y velocidad
- Configuración optimizada para modelos grandes
"""

import os, random, contextlib, multiprocessing as mp, atexit, math
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset
from transformers import T5Tokenizer, PreTrainedModel, PretrainedConfig, GenerationMixin, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from huggingface_hub import HfFolder, HfApi

# TensorBoard para monitoreo de entrenamiento
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
    print("✅ TensorBoard disponible para monitoreo de entrenamiento")
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard no disponible. Instala con: pip install tensorboard")
    SummaryWriter = None

# Para descargas de Kaggle
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
    print("✅ Kagglehub disponible para descargas de datasets")
except ImportError:
    KAGGLE_AVAILABLE = False
    print("⚠️  kagglehub no disponible. Datasets de Kaggle deshabilitados.")
    print("💡 Para habilitar, ejecuta: pip install kagglehub")

# Para detección de idioma
try:
    import langdetect
    LANGUAGE_DETECTION_AVAILABLE = True
    print("✅ Detección de idioma disponible con langdetect")
except ImportError:
    LANGUAGE_DETECTION_AVAILABLE = False
    print("⚠️  langdetect no disponible. Filtrado por idioma deshabilitado.")
    print("💡 Para habilitar automáticamente, ejecuta: pip install langdetect")
    
    # Intentar instalación automática si estamos en un entorno compatible
    try:
        import subprocess
        import sys
        response = input("¿Deseas instalar langdetect automáticamente? (y/n): ").strip().lower()
        if response in ['y', 'yes', 's', 'si']:
            print("🔄 Instalando langdetect...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "langdetect"])
            print("✅ langdetect instalado. Reiniciando detección...")
            try:
                import langdetect
                LANGUAGE_DETECTION_AVAILABLE = True
                print("✅ Detección de idioma ahora disponible")
            except ImportError:
                print("❌ Error al importar langdetect después de la instalación")
        else:
            print("⏩ Continuando sin detección de idioma")
    except Exception:
        pass  # Silenciar errores en entornos no interactivos

# ==============================================================================
# --- CONFIGURACIÓN MULTI-GPU ---
# ==============================================================================

def setup_distributed():
    """Inicializar entrenamiento distribuido si está disponible"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Inicializar proceso distribuido
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        print(f"🌐 Distributed training initialized - Rank: {rank}/{world_size}, Local rank: {local_rank}")
        return True, rank, world_size, local_rank
    else:
        # Auto-configuración para múltiples GPUs usando DataParallel (más simple)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            num_gpus = torch.cuda.device_count()
            print(f"🚀 MÚLTIPLES GPUs DETECTADAS - USANDO DATAPARALLEL")
            print(f"   📋 GPUs detectadas: {num_gpus}")
            print(f"   🎯 Usando DataParallel para aprovechar todas las GPUs")
            print(f"   💡 Para mejor rendimiento, considera usar: torchrun --nproc_per_node={num_gpus} {__file__}")
            
            # Retornar modo "pseudo-distribuido" que activará DataParallel
            return True, 0, num_gpus, 0
        elif torch.cuda.is_available():
            print(f"📱 Single-GPU training mode (1 GPU detectada)")
        else:
            print("📱 CPU training mode (sin GPU detectada)")
        return False, 0, 1, 0

def cleanup_distributed():
    """Limpia el entorno distribuido"""
    if dist.is_initialized():
        dist.destroy_process_group()

# Configurar distribución
DISTRIBUTED, RANK, WORLD_SIZE, LOCAL_RANK = setup_distributed()

if DISTRIBUTED:
    print(f"Proceso {RANK}/{WORLD_SIZE} en GPU {LOCAL_RANK}")

# Optimización específica para NVIDIA Ampere+
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    if not DISTRIBUTED or RANK == 0:
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
    # q, k shape: (batch, n_head, seq_len, head_dim)
    # cos, sin shape: (1, seq_len, 1, head_dim)
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
                 h_update_period=5,          # NUEVO: H-module se actualiza cada 5 pasos
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
        self.h_update_period = h_update_period

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
            # Ajustar las dimensiones de cos y sin para que coincidan con q y k
            cos = cos.expand(q.shape[0], -1, q.shape[1], -1)  # (batch, seq_len, n_head, head_dim)
            sin = sin.expand(q.shape[0], -1, q.shape[1], -1)  # (batch, seq_len, n_head, head_dim)
            # Transponer para que coincidan con q, k: (batch, n_head, seq_len, head_dim)
            cos = cos.transpose(1, 2)
            sin = sin.transpose(1, 2)
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
    """True HRM implementation with hierarchical temporal separation"""
    def __init__(self, config):
        super().__init__()
        self.H_module = HRMBlock(config)
        self.L_module = HRMBlock(config)
        self.config = config
        
        # Q-learning components for adaptive computation
        self.q_network = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.ReLU(),
            nn.Linear(config.n_embd // 4, 2)  # [continue, halt]
        )
        
        # Convergence threshold for L-module
        self.convergence_threshold = 1e-3
        self.max_l_steps = config.halt_max_steps
        self.h_update_period = getattr(config, 'h_update_period', 5)  # T steps for large model
    
    def forward(self, z_H, z_L, step_count=0, attn_mask=None, key_padding_mask=None, training=True):
        """Forward pass with proper HRM hierarchical reasoning"""
        batch_size, seq_len, d_model = z_H.shape
        device = z_H.device
        
        # Determine if this is an H-module update step
        is_h_update_step = (step_count % self.h_update_period) == 0
        
        if is_h_update_step:
            # H-module update: Run L-module to convergence, then update H-module
            z_L_converged, l_steps, q_values = self._run_l_module_to_convergence(
                z_H, z_L, attn_mask, key_padding_mask, training
            )
            
            # Update H-module with converged L-module output
            z_H_input = z_H + z_L_converged
            z_H_new = self.H_module(z_H_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            
            # Reset L-module (start fresh for next cycle)
            z_L_new = torch.zeros_like(z_L)
            
            return z_H_new, z_L_new, {
                'h_updated': True,
                'l_steps': l_steps,
                'q_values': q_values,
                'convergence_achieved': True
            }
        else:
            # L-module only step: Continue L-module processing
            z_L_input = z_L + z_H.detach()  # Detach H to prevent gradients
            z_L_new = self.L_module(z_L_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            
            return z_H, z_L_new, {
                'h_updated': False,
                'l_steps': 1,
                'q_values': None,
                'convergence_achieved': False
            }
    
    def _run_l_module_to_convergence(self, z_H, z_L, attn_mask, key_padding_mask, training):
        """Run L-module until convergence or max steps reached"""
        z_L_current = z_L
        z_L_prev = z_L
        all_q_values = []
        
        for l_step in range(self.max_l_steps):
            # L-module forward pass
            z_L_input = z_L_current + z_H.detach()
            z_L_next = self.L_module(z_L_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            
            # Q-learning decision: should we continue or halt?
            if training and l_step < self.max_l_steps - 1:
                q_values = self.q_network(z_L_next)
                all_q_values.append(q_values)
                
                # Epsilon-greedy exploration during training
                epsilon = max(0.1, 1.0 - l_step * 0.1)
                if torch.rand(1).item() < epsilon:
                    action = torch.randint(0, 2, (1,)).item()
                else:
                    # Average Q-values across batch and sequence dimensions, then select action
                    avg_q_values = q_values.mean(dim=[0, 1])  # Shape: [2]
                    action = torch.argmax(avg_q_values).item()
                
                # If action is halt (1), break
                if action == 1:
                    break
            
            # Check convergence
            diff = torch.norm(z_L_next - z_L_current, p=2, dim=-1).mean()
            if diff < self.convergence_threshold:
                break
            
            z_L_prev = z_L_current
            z_L_current = z_L_next
        
        return z_L_current, l_step + 1, all_q_values

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
        
        # Inicializar gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing
        
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
        
        # True HRM processing with hierarchical temporal separation
        step_count = 0
        q_loss_accumulator = []
        
        for layer_idx, (layer, halt_head) in enumerate(zip(self.layers, self.halt_heads)):
            layer_remainders = torch.ones((batch_size, seq_len), device=device)
            layer_total_z_H = torch.zeros_like(z_H)
            layer_n_updates = torch.zeros((batch_size, seq_len), device=device)
            
            for step in range(self.config.halt_max_steps):
                # Apply HRM layer with proper hierarchical separation
                if self.gradient_checkpointing and self.training:
                    # Need to wrap the layer call for checkpointing
                    def layer_call(z_H_in, z_L_in, step_count_in):
                        return layer(z_H_in, z_L_in, step_count=step_count_in, 
                                   attn_mask=causal_mask, key_padding_mask=key_padding_mask, 
                                   training=self.training)
                    z_H, z_L, hrm_info = torch.utils.checkpoint.checkpoint(
                        layer_call, z_H, z_L, step_count, use_reentrant=False
                    )
                else:
                    z_H, z_L, hrm_info = layer(z_H, z_L, step_count=step_count,
                                             attn_mask=causal_mask, key_padding_mask=key_padding_mask,
                                             training=self.training)
                
                # Accumulate Q-learning losses for training
                if hrm_info['q_values'] is not None:
                    q_loss_accumulator.extend(hrm_info['q_values'])
                
                # Traditional ACT halt mechanism (for compatibility)
                p_halt = halt_head(z_H).squeeze(-1).clamp(eps, 1 - eps)
                is_last_step = step == (self.config.halt_max_steps - 1)
                halt_now_prob = p_halt if not is_last_step else torch.ones_like(p_halt)
                
                # Weighted contribution
                contrib = layer_remainders * halt_now_prob
                layer_total_z_H = layer_total_z_H + contrib.unsqueeze(-1) * z_H
                layer_n_updates = layer_n_updates + contrib
                
                if is_last_step:
                    break
                
                # Update remainders
                layer_remainders = layer_remainders * (1 - p_halt)
                step_count += 1
                
                # Early stopping if all tokens decided to halt
                if torch.all(layer_remainders < eps):
                    break
            
            # Update z_H for next layer
            z_H = layer_total_z_H
            total_z_H = total_z_H + z_H  # Accumulate across layers
            n_updates = n_updates + layer_n_updates
        
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
            
            # Q-learning loss para adaptive computation
            q_learning_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if q_loss_accumulator:
                # Calcular recompensa basada en la pérdida de lenguaje (menor pérdida = mayor recompensa)
                reward = -lm_loss.detach()  # Recompensa inversa a la pérdida
                
                # Q-learning loss: minimize TD error
                for q_values in q_loss_accumulator:
                    # Target Q-value basado en la recompensa
                    target_q = reward.expand_as(q_values[..., 0])
                    current_q = q_values[..., 1]  # Q-value for halt action
                    q_learning_loss = q_learning_loss + F.mse_loss(current_q, target_q)
                
                q_learning_loss = q_learning_loss / len(q_loss_accumulator)
            
            # Pérdida total con Q-learning
            loss = (lm_loss + 
                   self.config.ponder_loss_weight * ponder_loss +
                   0.01 * q_learning_loss)  # Small weight for Q-learning
        
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", torch.ones_like(input_ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

# ==============================================================================
# --- CONFIGURACIÓN DEL SCRIPT PARA ~1B PARÁMETROS ---
# ==============================================================================

# --- CONFIGURACIÓN DE PORCENTAJES DE DATASETS ---
# Porcentaje del dataset completo a usar (1-100)
DATASET_SUBSET_PERCENT = 10  # Aumentado para más datos de entrenamiento

# CONFIGURACIÓN PERSONALIZADA DE MEZCLAS
# Puedes crear tus propias combinaciones aquí o modificar las existentes
CUSTOM_MIX_RATIOS = {
    # Ejemplo de mezcla personalizada enfocada en calidad para modelo 1B
    "high_quality_1b": {
        "slimpajama_en": 0.4,  # 40% SlimPajama inglés (alta calidad)
        "pile": 0.3,           # 30% The Pile (diversidad)
        "openwebtext": 0.2,    # 20% OpenWebText (web content)
        "fineweb": 0.1         # 10% FineWeb (muy alta calidad)
    },
    
    # Ejemplo de mezcla para contenido multilingüe balanceado para 1B
    "multilingual_balanced_1b": {
        "c4": 0.3,             # 30% C4 (multilingüe)
        "slimpajama_en": 0.3,  # 30% SlimPajama inglés
        "spanish": 0.2,        # 20% Español
        "slimpajama_es": 0.1,  # 10% SlimPajama español
        "fineweb": 0.1         # 10% FineWeb
    },
    
    # Ejemplo de mezcla experimental con todos los datasets para 1B
    "experimental_full_1b": {
        "slimpajama": 0.25,    # 25% SlimPajama completo
        "c4": 0.2,             # 20% C4 multilingüe
        "pile": 0.2,           # 20% The Pile
        "fineweb": 0.15,       # 15% FineWeb
        "openwebtext": 0.1,    # 10% OpenWebText
        "human_conversations": 0.05,  # 5% Conversaciones humanas
        "spanish": 0.05        # 5% Español
    },
    
    # Mezcla enfocada en conversaciones y calidad para chat
    "conversation_mix_1b": {
        "human_conversations": 0.4,  # 40% Conversaciones humanas
        "fineweb": 0.3,             # 30% Contenido de alta calidad
        "slimpajama_en": 0.2,       # 20% SlimPajama inglés
        "openwebtext": 0.1          # 10% OpenWebText
    }
}

# --- CONFIGURACIÓN DE DATASETS MÚLTIPLES ---
# Selecciona el dataset a usar cambiando ACTIVE_DATASET
ACTIVE_DATASET = "c4"  # Opciones: "c4", "openwebtext", "pile", "spanish", "mixed", "high_quality_1b", etc.

DATASETS_CONFIG = {
    "c4": {
        "name": "allenai/c4",
        "config": "multilingual",
        "train_samples": 364_868_892,
        "val_samples": 364_608,
        "repo_suffix": "C4",
        "description": "Common Crawl multilingüe"
    },
    "openwebtext": {
        "name": "openwebtext",
        "config": None,
        "train_samples": 8_013_769,
        "val_samples": None,  # Se usará split automático
        "repo_suffix": "OpenWebText",
        "description": "Dataset de texto web en inglés"
    },
    "pile": {
        "name": "EleutherAI/pile",
        "config": None,
        "train_samples": 210_607_728,
        "val_samples": 214_670,
        "repo_suffix": "Pile",
        "description": "Dataset diverso de EleutherAI"
    },
    "spanish": {
        "name": "allenai/c4",
        "config": "es",
        "train_samples": 58_395_538,
        "val_samples": None,  # Se usará split automático
        "repo_suffix": "Spanish",
        "description": "Texto en español del dataset C4"
    },
    "fineweb": {
        "name": "HuggingFaceFW/fineweb",
        "config": "default",
        "train_samples": 10_000_000_000,  # 10B tokens aproximadamente
        "val_samples": None,  # Se usará split automático
        "repo_suffix": "FineWeb",
        "description": "Dataset de alta calidad de texto web (FineWeb)"
    },
    "slimpajama": {
        "name": "cerebras/SlimPajama-627B",
        "config": None,
        "train_samples": 627_000_000_000,  # 627B tokens aproximadamente
        "val_samples": None,  # Se usará split automático
        "repo_suffix": "SlimPajama",
        "description": "Dataset SlimPajama de 627B tokens (multilingüe)",
        "language_filter": None  # Usar todo el dataset
    },
    "slimpajama_es": {
        "name": "cerebras/SlimPajama-627B",
        "config": None,
        "train_samples": 50_000_000_000,  # Estimación para contenido en español
        "val_samples": None,
        "repo_suffix": "SlimPajama-ES",
        "description": "SlimPajama filtrado para contenido en español",
        "language_filter": "es"  # Filtrar solo español
    },
    "slimpajama_en": {
        "name": "cerebras/SlimPajama-627B",
        "config": None,
        "train_samples": 400_000_000_000,  # Estimación para contenido en inglés
        "val_samples": None,
        "repo_suffix": "SlimPajama-EN",
        "description": "SlimPajama filtrado para contenido en inglés",
        "language_filter": None  # Deshabilitado para evitar datasets vacíos
    },
    "mixed": {
        "name": "mixed",  # Identificador especial
        "config": None,
        "train_samples": 500_000_000,  # Estimación combinada
        "val_samples": 200_000,
        "repo_suffix": "Mixed",
        "description": "Combinación de múltiples datasets",
        "mix_ratios": {  # Proporción de cada dataset en la mezcla
            "c4": 0.35,
            "fineweb": 0.20,
            "slimpajama_en": 0.35,
            "spanish": 0.10
        }
    },
    "mixed_es": {
        "name": "mixed",  # Identificador especial
        "config": None,
        "train_samples": 150_000_000,  # Estimación para español
        "val_samples": 75_000,
        "repo_suffix": "Mixed-ES",
        "description": "Combinación de datasets con contenido en español",
        "mix_ratios": {  # Proporción de cada dataset en la mezcla
            "slimpajama_es": 0.6,
            "spanish": 0.4
        }
    },
    "human_conversations": {
        "name": "projjal1/human-conversation-training-data",
        "config": None,
        "train_samples": 100_000,  # Estimación aproximada
        "val_samples": None,  # Se creará automáticamente
        "repo_suffix": "HumanConv",
        "description": "Dataset de conversaciones humanas de Kaggle",
        "type": "kaggle"  # Identificador especial para datasets de Kaggle
    }
}

# Añadir las mezclas personalizadas a la configuración principal
for custom_name, mix_ratios in CUSTOM_MIX_RATIOS.items():
    DATASETS_CONFIG[custom_name] = {
        "name": "mixed",
        "config": None,
        "train_samples": 500_000_000,  # Estimación para modelo 1B
        "val_samples": 250_000,
        "repo_suffix": f"Custom-{custom_name.replace('_', '-').title()}",
        "description": f"Mezcla personalizada para 1B: {custom_name.replace('_', ' ').title()}",
        "mix_ratios": mix_ratios
    }

# Mostrar datasets disponibles
print("=== DATASETS DISPONIBLES PARA MODELO 1B ===")
for key, config in DATASETS_CONFIG.items():
    marker = " ← SELECCIONADO" if key == ACTIVE_DATASET else ""
    print(f"• {key}: {config['description']}{marker}")
print("=" * 40)

# Configuración del dataset activo
DATASET_INFO = DATASETS_CONFIG[ACTIVE_DATASET]
DATASET_NAME = DATASET_INFO["name"]
DATASET_CONFIG = DATASET_INFO["config"]

HF_REPO_ID = f"dreamwar/HRM-Text1-{DATASET_INFO['repo_suffix']}-1B"
SEED = 42
NUM_EPOCHS = 3
BLOCK_SIZE = 2048  # Contexto extendido

# --- CONFIGURACIÓN DE BATCH SIZE PARA MULTI-GPU (1B PARÁMETROS) ---
# Configuración optimizada para 8x H200 (80GB VRAM cada una) y modelo de 1B parámetros
# Total effective batch size: BATCH_SIZE * GRAD_ACCUM_STEPS * WORLD_SIZE

if DISTRIBUTED:
    # Para 8 GPUs H200 con modelo de 1B parámetros: batch size conservador por GPU
    BATCH_SIZE = 24  # Por GPU - Total: 24 * 8 = 192 per step
    GRAD_ACCUM_STEPS = 4  # Total effective batch: 192 * 4 = 768
    if RANK == 0:
        print(f"🔢 Configuración Multi-GPU (1B params):")
        print(f"   📦 Batch size por GPU: {BATCH_SIZE}")
        print(f"   🔄 Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
        print(f"   📊 Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS * WORLD_SIZE}")
else:
    # Para GPU única con modelo de 1B parámetros: batch size muy conservador
    BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 8  # Batch efectivo: 64
    print(f"🔢 Configuración GPU única (1B params):")
    print(f"   📦 Batch size: {BATCH_SIZE}")
    print(f"   🔄 Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
    print(f"   📊 Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")

EVAL_STEPS = 1000        # Evaluar cada 1000 pasos

# Learning rate schedule optimizado para datasets grandes con decaimiento suave
LEARNING_RATE_MAX = 4e-4  # Reducido significativamente para datasets grandes y modelo 1B
LEARNING_RATE_MIN = 1e-6  # Mínimo apropiado para modelo grande
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.2        # 20% de warmup más largo para modelo grande

# Optimizaciones
MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 3
USE_GRADIENT_CHECKPOINTING = False  # Temporarily disabled - HRM dynamic computation needs special handling

# --- CAMBIOS PARA EL MODELO 1B ---
# Configuración escalada para aproximadamente 1B de parámetros
# Fórmula aproximada: params ≈ vocab_size * n_embd + n_layers * (4 * n_embd² + 3 * n_embd * d_ff)
MODEL_PARAMS = {
    "n_embd": 1536,                    # Dimensión principal del modelo
    "n_head": 24,                      # 24 cabezas de atención (1536/24 = 64 dim por cabeza)
    "n_layers": 24,                    # 24 capas HRM apiladas
    "d_ff": 6144,                      # 4 * n_embd para FFN
    "dropout": 0.1,
    "halt_max_steps": 12,              # Más pasos para secuencias largas
    "ponder_loss_weight": 1e-2,
    "halt_bias_init": -2.2,
    "use_rotary_embeddings": True,     # RoPE para mejor extrapolación
    "use_flash_attention": True,       # Flash Attention si está disponible
    "gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
    "h_update_period": 5,              # H-module se actualiza cada 5 pasos (para modelo grande)
}

T5_TOKENIZER_REPO = "t5-small"

# ==============================================================================
# --- CONFIGURACIÓN DE RUTAS PERSONALIZADAS ---
# ==============================================================================

# CONFIGURACIÓN DE RUTA BASE (personalizable)
# Puedes cambiar esta ruta para usar tu directorio preferido
CUSTOM_BASE_PATH = None  # Dejar None para usar la ruta por defecto

# Variable de entorno para ruta base (sobrescribe CUSTOM_BASE_PATH)
# Usar: export HRM_OUTPUT_BASE="/tu/ruta" antes de ejecutar el script
HRM_OUTPUT_BASE_ENV = os.environ.get('HRM_OUTPUT_BASE')

# Determinar ruta base final
def determine_output_base():
    """Determina la ruta base según la configuración"""
    # Prioridad: Variable de entorno > Ruta personalizada > Ruta por defecto
    if HRM_OUTPUT_BASE_ENV:
        return HRM_OUTPUT_BASE_ENV
    elif CUSTOM_BASE_PATH:
        return CUSTOM_BASE_PATH
    else:
        # Rutas por defecto según el entorno
        if os.path.exists("/content/drive/MyDrive"):
            return "/content/drive/MyDrive/HRM"  # Google Colab
        elif os.path.exists(os.path.expanduser("~/Documents")):
            return os.path.expanduser("~/Documents/HRM")  # Sistemas Unix/Mac
        else:
            return "./HRM_Models"  # Directorio actual como fallback

# --- CONFIGURACIÓN PARA ENTRENAMIENTO SECUENCIAL ---
# Flag para mantener el mismo directorio durante entrenamiento secuencial
SEQUENTIAL_TRAINING = False  # Cambiar a True para mantener checkpoints entre datasets
BASE_MODEL_NAME = "hrm_text1_c4_1b_output"  # Nombre base para entrenamiento secuencial

# Configurar rutas finales
OUTPUT_BASE = determine_output_base()

# Determinar directorio de salida según modo de entrenamiento
if SEQUENTIAL_TRAINING:
    # Modo secuencial: usar directorio base fijo para mantener checkpoints
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, BASE_MODEL_NAME)
    print(f"🔄 MODO SECUENCIAL ACTIVADO: Usando directorio fijo para checkpoints")
else:
    # Modo normal: directorio específico por dataset
    dataset_suffix = DATASET_INFO['repo_suffix'].lower().replace('-', '_')
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"hrm_text1_{dataset_suffix}_1b_output")

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.bin")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")

print(f"📁 Ruta base configurada: {OUTPUT_BASE}")
print(f"📁 Directorio de salida: {OUTPUT_DIR}")
if SEQUENTIAL_TRAINING:
    print(f"📁 MODO SECUENCIAL: Los checkpoints se mantendrán entre cambios de dataset")
else:
    print(f"📁 MODO NORMAL: Directorio específico para dataset {ACTIVE_DATASET}")

# ==============================================================================
# --- FUNCIONES AUXILIARES PARA DATALOADER ---
# ==============================================================================

def get_dataloader_workers():
    """Determina el número seguro de workers para DataLoader"""
    try:
        # Detectar si estamos en Google Colab
        if 'google.colab' in str(get_ipython()):
            print("Detectado entorno Google Colab. Usando num_workers=0 para evitar problemas de multiprocessing.")
            return 0
    except:
        pass

    try:
        # Detectar si estamos en Jupyter/IPython
        get_ipython()
        print("Detectado entorno Jupyter/IPython. Usando num_workers=0 para mayor estabilidad.")
        return 0
    except:
        pass

    # Para sistemas normales, usar menos workers para evitar problemas
    workers = min(2, mp.cpu_count())
    print(f"Detectado sistema normal. Usando {workers} workers para DataLoader.")
    return workers

def cleanup_dataloaders():
    """Función para limpiar DataLoaders al salir"""
    global train_loader, val_loader
    try:
        if 'train_loader' in globals():
            del train_loader
        if 'val_loader' in globals():
            del val_loader
        torch.cuda.empty_cache()
        print("DataLoaders limpiados correctamente.")
    except:
        pass

# Registrar la función de limpieza
atexit.register(cleanup_dataloaders)

# ==============================================================================
# --- FUNCIONES AUXILIARES PARA VALIDACIÓN DE CONFIGURACIÓN ---
# ==============================================================================

def validate_mix_ratios(mix_ratios, dataset_name=""):
    """
    Valida que los ratios de mezcla sumen 1.0 y que todos los datasets existan
    """
    if not mix_ratios:
        return True, "No hay ratios de mezcla definidos"
    
    # Verificar que los datasets existen
    available_datasets = set(DATASETS_CONFIG.keys()) - {"mixed", "mixed_es"} - set(CUSTOM_MIX_RATIOS.keys())
    for dataset in mix_ratios.keys():
        if dataset not in available_datasets:
            return False, f"Dataset '{dataset}' no existe. Disponibles: {sorted(available_datasets)}"
    
    # Verificar que suman aproximadamente 1.0
    total = sum(mix_ratios.values())
    if abs(total - 1.0) > 0.01:  # Tolerancia de 1%
        return False, f"Los ratios deben sumar 1.0, actualmente suman {total:.3f}"
    
    # Verificar que todos los valores son positivos
    for dataset, ratio in mix_ratios.items():
        if ratio <= 0:
            return False, f"El ratio para '{dataset}' debe ser positivo, actual: {ratio}"
    
    return True, f"Configuración válida para {dataset_name}"

def normalize_mix_ratios(mix_ratios):
    """
    Normaliza los ratios para que sumen exactamente 1.0
    """
    total = sum(mix_ratios.values())
    if total == 0:
        return mix_ratios
    
    return {dataset: ratio / total for dataset, ratio in mix_ratios.items()}

def show_mix_summary(mix_ratios, dataset_name=""):
    """
    Muestra un resumen de la configuración de mezcla
    """
    print(f"\n=== CONFIGURACIÓN DE MEZCLA: {dataset_name.upper()} ===")
    for dataset, ratio in sorted(mix_ratios.items()):
        desc = DATASETS_CONFIG.get(dataset, {}).get("description", "Desconocido")
        print(f"• {dataset:20} {ratio:>6.1%} - {desc}")
    print("=" * 60)

# ==============================================================================
# --- FUNCIONES AUXILIARES PARA FILTRADO DE IDIOMA ---
# ==============================================================================

def detect_language(text, target_lang=None, confidence_threshold=0.7):
    """
    Detecta el idioma de un texto y retorna True si coincide con el target_lang
    """
    if not LANGUAGE_DETECTION_AVAILABLE or target_lang is None:
        return True
    
    try:
        # Usar solo una muestra del texto para eficiencia
        sample_text = text[:500] if len(text) > 500 else text
        
        if len(sample_text.strip()) < 50:  # Texto muy corto
            return True
        
        detected_lang = langdetect.detect(sample_text)
        
        # Para algunos idiomas comunes, usar códigos alternativos
        lang_mapping = {
            'es': ['es', 'ca'],  # Español incluye catalán
            'en': ['en'],
            'fr': ['fr'],
            'de': ['de'],
            'it': ['it'],
            'pt': ['pt']
        }
        
        target_langs = lang_mapping.get(target_lang, [target_lang])
        return detected_lang in target_langs
        
    except Exception:
        # En caso de error, incluir el texto
        return True

def create_language_filter_function(target_lang, relaxed=False):
    """
    Crea una función de filtro para un idioma específico
    
    Args:
        target_lang: Idioma objetivo (ej: 'en', 'es')
        relaxed: Si True, usa criterios menos restrictivos
    """
    def language_filter(examples):
        if not LANGUAGE_DETECTION_AVAILABLE or target_lang is None:
            return examples
        
        filtered_examples = {key: [] for key in examples.keys()}
        
        # Detectar campo de texto
        text_field = None
        for field in ['text', 'content', 'document']:
            if field in examples:
                text_field = field
                break
        
        if text_field is None:
            return examples
        
        # Configurar umbrales según el modo
        if relaxed:
            min_text_length = 10  # Más permisivo
            fallback_threshold = 0.05  # Permitir hasta 95% de filtrado
            print(f"    🔧 Modo relajado: min_length={min_text_length}, threshold={fallback_threshold}")
        else:
            min_text_length = 20
            fallback_threshold = 0.1  # Permitir hasta 90% de filtrado
        
        # Filtrar por idioma con manejo de errores y fallback
        total_texts = len(examples[text_field])
        accepted_count = 0
        
        for i, text in enumerate(examples[text_field]):
            should_include = True
            
            try:
                if isinstance(text, str) and len(text.strip()) > min_text_length:
                    should_include = detect_language(text, target_lang)
                else:
                    # Incluir textos muy cortos sin filtrar
                    should_include = True
            except Exception:
                # En caso de error en detección, incluir el texto
                should_include = True
            
            if should_include:
                for key in examples.keys():
                    filtered_examples[key].append(examples[key][i])
                accepted_count += 1
        
        # Aplicar umbral de fallback
        if total_texts > 0 and accepted_count / total_texts < fallback_threshold:
            rejection_rate = (total_texts - accepted_count) / total_texts * 100
            print(f"    ⚠️  Filtro muy restrictivo ({accepted_count}/{total_texts}, {rejection_rate:.1f}% rechazado)")
            print(f"    🔄 Manteniendo batch original para evitar dataset vacío")
            return examples
        
        return filtered_examples
    
    return language_filter

# ==============================================================================
# --- VALIDACIÓN Y CREACIÓN DE DIRECTORIOS ---
# ==============================================================================

def validate_and_create_output_dir(output_dir, force_create=True):
    """
    Valida y crea el directorio de salida con verificaciones de seguridad
    """
    try:
        # Verificar que el directorio padre sea accesible
        parent_dir = os.path.dirname(output_dir)
        
        if not os.path.exists(parent_dir):
            if force_create:
                print(f"🔨 Creando directorio padre: {parent_dir}")
                os.makedirs(parent_dir, exist_ok=True)
            else:
                raise FileNotFoundError(f"Directorio padre no existe: {parent_dir}")
        
        # Crear directorio de salida
        if not os.path.exists(output_dir):
            print(f"🔨 Creando directorio de salida: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        else:
            print(f"✅ Directorio de salida existe: {output_dir}")
        
        # Verificar permisos de escritura
        test_file = os.path.join(output_dir, ".write_test")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"✅ Permisos de escritura verificados")
        except PermissionError:
            raise PermissionError(f"Sin permisos de escritura en: {output_dir}")
        
        # Verificar espacio disponible (estimación básica)
        try:
            import shutil
            free_space = shutil.disk_usage(output_dir).free
            free_gb = free_space / (1024**3)
            print(f"💾 Espacio libre disponible: {free_gb:.1f} GB")
            
            if free_gb < 5:
                print(f"⚠️  ADVERTENCIA: Poco espacio libre ({free_gb:.1f} GB). Se recomiendan al menos 5 GB para modelo 1B")
            elif free_gb < 20:
                print(f"💡 Espacio moderado ({free_gb:.1f} GB). Para entrenamientos largos se recomiendan al menos 20 GB")
        except:
            print("ℹ️  No se pudo verificar el espacio disponible")
        
        return True
        
    except Exception as e:
        print(f"❌ Error configurando directorio de salida: {e}")
        print(f"💡 Sugerencias:")
        print(f"   - Verificar permisos del directorio padre")
        print(f"   - Usar una ruta diferente con CUSTOM_BASE_PATH")
        print(f"   - Verificar que tengas suficiente espacio en disco")
        return False

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

# Validar y crear directorios
print("\n🔍 Validando configuración de directorios...")
if not validate_and_create_output_dir(OUTPUT_DIR):
    print("❌ No se pudo configurar el directorio de salida. Abortando.")
    exit(1)

print(f"✅ Configuración de directorios completada")
print(f"📋 Archivos que se guardarán:")
print(f"   🏆 Mejor modelo: {BEST_MODEL_PATH}")
print(f"   💾 Checkpoints: {CHECKPOINT_PATH}")
print(f"   📝 Modelo final: {OUTPUT_DIR}/")

# Configurar TensorBoard
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, "tensorboard_logs")
if TENSORBOARD_AVAILABLE:
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    print(f"📊 TensorBoard logs: {TENSORBOARD_DIR}")
    print(f"💡 Para ver TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}")

# Configurar device según si estamos en modo distribuido o no
if DISTRIBUTED:
    device = torch.device(f"cuda:{LOCAL_RANK}")
    if RANK == 0:
        print(f"🎯 Dispositivo distribuido: usando {WORLD_SIZE} GPUs")
        print(f"   📍 Proceso {RANK} usando GPU {LOCAL_RANK}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Dispositivo único detectado: {device}")

# Verificar memoria disponible
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"🔥 {num_gpus} GPU(s) detectada(s):")
    
    total_vram = 0
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1e9
        total_vram += vram_gb
        print(f"   GPU {i}: {props.name} - {vram_gb:.1f} GB VRAM")
    
    print(f"💾 VRAM total disponible: {total_vram:.1f} GB")
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

# Usar las cifras específicas del dataset seleccionado y calcular muestras
TOTAL_TRAIN_SAMPLES = DATASET_INFO["train_samples"]
TOTAL_VAL_SAMPLES = DATASET_INFO["val_samples"]

num_train_samples = int(TOTAL_TRAIN_SAMPLES * (DATASET_SUBSET_PERCENT / 100.0))

# Manejar datasets que no tienen split de validación predefinido
if TOTAL_VAL_SAMPLES is None:
    # Para datasets sin validación, usar el 1% del entrenamiento como validación
    num_val_samples = max(1000, int(num_train_samples * 0.01))
    print(f"Dataset sin split de validación. Usando {num_val_samples:,} ejemplos como validación.")
else:
    num_val_samples = int(TOTAL_VAL_SAMPLES * (DATASET_SUBSET_PERCENT / 100.0))

print(f"Loading dataset '{DATASET_NAME}' ({DATASET_INFO['description']}) in streaming mode.")

if ACTIVE_DATASET == "mixed" or ACTIVE_DATASET in CUSTOM_MIX_RATIOS or "mix_ratios" in DATASET_INFO:
    # Cargar y mezclar múltiples datasets
    print("--- CARGANDO DATASETS PARA MEZCLA (MODELO 1B) ---")
    mixed_datasets = {}
    mix_ratios = DATASET_INFO["mix_ratios"]
    
    # Validar configuración de mezcla
    is_valid, message = validate_mix_ratios(mix_ratios, ACTIVE_DATASET)
    if not is_valid:
        print(f"❌ ERROR EN CONFIGURACIÓN: {message}")
        print("Usa normalize_mix_ratios() para corregir automáticamente")
        exit(1)
    else:
        print(f"✅ {message}")
    
    # Normalizar ratios para asegurar que sumen 1.0
    mix_ratios = normalize_mix_ratios(mix_ratios)
    
    # Mostrar resumen de la mezcla
    show_mix_summary(mix_ratios, ACTIVE_DATASET)
    
    for dataset_key, ratio in mix_ratios.items():
        if ratio > 0:
            ds_config = DATASETS_CONFIG[dataset_key]
            print(f"Cargando {dataset_key} ({ratio*100:.1f}%): {ds_config['description']}")
            
            if ds_config["config"]:
                ds = load_dataset(ds_config["name"], ds_config["config"], streaming=True)
            else:
                ds = load_dataset(ds_config["name"], streaming=True)
            
            # Aplicar filtro de idioma específico del dataset si existe
            ds_lang_filter = ds_config.get("language_filter")
            if ds_lang_filter and LANGUAGE_DETECTION_AVAILABLE:
                print(f"  Aplicando filtro de idioma {ds_lang_filter} a {dataset_key}")
                try:
                    # Para SlimPajama, usar un enfoque menos restrictivo
                    if "slimpajama" in dataset_key.lower():
                        print(f"    📝 Usando filtro menos restrictivo para {dataset_key}")
                        lang_filter_func = create_language_filter_function(ds_lang_filter, relaxed=True)
                        # Usar batch size aún más pequeño para SlimPajama
                        ds["train"] = ds["train"].filter(lang_filter_func, batched=True, batch_size=20)
                    else:
                        lang_filter_func = create_language_filter_function(ds_lang_filter)
                        ds["train"] = ds["train"].filter(lang_filter_func, batched=True, batch_size=50)
                    
                    if "validation" in ds:
                        ds["validation"] = ds["validation"].filter(lang_filter_func, batched=True, batch_size=50)
                        
                except Exception as e:
                    print(f"  ⚠️  Error aplicando filtro de idioma a {dataset_key}: {e}")
                    print(f"  🔄 Continuando sin filtro de idioma para {dataset_key}")
            elif ds_lang_filter and not LANGUAGE_DETECTION_AVAILABLE:
                print(f"  ⚠️  Filtro de idioma solicitado para {dataset_key} pero langdetect no disponible")
            
            # Calcular muestras según la proporción
            samples_for_this_ds = int(num_train_samples * ratio)
            # Usar hash absoluto para evitar seeds negativos
            dataset_seed = SEED + abs(hash(dataset_key)) % 1000000
            
            # Asegurar que el dataset tenga la estructura correcta antes de agregarlo
            train_ds = ds["train"].take(samples_for_this_ds).shuffle(seed=dataset_seed, buffer_size=5_000)
            
            # Debug: mostrar las columnas del dataset
            try:
                sample = next(iter(train_ds))
                print(f"  Columnas en {dataset_key}: {list(sample.keys())}")
                
                # Solo agregar al diccionario si el dataset es válido
                mixed_datasets[dataset_key] = {
                    "train": train_ds,
                    "validation": ds.get("validation", ds["train"]).take(int(num_val_samples * ratio)) if ds.get("validation") else None
                }
                
            except Exception as e:
                print(f"  ⚠️  Error al obtener muestra de {dataset_key}: {e}")
                print(f"  ❌ Excluyendo {dataset_key} de la mezcla debido al error")
                continue
    
    # Combinar los datasets
    from datasets import interleave_datasets
    
    # Función para estandarizar columnas de datasets
    def standardize_dataset_columns(dataset, target_columns=None):
        """Estandariza las columnas de un dataset para hacerlo compatible con otros"""
        sample = next(iter(dataset))
        current_columns = list(sample.keys())
        
        # Si no se especifican columnas objetivo, usar las columnas estándar de texto
        if target_columns is None:
            # Buscar campo de texto principal
            text_field = None
            for field in ['text', 'content', 'document']:
                if field in current_columns:
                    text_field = field
                    break
            
            if text_field is None:
                # Usar la primera columna que parezca texto
                for field in current_columns:
                    if isinstance(sample[field], str) and len(sample[field]) > 50:
                        text_field = field
                        break
            
            if text_field is None:
                raise ValueError(f"No se encontró campo de texto en dataset con columnas: {current_columns}")
            
            # Mapear al campo estándar 'text'
            if text_field != 'text':
                dataset = dataset.rename_column(text_field, 'text')
        
        return dataset
    
    # Estandarizar todos los datasets antes de combinar
    standardized_train_datasets = []
    successfully_processed_keys = []  # Track which datasets were successfully processed
    
    for key in mix_ratios.keys():
        if mix_ratios[key] > 0 and key in mixed_datasets:
            try:
                std_dataset = standardize_dataset_columns(mixed_datasets[key]["train"])
                standardized_train_datasets.append(std_dataset)
                successfully_processed_keys.append(key)
                print(f"  ✅ Dataset {key} estandarizado correctamente")
            except Exception as e:
                print(f"  ❌ Error estandarizando {key}: {e}")
                print(f"  ❌ Excluyendo {key} de la mezcla debido al error de estandarización")
                # Continuar sin este dataset si hay error
                continue
    
    if len(standardized_train_datasets) == 0:
        raise ValueError("No se pudieron cargar datasets válidos para la mezcla")
    
    # Calcular probabilidades exactamente para los datasets que se procesaron exitosamente
    valid_probs = [mix_ratios[key] for key in successfully_processed_keys]
    prob_sum = sum(valid_probs)
    train_probs = [p / prob_sum for p in valid_probs]
    
    print(f"  📊 Datasets exitosos: {successfully_processed_keys}")
    print(f"  📊 Probabilidades normalizadas: {train_probs}")
    print(f"  📊 Suma de probabilidades: {sum(train_probs):.6f}")
    
    print(f"Creando dataset mezclado con {len(standardized_train_datasets)} fuentes...")
    
    # Validación final antes de interleaving
    if len(standardized_train_datasets) != len(train_probs):
        raise ValueError(f"Mismatch: {len(standardized_train_datasets)} datasets pero {len(train_probs)} probabilidades")
    
    if abs(sum(train_probs) - 1.0) > 1e-6:
        raise ValueError(f"Probabilidades no suman 1.0: {sum(train_probs)}")
    
    try:
        raw_datasets = {
            "train": interleave_datasets(standardized_train_datasets, probabilities=train_probs, seed=SEED, stopping_strategy="all_exhausted")
        }
        print("✅ Dataset de entrenamiento mezclado creado exitosamente")
    except Exception as e:
        print(f"❌ Error al crear dataset mezclado: {e}")
        print("🔄 Intentando estrategia de respaldo...")
        
        # Estrategia de respaldo: usar solo el primer dataset válido
        if len(standardized_train_datasets) > 0:
            print(f"Usando solo el primer dataset como respaldo")
            raw_datasets = {
                "train": standardized_train_datasets[0]
            }
        else:
            raise ValueError("No hay datasets válidos disponibles")
    
    # Para validación, tomar una muestra pequeña de cada dataset
    val_datasets = []
    val_probs = []
    
    for key in mix_ratios.keys():
        if mix_ratios[key] > 0 and key in mixed_datasets and mixed_datasets[key]["validation"] is not None:
            val_datasets.append(mixed_datasets[key]["validation"])
            val_probs.append(mix_ratios[key])
    
    if val_datasets and len(val_datasets) > 1:
        # Normalizar probabilidades para validación
        val_probs_sum = sum(val_probs)
        val_probs = [p / val_probs_sum for p in val_probs]
        
        try:
            raw_datasets["validation"] = interleave_datasets(val_datasets, probabilities=val_probs, seed=SEED, stopping_strategy="all_exhausted")
        except Exception as e:
            print(f"⚠️  Error al crear dataset de validación mezclado: {e}")
            print("Usando muestra del dataset de entrenamiento para validación")
            raw_datasets["validation"] = raw_datasets["train"].take(num_val_samples)
    elif val_datasets and len(val_datasets) == 1:
        # Solo un dataset de validación disponible
        raw_datasets["validation"] = val_datasets[0]
    else:
        # Si no hay validación, usar una muestra del entrenamiento
        print("No hay datasets de validación disponibles. Usando muestra del entrenamiento.")
        raw_datasets["validation"] = raw_datasets["train"].take(num_val_samples)
    
    print(f"Dataset mezclado creado con {len(standardized_train_datasets)} fuentes")
    
else:
    # Cargar dataset único
    if DATASET_INFO.get("type") == "kaggle":
        # Lógica especial para datasets de Kaggle
        if not KAGGLE_AVAILABLE:
            print(f"❌ Error: Dataset de Kaggle seleccionado pero kagglehub no está disponible")
            print("💡 Instala kagglehub con: pip install kagglehub")
            exit(1)
        
        print(f"📥 Descargando dataset de Kaggle: {DATASET_NAME}")
        try:
            # Download latest version
            kaggle_path = kagglehub.dataset_download(DATASET_NAME)
            print(f"✅ Dataset descargado en: {kaggle_path}")
            
            # Cargar desde archivos locales
            import glob
            
            # Buscar archivos de datos en el directorio descargado
            data_files = glob.glob(os.path.join(kaggle_path, "*.json")) + \
                        glob.glob(os.path.join(kaggle_path, "*.csv")) + \
                        glob.glob(os.path.join(kaggle_path, "*.jsonl"))
            
            if not data_files:
                raise FileNotFoundError(f"No se encontraron archivos de datos en {kaggle_path}")
            
            print(f"📁 Archivos encontrados: {[os.path.basename(f) for f in data_files]}")
            
            # Crear dataset de Hugging Face desde archivos locales
            if data_files[0].endswith('.json') or data_files[0].endswith('.jsonl'):
                raw_datasets = load_dataset('json', data_files={'train': data_files}, streaming=True)
            elif data_files[0].endswith('.csv'):
                raw_datasets = load_dataset('csv', data_files={'train': data_files}, streaming=True)
            else:
                raise ValueError(f"Formato de archivo no soportado: {data_files[0]}")
                
            # Crear split de validación si no existe
            if 'validation' not in raw_datasets:
                print("Creando split de validación a partir del entrenamiento...")
                train_dataset = raw_datasets['train']
                raw_datasets = {
                    'train': train_dataset.skip(1000),
                    'validation': train_dataset.take(1000)
                }
                
        except Exception as e:
            print(f"❌ Error descargando dataset de Kaggle: {e}")
            print("🔄 Cambiando a dataset C4 como respaldo...")
            raw_datasets = load_dataset("allenai/c4", "multilingual", streaming=True)
    
    else:
        # Datasets normales de Hugging Face
        if DATASET_CONFIG:
            raw_datasets = load_dataset(DATASET_NAME, DATASET_CONFIG, streaming=True)
        else:
            raw_datasets = load_dataset(DATASET_NAME, streaming=True)
    
    # Aplicar filtro de idioma si está especificado
    language_filter = DATASET_INFO.get("language_filter")
    if language_filter and LANGUAGE_DETECTION_AVAILABLE:
        print(f"--- APLICANDO FILTRO DE IDIOMA: {language_filter.upper()} ---")
        print("NOTA: Esto puede reducir significativamente la velocidad de carga inicial")
        
        # Crear función de filtro
        lang_filter_func = create_language_filter_function(language_filter)
        
        # Aplicar filtro a los datasets
        raw_datasets["train"] = raw_datasets["train"].filter(lang_filter_func, batched=True, batch_size=100)
        if "validation" in raw_datasets:
            raw_datasets["validation"] = raw_datasets["validation"].filter(lang_filter_func, batched=True, batch_size=100)
    elif language_filter and not LANGUAGE_DETECTION_AVAILABLE:
        print(f"⚠️  ADVERTENCIA: Filtro de idioma '{language_filter}' solicitado pero langdetect no está disponible")
        print("💡 Puedes instalar langdetect con: pip install langdetect")
        print("🔄 Continuando sin filtro de idioma...")


language_filter_info = ""
if DATASET_INFO.get("language_filter"):
    language_filter_info = f" (FILTRADO: {DATASET_INFO['language_filter'].upper()})"

print(f"\n!!! USANDO DATASET: {ACTIVE_DATASET.upper()} - {DATASET_INFO['description']}{language_filter_info} !!!")
print(f"!!! USANDO UN SUBCONJUNTO DEL {DATASET_SUBSET_PERCENT}% DEL DATASET !!!")
print(f"Tomando aprox. {num_train_samples:,} ejemplos de entrenamiento.")
print(f"Tomando aprox. {num_val_samples:,} ejemplos de validación.\n")

# Configurar los splits según el dataset
if ACTIVE_DATASET not in ["mixed"] and ACTIVE_DATASET not in CUSTOM_MIX_RATIOS and "mix_ratios" not in DATASET_INFO:
    # Para datasets únicos, aplicar la lógica original
    if "validation" in raw_datasets:
        raw_datasets["train"] = raw_datasets["train"].take(num_train_samples).shuffle(seed=SEED, buffer_size=10_000)
        raw_datasets["validation"] = raw_datasets["validation"].take(num_val_samples)
    else:
        # Para datasets sin split de validación, dividir el entrenamiento
        print("Dividiendo dataset de entrenamiento para crear validación...")
        total_for_split = num_train_samples + num_val_samples
        train_dataset = raw_datasets["train"].take(total_for_split).shuffle(seed=SEED, buffer_size=10_000)
        
        # Crear splits manualmente
        raw_datasets["train"] = train_dataset.skip(num_val_samples).take(num_train_samples)
        raw_datasets["validation"] = train_dataset.take(num_val_samples)
# Para dataset mezclado, los splits ya están configurados

def tokenize_function(examples):
    """Función de tokenización flexible que maneja diferentes formatos de dataset"""
    text_field_name = next((f for f in ['text', 'content', 'document'] if f in examples), None)
    if not text_field_name:
        raise ValueError(f"No se encontró campo de texto válido. Campos: {list(examples.keys())}")
    
    # Procesar cada ejemplo individualmente para mantener la correspondencia
    texts = []
    for text in examples[text_field_name]:
        if isinstance(text, str) and len(text) > 100:
            texts.append(text + tokenizer.eos_token)
        else:
            # Si el texto no es válido, usar placeholder
            texts.append(tokenizer.eos_token * 10)
    
    # Tokenizar todos los textos
    tokenized = tokenizer(texts, truncation=True, max_length=BLOCK_SIZE, padding="max_length", add_special_tokens=False)
    
    # Para datasets streaming, devolver solo los campos del tokenizer
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask']
    }

print("Applying tokenization function...")
# Verificar que los datasets se cargaron correctamente
if raw_datasets is None or raw_datasets.get("train") is None:
    raise ValueError("❌ Error: Los datasets no se cargaron correctamente. raw_datasets['train'] es None.")

print(f"Tipo de dataset: {type(raw_datasets['train'])}")

# Para datasets streaming, necesitamos manejar las columnas de manera diferente
if hasattr(raw_datasets["train"], 'features') and raw_datasets["train"].features is not None:
    # Dataset no streaming (tiene .features)
    columns_to_remove = list(raw_datasets["train"].features.keys())
    print(f"Columnas a eliminar después de tokenización: {columns_to_remove}")
    tokenized_splits = {}
    for split_name in ["train", "validation"]:
        tokenized_splits[split_name] = raw_datasets[split_name].map(
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove,
            num_proc=max(1, mp.cpu_count() // 2)
        )
else:
    # Dataset streaming (IterableDataset)
    print("Dataset streaming detectado - aplicando tokenización sin remove_columns")
    tokenized_splits = {
        "train": raw_datasets["train"].map(tokenize_function, batched=True),
        "validation": raw_datasets["validation"].map(tokenize_function, batched=True)
    }

safe_num_workers = get_dataloader_workers()
print(f"Creando DataLoaders con {safe_num_workers} workers...")

# Detectar si es IterableDataset para ajustar parámetros
is_iterable = hasattr(tokenized_splits["train"], '__iter__') and not hasattr(tokenized_splits["train"], '__len__')
train_shuffle = False if is_iterable else True

print(f"Dataset iterable detectado: {is_iterable}, shuffle para entrenamiento: {train_shuffle}")

# Función de collate personalizada para filtrar tipos no compatibles
def custom_collate_fn(batch):
    """Collate personalizado que filtra campos no compatibles con PyTorch"""
    # Solo procesar los campos que necesitamos para el entrenamiento
    filtered_batch = []
    for item in batch:
        # Filtrar solo los campos necesarios: input_ids, attention_mask
        filtered_item = {}
        for key in ['input_ids', 'attention_mask']:
            if key in item and isinstance(item[key], (torch.Tensor, list, int, float)):
                # Asegurar que las listas se conviertan a tensores
                if isinstance(item[key], list):
                    filtered_item[key] = torch.tensor(item[key])
                else:
                    filtered_item[key] = item[key]
        filtered_batch.append(filtered_item)
    
    return default_collate(filtered_batch)

train_loader = DataLoader(
    tokenized_splits["train"],
    batch_size=BATCH_SIZE,
    num_workers=safe_num_workers,
    pin_memory=True,
    shuffle=train_shuffle,
    collate_fn=custom_collate_fn
)

val_loader = DataLoader(
    tokenized_splits["validation"],
    batch_size=BATCH_SIZE,
    num_workers=safe_num_workers,
    pin_memory=True,
    shuffle=False,
    collate_fn=custom_collate_fn
)

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
if is_iterable:
    # Para IterableDataset, calculamos basado en las muestras objetivo
    estimated_steps_per_epoch = num_train_samples // BATCH_SIZE
    num_training_steps = estimated_steps_per_epoch * NUM_EPOCHS
    print(f"Dataset iterable: calculando pasos estimados basado en {num_train_samples:,} muestras")
else:
    # Para datasets regulares, usar cálculo tradicional
    num_training_steps = (num_train_samples // (BATCH_SIZE * GRAD_ACCUM_STEPS)) * NUM_EPOCHS

num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
print(f"Total de pasos de entrenamiento (estimado): {num_training_steps:,}")
print(f"Pasos de warmup: {num_warmup_steps:,}")

# Scheduler coseno con warmup para decaimiento más suave
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    num_cycles=0.5  # Media vuelta coseno para decaimiento más suave
)

# Mixed precision scaler
scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and device.type == 'cuda'))

# Inicializar TensorBoard Writer (solo en proceso principal)
writer = None
if TENSORBOARD_AVAILABLE and (not DISTRIBUTED or LOCAL_RANK == 0):
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
    print(f"📊 TensorBoard Writer inicializado")
    
    # Log hyperparameters
    hyperparams = {
        'model/n_embd': MODEL_PARAMS['n_embd'],
        'model/n_layers': MODEL_PARAMS['n_layers'],
        'model/n_head': MODEL_PARAMS['n_head'],
        'model/block_size': BLOCK_SIZE,
        'training/batch_size': BATCH_SIZE,
        'training/learning_rate': LEARNING_RATE_MAX,
        'training/weight_decay': WEIGHT_DECAY,
        'training/warmup_steps': num_warmup_steps,
        'training/max_epochs': NUM_EPOCHS,
        'training/mixed_precision': MIXED_PRECISION,
        'training/gradient_clipping': GRADIENT_CLIPPING,
        'hardware/device': str(device),
        'hardware/device_count': torch.cuda.device_count() if torch.cuda.is_available() else 1,
        'distributed/world_size': WORLD_SIZE if DISTRIBUTED else 1,
    }
    
    # Log hyperparams como texto
    hyperparams_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
    writer.add_text("Hyperparameters", hyperparams_text, 0)

# --- CONFIGURACIÓN PARA MODIFICACIÓN DE LEARNING RATE ---
# Flag para activar/desactivar la modificación del learning rate al cargar checkpoint
# USO: Cambiar MODIFY_LR_ON_LOAD a True y ajustar NEW_LEARNING_RATE según sea necesario
# Esto permite continuar el entrenamiento con un learning rate diferente sin perder el progreso
MODIFY_LR_ON_LOAD = False  # Cambiar a True para activar la modificación
NEW_LEARNING_RATE = 1e-4   # Nuevo valor del learning rate cuando MODIFY_LR_ON_LOAD es True

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

    # Modificar el learning rate si el flag está activado
    if MODIFY_LR_ON_LOAD:
        print(f"--- Modificando learning rate de {optimizer.param_groups[0]['lr']:.6f} a {NEW_LEARNING_RATE:.6f} ---")
        # Modificar el learning rate en el checkpoint del optimizer antes de cargarlo
        for param_group in checkpoint['optimizer_state_dict']['param_groups']:
            param_group['lr'] = NEW_LEARNING_RATE
        print(f"Learning rate modificado en el checkpoint del optimizer")
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Verificar que el learning rate se aplicó correctamente
    if MODIFY_LR_ON_LOAD:
        actual_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate después de cargar: {actual_lr:.6f}")
        if abs(actual_lr - NEW_LEARNING_RATE) > 1e-8:
            print(f"⚠️  Advertencia: El learning rate no se aplicó correctamente")
        else:
            print(f"✅ Learning rate modificado exitosamente a: {NEW_LEARNING_RATE:.6f}")
    
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

# === NUEVAS FUNCIONES DE ENTRENAMIENTO CON LIMPIEZA ===

def main_training():
    """Función principal de entrenamiento con manejo de limpieza"""
    global train_loader, val_loader, global_step, best_val_loss, patience_counter

    try:
        # El bucle de entrenamiento original aquí
        return True
    finally:
        # Limpieza explícita al finalizar
        try:
            if 'train_loader' in globals():
                del train_loader
            if 'val_loader' in globals():
                del val_loader
            torch.cuda.empty_cache()
            print("Limpieza post-entrenamiento completada.")
        except:
            pass

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
                
                # TensorBoard logging (solo en proceso principal)
                if writer is not None:
                    current_lr = scheduler.get_last_lr()[0]
                    train_loss = loss.item() * GRAD_ACCUM_STEPS
                    
                    # Métricas básicas de entrenamiento
                    writer.add_scalar('Loss/Train', train_loss, global_step)
                    writer.add_scalar('Learning_Rate/Current', current_lr, global_step)
                    writer.add_scalar('Training/Epoch_Progress', epoch + (i / len(progress)), global_step)
                    writer.add_scalar('Training/Global_Step', global_step, global_step)
                    
                    # Métricas de gradientes cada 50 pasos para evitar overhead
                    if global_step % 50 == 0:
                        total_norm = 0
                        param_count = 0
                        grad_norm = 0
                        param_norm = 0
                        
                        for p in model.parameters():
                            if p.requires_grad and p.grad is not None:
                                param_norm_sq = p.data.norm().item() ** 2
                                param_norm += param_norm_sq
                                
                                grad_norm_sq = p.grad.data.norm().item() ** 2
                                grad_norm += grad_norm_sq
                                
                                param_count += p.numel()
                        
                        if param_count > 0:
                            grad_norm = (grad_norm ** 0.5)
                            param_norm = (param_norm ** 0.5)
                            
                            writer.add_scalar('Gradients/Global_Norm', grad_norm, global_step)
                            writer.add_scalar('Parameters/Global_Norm', param_norm, global_step)
                            writer.add_scalar('Gradients/Param_Ratio', grad_norm / (param_norm + 1e-8), global_step)
                    
                    # Métricas de memoria GPU cada 100 pasos
                    if global_step % 100 == 0 and torch.cuda.is_available():
                        for gpu_id in range(torch.cuda.device_count()):
                            memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1e9  # GB
                            memory_cached = torch.cuda.memory_reserved(gpu_id) / 1e9     # GB
                            writer.add_scalar(f'GPU_{gpu_id}/Memory_Allocated_GB', memory_allocated, global_step)
                            writer.add_scalar(f'GPU_{gpu_id}/Memory_Cached_GB', memory_cached, global_step)
                            
                            # Calcular utilización de memoria
                            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                            memory_util = (memory_allocated / total_memory) * 100
                            writer.add_scalar(f'GPU_{gpu_id}/Memory_Utilization_%', memory_util, global_step)
                
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
        
        # Log expandido de validation metrics a TensorBoard
        if writer is not None:
            # Métricas de validación principales
            writer.add_scalar('Loss/Validation', avg_val_loss, global_step)
            writer.add_scalar('Loss/Best_Validation', best_val_loss, global_step)
            writer.add_scalar('Training/Patience_Counter', patience_counter, global_step)
            
            # Calcular diferencia con mejor loss
            val_loss_diff = avg_val_loss - best_val_loss
            writer.add_scalar('Loss/Val_vs_Best_Diff', val_loss_diff, global_step)
            
            # Ratio de validación vs entrenamiento (si tenemos train loss reciente)
            if hasattr(writer, '_last_train_loss'):
                train_val_ratio = avg_val_loss / (writer._last_train_loss + 1e-8)
                writer.add_scalar('Loss/Train_Val_Ratio', train_val_ratio, global_step)
                
                # Indicador de overfitting (val loss > train loss)
                overfitting_indicator = 1.0 if avg_val_loss > writer._last_train_loss else 0.0
                writer.add_scalar('Training/Overfitting_Signal', overfitting_indicator, global_step)
            
            # Guardar último train loss para próxima comparación
            if 'train_loss' in locals():
                writer._last_train_loss = train_loss
        
        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        # Guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Nueva mejor pérdida de validación. Guardando modelo en {BEST_MODEL_PATH}")
            torch.save(model_to_save.state_dict(), BEST_MODEL_PATH)
            patience_counter = 0
            
            # Log mejora del modelo a TensorBoard con histogramas
            if writer is not None:
                writer.add_scalar('Training/New_Best_Loss', best_val_loss, global_step)
                
                # Agregar histogramas de pesos cuando se guarda el mejor modelo
                for name, param in model_to_save.named_parameters():
                    if param.requires_grad and param.dim() > 1:  # Solo capas con peso significativo
                        # Limpiar nombre para TensorBoard
                        clean_name = name.replace('.', '/')
                        writer.add_histogram(f'Weights/{clean_name}', param.data, global_step)
                        
                        # Estadísticas de pesos
                        writer.add_scalar(f'Weight_Stats/{clean_name}_mean', param.data.mean().item(), global_step)
                        writer.add_scalar(f'Weight_Stats/{clean_name}_std', param.data.std().item(), global_step)
                        writer.add_scalar(f'Weight_Stats/{clean_name}_max', param.data.max().item(), global_step)
                        writer.add_scalar(f'Weight_Stats/{clean_name}_min', param.data.min().item(), global_step)
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

# Cerrar TensorBoard Writer
if writer is not None:
    writer.close()
    print("📊 TensorBoard Writer cerrado")

# Guardar modelo final
model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

if os.path.exists(BEST_MODEL_PATH):
    print(f"Cargando el mejor modelo desde '{BEST_MODEL_PATH}' para el guardado final.")
    model_to_save.load_state_dict(torch.load(BEST_MODEL_PATH))

model_to_save.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Modelo y tokenizador guardados en '{OUTPUT_DIR}'")

# Subir modelo a Hugging Face Hub
if HF_TOKEN:
    try:
        print(f"\nSubiendo modelo a Hugging Face Hub: {HF_REPO_ID}")
        api = HfApi()

        # Subir el modelo usando push_to_hub
        model_to_save.push_to_hub(
            HF_REPO_ID,
            token=HF_TOKEN,
            commit_message=f"Upload HRM-Text1 1B model trained on C4 dataset"
        )

        # Subir el tokenizador
        tokenizer.push_to_hub(
            HF_REPO_ID,
            token=HF_TOKEN,
            commit_message=f"Upload tokenizer for HRM-Text1 1B model"
        )

        print(f"✅ Modelo subido exitosamente a https://huggingface.co/{HF_REPO_ID}")

    except Exception as e:
        print(f"❌ Error al subir el modelo a Hugging Face: {e}")
        print("El modelo se guardó localmente pero no se pudo subir al Hub.")
else:
    print("\n⚠️  No se encontró HF_TOKEN. El modelo solo se guardó localmente.")
    print("Para subir a Hugging Face Hub, configura la variable de entorno HF_TOKEN.")

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

print("\n--- Script completado exitosamente ---")

# ==============================================================================
# --- EJEMPLOS DE CONFIGURACIONES PERSONALIZADAS PARA MODELO 1B ---
# ==============================================================================

"""
EJEMPLOS DE USO AVANZADO PARA MODELO 1B PARÁMETROS:

1. CONFIGURACIÓN RÁPIDA PARA PRUEBAS:
   DATASET_SUBSET_PERCENT = 1  # Solo 1% del dataset
   ACTIVE_DATASET = "openwebtext"  # Dataset pequeño y rápido

2. CONFIGURACIÓN PARA CALIDAD MÁXIMA (1B):
   ACTIVE_DATASET = "high_quality_1b"  # Mezcla de alta calidad
   DATASET_SUBSET_PERCENT = 15  # 15% del dataset

3. CONFIGURACIÓN MULTILINGÜE BALANCEADA (1B):
   ACTIVE_DATASET = "multilingual_balanced_1b"
   DATASET_SUBSET_PERCENT = 12  # 12% del dataset

4. CONFIGURACIÓN EXPERIMENTAL COMPLETA (1B):
   ACTIVE_DATASET = "experimental_full_1b"
   DATASET_SUBSET_PERCENT = 20  # 20% del dataset

5. CONFIGURACIÓN PERSONALIZADA PARA INVESTIGACIÓN:
   CUSTOM_MIX_RATIOS = {
       "research_1b": {
           "slimpajama_en": 0.4,  # 40% datos de alta calidad
           "fineweb": 0.3,        # 30% contenido muy filtrado
           "c4": 0.2,             # 20% diversidad multilingüe
           "pile": 0.1            # 10% contenido especializado
       }
   }
   ACTIVE_DATASET = "research_1b"

6. CONFIGURACIÓN PARA ESPAÑOL (1B):
   ACTIVE_DATASET = "mixed_es"
   DATASET_SUBSET_PERCENT = 10

7. CONFIGURACIÓN PARA CONVERSACIONES (1B):
   ACTIVE_DATASET = "human_conversations"  # Dataset de Kaggle
   DATASET_SUBSET_PERCENT = 50  # Usar más porcentaje para datasets pequeños

8. CONFIGURACIÓN MIXTA CON CONVERSACIONES (1B):
   ACTIVE_DATASET = "conversation_mix_1b"  # Mezcla enfocada en chat
   DATASET_SUBSET_PERCENT = 15

NOTAS IMPORTANTES PARA MODELO 1B:
- Requiere al menos 16GB de VRAM para entrenamiento
- Los porcentajes más altos requieren más tiempo y memoria
- Las mezclas personalizadas deben sumar 1.0 (100%)
- El script valida automáticamente las configuraciones
- Usa "slimpajama" completo solo si tienes suficiente almacenamiento
- El contexto de 2048 tokens requiere más memoria que el modelo 99M
- Recomendado usar gradient_checkpointing=True para ahorrar memoria
- Flash Attention mejora significativamente la velocidad si está disponible
- Para usar datasets de Kaggle, instala: pip install kagglehub
- Los datasets de conversaciones son ideales para modelos de chat

ENTRENAMIENTO SECUENCIAL CON MÚLTIPLES DATASETS:

⚠️  IMPORTANTE: CONFIGURACIÓN DE DIRECTORIO PARA ENTRENAMIENTO SECUENCIAL ⚠️

PROBLEMA: Por defecto, cada dataset usa un directorio diferente, perdiendo checkpoints.
SOLUCIÓN: Activar SEQUENTIAL_TRAINING = True

Para continuar entrenando con otro dataset después de terminar:

1. MÉTODO AUTOMÁTICO (Cambiar configuración):
   - ANTES DE EMPEZAR: Configura SEQUENTIAL_TRAINING = True
   - Termina el entrenamiento actual
   - Cambia ACTIVE_DATASET al nuevo dataset deseado
   - Ajusta MODIFY_LR_ON_LOAD = True y NEW_LEARNING_RATE = 1e-5
   - Reduce NUM_EPOCHS = 1 para fine-tuning
   - Ejecuta el script - cargará automáticamente el checkpoint

2. MÉTODO MANUAL (Cargar modelo y continuar):
   - Usa la función load_and_continue_training() (ver abajo)
   - Especifica el modelo base y el nuevo dataset
   - Control total sobre hiperparámetros

3. EJEMPLOS DE SECUENCIAS RECOMENDADAS:
   a) Entrenamiento base → Especialización:
      "c4" → "human_conversations" (para chat)
      "mixed" → "spanish" (para español)
   
   b) Calidad progresiva:
      "c4" → "fineweb" → "human_conversations"
   
   c) Multilingüe escalonado:
      "c4" → "mixed" → "multilingual_balanced_1b"

4. CONFIGURACIÓN RECOMENDADA PARA FINE-TUNING:
   - Learning rate: 1/10 del original (3e-4 → 3e-5)
   - Épocas: 1-2 épocas máximo
   - Subset: 50-100% del nuevo dataset
   - Early stopping: Patience más baja (1-2)

def load_and_continue_training(base_model_path, new_dataset, new_lr=1e-5, epochs=1):
    \"\"\"
    Función para continuar entrenamiento con un dataset diferente
    
    Args:
        base_model_path: Ruta al modelo entrenado
        new_dataset: Nombre del nuevo dataset a usar
        new_lr: Nuevo learning rate (más bajo para fine-tuning)
        epochs: Número de épocas para el nuevo dataset
    \"\"\"
    # Esta función se implementaría para cargar el modelo base
    # y continuar entrenamiento con el nuevo dataset
    pass

EJEMPLO DE USO SECUENCIAL CORRECTO:

CONFIGURACIÓN INICIAL (CRUCIAL):
SEQUENTIAL_TRAINING = True  # ⭐ ESTO ES ESENCIAL
ACTIVE_DATASET = "c4"
NUM_EPOCHS = 2

PASOS:
1. python hrm_training_large_1b.py  # Primera ejecución con C4
2. [Esperar a que termine completamente]
3. Editar configuración sin cambiar SEQUENTIAL_TRAINING:
   - ACTIVE_DATASET = "human_conversations"
   - MODIFY_LR_ON_LOAD = True
   - NEW_LEARNING_RATE = 1e-5
   - NUM_EPOCHS = 1
4. python hrm_training_large_1b.py  # Continuará desde checkpoint

DIRECTORIOS USADOS:
- SEQUENTIAL_TRAINING = True:  ./HRM_Models/hrm_text1_c4_1b_output/
- SEQUENTIAL_TRAINING = False: ./HRM_Models/hrm_text1_[dataset]_1b_output/

¡SIEMPRE usar SEQUENTIAL_TRAINING = True para entrenamiento continuo!

"""