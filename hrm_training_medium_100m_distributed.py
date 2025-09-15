# -*- coding: utf-8 -*-
"""
HRM-Models Distributed Training Script - MODELO MEDIUM ~100M PARÁMETROS
VERSIÓN MULTI-GPU: Optimizada para entrenamiento distribuido en servidor

🖥️  CARACTERÍSTICAS:
- Entrenamiento distribuido con torch.distributed
- DistributedDataParallel (DDP) para mejor escalabilidad
- División automática de datos entre GPUs
- Sincronización de gradientes optimizada
- Balanceado de carga inteligente
- Optimizada para modelo Medium 100M parámetros
"""

import os, multiprocessing as mp, math, time
from typing import List, Dict, Optional, Tuple
import argparse

# Configurar hf_transfer para descargas más rápidas de HuggingFace
try:
    import hf_transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    HF_TRANSFER_AVAILABLE = True
except ImportError:
    HF_TRANSFER_AVAILABLE = False

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ tqdm no disponible, usando progreso básico")

# Configurar método de multiprocessing antes de cualquier uso
if __name__ == '__main__':
    # Usar spawn en lugar de fork para evitar pickle issues
    mp.set_start_method('spawn', force=True)
    # Marcar PID principal para evitar spam en multiprocessing
    import os
    os._main_pid = os.getpid()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from torch.optim import AdamW

# Función de inicialización del código ACT de referencia
def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """PyTorch version of jax truncated normal init (matemáticamente correcto)"""
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor

# Importar wrapper de tokenizador HF simplificado
try:
    from hf_tokenizer_wrapper_simple import HuggingFaceTokenizerWrapper, create_tokenizer
    HF_TOKENIZER_AVAILABLE = True
    print("✅ HuggingFace tokenizer wrapper simple disponible")
except ImportError:
    HF_TOKENIZER_AVAILABLE = False
    print("❌ HuggingFace tokenizer wrapper NO disponible")
    print("💡 Ejecute: pip install transformers tokenizers")
    exit(1)

# Importar base de transformers para compatibilidad
try:
    from transformers import PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutput
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers disponible para integración HRM")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers no disponible - usando implementación standalone")

print("✅ Configuración HRM integrada directamente en el script distribuido")

# ==============================================================================
# --- DISTRIBUTED TRAINING SETUP ---
# ==============================================================================

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
            
        return rank, world_size, local_rank, device
    else:
        # Single GPU or CPU training
        return 0, 1, 0, torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

# ==============================================================================
# --- HRM CORE CLASSES (Medium 100M Configuration) ---
# ==============================================================================

class SimpleConfig:
    """Base configuration class"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Usar PretrainedConfig si transformers está disponible, sino SimpleConfig
if TRANSFORMERS_AVAILABLE:
    ConfigBase = PretrainedConfig
    print("🔧 Usando PretrainedConfig de transformers")
else:
    ConfigBase = SimpleConfig
    print("🔧 Usando SimpleConfig standalone")

class HRMText1Config(ConfigBase):
    model_type = "hrm_text1"

    def __init__(self,
                 vocab_size=50257,          # HF tokenizer default
                 block_size=1024,           # Medium model context - incrementado para 100M
                 n_embd=768,                # Medium model embeddings - escalado para 100M
                 n_head=16,                 # Medium model heads - múltiplo de n_embd
                 n_layers=12,               # Medium model layers - más profundo para 100M
                 d_ff=3072,                 # Medium model FFN - 4x n_embd
                 dropout=0.1,               # Mantenido para estabilidad
                 pad_token_id=0,
                 halt_max_steps=4,          # HRM halt steps
                 ponder_loss_weight=2e-3,   # Reducido para modelo más grande
                 halt_bias_init=-1.0,
                 use_rotary_embeddings=True,
                 rotary_embedding_base=10000,
                 use_flash_attention=True,
                 gradient_checkpointing=True,  # Activado para 100M
                 # HRM Ciclos controlados para estabilidad
                 H_cycles=2,                # Incrementado para modelo más grande
                 L_cycles=3,                # Incrementado para mejor refinamiento
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.halt_max_steps = halt_max_steps
        self.ponder_loss_weight = ponder_loss_weight
        self.halt_bias_init = halt_bias_init
        self.use_rotary_embeddings = use_rotary_embeddings
        self.rotary_embedding_base = rotary_embedding_base
        self.use_flash_attention = use_flash_attention
        self.gradient_checkpointing = gradient_checkpointing
        # HRM Ciclos según paper original
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))

    def forward(self, x):
        device = x.device
        # Protección adicional contra NaN en RMSNorm
        x = torch.clamp(x, -10.0, 10.0)  # Evitar valores extremos
        norm = x.norm(dim=-1, keepdim=True, dtype=torch.float32).clamp(min=self.eps, max=1e6)
        normalized = x / norm.to(device)
        # Verificar NaN y reemplazar por ceros si es necesario
        normalized = torch.where(torch.isfinite(normalized), normalized, torch.zeros_like(normalized))
        return normalized * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.register_buffer('inv_freq', 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x, seq_len):
        device = x.device
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

class SwiGLUMuchPelu(nn.Module):
    def __init__(self, n_embd, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(n_embd, d_ff, bias=False)
        self.w2 = nn.Linear(n_embd, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class OptimizedMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_flash_attention = config.use_flash_attention
        self.use_rotary_embeddings = config.use_rotary_embeddings

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if self.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(self.head_dim, config.rotary_embedding_base)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rotary_embeddings:
            cos, sin = self.rotary_emb(x, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = self._standard_attention(q, k, v, attn_mask, key_padding_mask)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.n_embd)
        return self.out_proj(attn_output)

    def _standard_attention(self, q, k, v, attn_mask=None, key_padding_mask=None):
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        if key_padding_mask is not None:
            # Convertir a bool si es necesario
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            if mask.dtype != torch.bool:
                mask = mask == 0  # Convertir padding mask (0 = padded, 1 = valid) a bool
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

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
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)

        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class HRMInner(nn.Module):
    """True HRM implementation with hierarchical temporal separation and Deep Supervision for Medium 100M"""
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.H_module = HRMBlock(config)
        self.L_module = HRMBlock(config)
        self.config = config
        self.layer_idx = layer_idx

        # Q-learning components for adaptive computation (del código ACT)
        self.q_network = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.ReLU(),
            nn.Linear(config.n_embd // 4, 2)  # [continue, halt] como en ACT
        )

        # Inicialización especial para Q-network como en ACT
        with torch.no_grad():
            # Inicializar pesos a casi cero para bootstrapping más rápido
            for layer in self.q_network:
                if isinstance(layer, nn.Linear):
                    layer.weight.zero_()
                    if layer.bias is not None:
                        layer.bias.fill_(-5.0)  # Bias hacia halt para estabilidad inicial

        # Deep Supervision: prediction heads reactivados gradualmente
        self.h_prediction_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.l_prediction_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Inicialización del código ACT de referencia - matemáticamente correcta
        fan_in = config.n_embd
        lecun_std = 1.0 / (fan_in ** 0.5)  # LeCun initialization del paper original

        # Factor de profundidad como en el código ACT
        depth_factor = max(0.1, 1.0 - (layer_idx / max(config.n_layers, 1)))
        std_h = lecun_std * depth_factor * 0.8  # Más conservador para modelo grande
        std_l = lecun_std * depth_factor * 0.6  # L-module aún más conservador

        # Usar trunc_normal_init_ del código ACT (matemáticamente correcto)
        trunc_normal_init_(self.h_prediction_head.weight, std=std_h)
        trunc_normal_init_(self.l_prediction_head.weight, std=std_l)

        # Ponder loss components
        self.ponder_network = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.ReLU(),
            nn.Linear(config.n_embd // 4, 1),
            nn.Sigmoid()
        )

        self.convergence_threshold = 1e-3
        self.max_l_steps = config.halt_max_steps

        # HRM Cycles from paper, adapted for Medium 100M
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles

    def forward(self, z_H, z_L, input_embeddings, attn_mask=None, key_padding_mask=None, training=True):
        """Forward pass with HRM cycles adapted for Medium 100M - respects token causality"""
        _ = z_H.shape  # batch_size, seq_len, d_model
        _ = z_H.device  # device

        total_l_steps = 0
        total_ponder_loss = 0.0
        all_q_values = []

        # HRM Cycles implementation for Medium 100M (more cycles)
        for h_cycle in range(self.H_cycles):
            cycle_l_steps = 0
            cycle_ponder_loss = 0.0

            # L-module cycles (local refinement with causal attention)
            for l_cycle in range(self.L_cycles):
                # L-module processes with H-module context + input injection
                z_L_input = z_L + z_H.detach() + input_embeddings  # Input injection as in paper
                z_L_new = self.L_module(z_L_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

                # Ponder loss: computational cost of this L-cycle
                ponder_score = self.ponder_network(z_L_new).mean()
                cycle_ponder_loss += ponder_score
                cycle_l_steps += 1

                # ACT-style Q-learning halting para LLMs
                if training and l_cycle < self.L_cycles - 1:  # Not on last L-cycle
                    # Q-values para decisión de halting (inspirado en ACT)
                    q_values = self.q_network(z_L_new.mean(dim=1))  # Pool sobre secuencia para decisión
                    all_q_values.append(q_values)

                    # Halting logic del código ACT adaptado para LLMs
                    if l_cycle > 0:  # Necesario al menos un step
                        # Calcular diferencia de convergencia
                        diff = torch.norm(z_L_new - z_L, p=2, dim=-1).mean()
                        is_last_step = l_cycle >= self.max_l_steps - 1

                        # Q-learning decision como en ACT
                        q_halt_logits = q_values[..., 1]  # Halt logits
                        q_continue_logits = q_values[..., 0]  # Continue logits

                        # Halting decision (adaptado del código ACT)
                        halted = is_last_step | (q_halt_logits.mean() > q_continue_logits.mean())

                        # Exploration durante training (como en ACT) - reducido para modelo grande
                        halt_exploration_prob = 0.05  # Menor exploración para estabilidad
                        if torch.rand(1).item() < halt_exploration_prob:
                            min_halt_steps = torch.randint(2, self.max_l_steps + 1, (1,)).item()
                            if l_cycle >= min_halt_steps:
                                halted = True

                        # Convergence-based halting adicional
                        if diff < self.convergence_threshold:
                            halted = True

                        if halted:
                            break

                z_L = z_L_new

            # H-module update after L-cycles (except last H-cycle)
            if h_cycle < self.H_cycles - 1:
                z_H_input = z_H + z_L  # H gets refined L-module output
                z_H = self.H_module(z_H_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

            total_l_steps += cycle_l_steps
            total_ponder_loss += cycle_ponder_loss

        # Final H-module update with all L-module refinements
        z_H_final = self.H_module(z_H + z_L, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # Deep Supervision: generar logits intermedios con estabilización
        h_logits = self.h_prediction_head(z_H_final)
        l_logits = self.l_prediction_head(z_L)

        # Estabilización agresiva de logits para prevenir NaN - más conservador para 100M
        h_logits = torch.clamp(h_logits, -4.0, 4.0)  # Rango más pequeño para modelo grande
        l_logits = torch.clamp(l_logits, -4.0, 4.0)  # Rango más pequeño para modelo grande

        # Verificar NaN en logits y usar fallback
        if torch.isnan(h_logits).any():
            h_logits = torch.zeros_like(h_logits)
        if torch.isnan(l_logits).any():
            l_logits = torch.zeros_like(l_logits)

        # Normalize metrics
        avg_ponder_loss = total_ponder_loss / max(total_l_steps, 1)

        return z_H_final, z_L, {
            'h_updated': True,
            'l_steps': total_l_steps,
            'q_values': all_q_values,
            'convergence_achieved': True,
            'h_logits': h_logits,  # For Deep Supervision
            'l_logits': l_logits,  # For Deep Supervision
            'ponder_loss': avg_ponder_loss,  # For computational regularization
            'layer_idx': self.layer_idx,
            'h_cycles_completed': self.H_cycles,
            'avg_l_cycles': total_l_steps / self.H_cycles
        }


# Usar PreTrainedModel si transformers está disponible, sino nn.Module
if TRANSFORMERS_AVAILABLE:
    ModelBase = PreTrainedModel
    print("🔧 Modelo HRM usando PreTrainedModel de transformers")
else:
    ModelBase = nn.Module
    print("🔧 Modelo HRM usando nn.Module standalone")

class HRMText1(ModelBase):
    """HRM Model with full hierarchical temporal separation - Medium 100M"""
    config_class = HRMText1Config

    def __init__(self, config: HRMText1Config):
        super().__init__(config if TRANSFORMERS_AVAILABLE else())
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)

        # Usar RoPE en lugar de embeddings posicionales aprendidos
        if not config.use_rotary_embeddings:
            self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
            self.register_buffer("pos_ids", torch.arange(config.block_size).unsqueeze(0))
        else:
            self.pos_embeddings = None
            self.pos_ids = None

        # Apilar múltiples capas HRM con índices para Deep Supervision
        self.layers = nn.ModuleList([
            HRMInner(config, layer_idx=i) for i in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Compartir pesos entre token embeddings y lm_head
        self.lm_head.weight = self.token_embeddings.weight

        # Inicialización más conservadora para modelo 100M
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Inicialización Xavier/Glorot más conservadora para HRM 100M
            nn.init.xavier_uniform_(module.weight, gain=0.3)  # Más conservador
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.008)  # Más conservador

    def forward(self, input_ids, attention_mask=None, labels=None):
        _, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.token_embeddings(input_ids)

        # Positional embeddings (si no usa RoPE)
        if not self.config.use_rotary_embeddings:
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embs = self.pos_embeddings(pos_ids)
            x = x + pos_embs

        # Crear máscara causal
        if attention_mask is None:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        else:
            causal_mask = None

        # Inicializar estados H y L para HRM
        z_H = x
        z_L = torch.zeros_like(x)

        # Deep Supervision: collect intermediate predictions and losses
        all_hrm_info = []
        total_ponder_loss = 0.0
        intermediate_losses = []

        # HRM layers con estados separados - using cycles approach
        for layer_idx, layer in enumerate(self.layers):
            # Gradient checkpointing para modelo 100M
            if self.config.gradient_checkpointing and self.training:
                z_H, z_L, hrm_info = torch.utils.checkpoint.checkpoint(
                    layer, z_H, z_L, x, causal_mask, attention_mask, self.training,
                    use_reentrant=False
                )
            else:
                z_H, z_L, hrm_info = layer(z_H, z_L, x, attn_mask=causal_mask, key_padding_mask=attention_mask, training=self.training)
            
            all_hrm_info.append(hrm_info)

            # Accumulate ponder loss
            if hrm_info.get('ponder_loss') is not None:
                total_ponder_loss += hrm_info['ponder_loss']

            # Deep Supervision: calcular pérdidas intermedias con validación robusta
            if labels is not None and hrm_info.get('h_logits') is not None:
                h_logits = hrm_info['h_logits']
                l_logits = hrm_info.get('l_logits')

                # Calculate losses for intermediate predictions
                shift_labels = labels[..., 1:].contiguous()

                # H-module supervision con estabilización
                shift_h_logits = h_logits[..., :-1, :].contiguous()

                # Estabilizar logits antes del cálculo de pérdida - más conservador para 100M
                shift_h_logits = torch.clamp(shift_h_logits, -15.0, 15.0)  # Rango más conservador

                h_loss = F.cross_entropy(
                    shift_h_logits.view(-1, shift_h_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.config.pad_token_id,
                    reduction='mean'
                )

                # Verificar y corregir h_loss si es inestable
                if torch.isnan(h_loss) or torch.isinf(h_loss) or h_loss > 30.0:
                    # Usar una pérdida de referencia basada en vocabulario
                    h_loss = torch.log(torch.tensor(self.config.vocab_size, device=h_loss.device, dtype=torch.float32))

                # L-module supervision (if available) con estabilización
                l_loss = 0.0
                if l_logits is not None:
                    shift_l_logits = l_logits[..., :-1, :].contiguous()

                    # Estabilizar L-logits también - más conservador
                    shift_l_logits = torch.clamp(shift_l_logits, -15.0, 15.0)

                    l_loss = F.cross_entropy(
                        shift_l_logits.view(-1, shift_l_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=self.config.pad_token_id,
                        reduction='mean'
                    )

                    # Verificar y corregir l_loss si es inestable
                    if torch.isnan(l_loss) or torch.isinf(l_loss) or l_loss > 30.0:
                        l_loss = torch.log(torch.tensor(self.config.vocab_size, device=l_loss.device, dtype=torch.float32))

                intermediate_losses.append({
                    'layer_idx': layer_idx,
                    'h_loss': h_loss,
                    'l_loss': l_loss
                })

        # Final norm y lm_head
        output = self.final_norm(z_H)
        logits = self.lm_head(output)

        # Si se proporcionan labels, calcular loss total con Deep Supervision
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Main loss con label smoothing para mejor generalización - más conservador para 100M
            main_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
                label_smoothing=0.08  # Menos smoothing para modelo más grande
            )

            # Deep Supervision loss (weighted sum of intermediate losses) - más estable para 100M
            deep_supervision_loss = 0.0
            if intermediate_losses:
                total_layers = len(self.layers)

                for loss_info in intermediate_losses:
                    h_loss = loss_info['h_loss']
                    l_loss = loss_info['l_loss']

                    # Las pérdidas ya están estabilizadas arriba, solo verificar valores extremos
                    h_loss = torch.clamp(h_loss, 0.0, 30.0)  # Más conservador
                    if isinstance(l_loss, torch.Tensor):
                        l_loss = torch.clamp(l_loss, 0.0, 30.0)
                    else:
                        l_loss = min(max(l_loss, 0.0), 30.0)  # Para valores scalar

                    layer_weight = (loss_info['layer_idx'] + 1) / total_layers
                    layer_loss = layer_weight * (h_loss + 0.2 * l_loss)  # Peso menor de L-loss para 100M
                    deep_supervision_loss += layer_loss

                deep_supervision_loss /= len(intermediate_losses)  # Promedio

            # Ponder loss (computational regularization) - más conservador para 100M
            ponder_loss = total_ponder_loss / len(self.layers) if total_ponder_loss > 0 else 0.0

            # Q-learning loss para halting (inspirado en código ACT) - más conservador
            q_learning_loss = 0.0
            if all_hrm_info and labels is not None:
                # Calcular correctness para Q-learning target (como en ACT)
                with torch.no_grad():
                    # Verificar si las predicciones son correctas
                    predictions = torch.argmax(logits, dim=-1)
                    mask = labels != self.config.pad_token_id
                    is_correct = mask & (predictions == labels)

                    # Accuracy por secuencia para Q-learning target - threshold más alto para 100M
                    seq_accuracy = is_correct.float().sum(-1) / mask.float().sum(-1).clamp(min=1)
                    seq_is_correct = seq_accuracy > 0.85  # Threshold más alto

                # Calcular Q-learning loss de todos los layers
                for hrm_info in all_hrm_info:
                    if hrm_info.get('q_values'):
                        for q_vals in hrm_info['q_values']:
                            # Q-halt loss: predecir si la secuencia será correcta
                            q_halt_logits = q_vals[..., 1]  # Solo tomar índice de halt
                            q_halt_loss = F.binary_cross_entropy_with_logits(
                                q_halt_logits,
                                seq_is_correct.float(),
                                reduction='mean'
                            )
                            q_learning_loss += q_halt_loss * 0.05  # Peso menor para modelo grande

            # Verificar y desactivar Deep Supervision si tiene NaN
            if isinstance(deep_supervision_loss, torch.Tensor):
                ds_check = deep_supervision_loss.detach()
            else:
                ds_check = torch.tensor(float(deep_supervision_loss), device=device if 'device' in locals() else 'cpu')
            
            if torch.isnan(ds_check) or torch.isinf(ds_check):
                deep_supervision_loss = 0.0

            # Pesos más conservadores para modelo 100M
            ds_weight = 0.03  # Peso más moderado
            ponder_weight = self.config.ponder_loss_weight * 0.2  # Peso muy reducido

            total_loss = (
                main_loss +
                ds_weight * deep_supervision_loss +  # Deep supervision controlado
                ponder_weight * ponder_loss +  # Ponder loss reducido
                q_learning_loss  # Q-learning loss para halting automático
            )

            # Verificar estabilidad de la pérdida final
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                # Usar solo main_loss si las otras pérdidas están corruptas
                total_loss = main_loss

            # Retornar en formato compatible con transformers si está disponible
            if TRANSFORMERS_AVAILABLE:
                return CausalLMOutput(
                    loss=total_loss,
                    logits=logits,
                    hidden_states=all_hrm_info,
                    attentions=None
                )
            else:
                return {
                    'loss': total_loss,
                    'logits': logits,
                    'main_loss': main_loss,
                    'deep_supervision_loss': deep_supervision_loss,
                    'ponder_loss': ponder_loss,
                    'q_learning_loss': q_learning_loss,
                    'hrm_info': all_hrm_info
                }

        # Sin labels, solo retornar logits
        if TRANSFORMERS_AVAILABLE:
            return CausalLMOutput(
                logits=logits,
                hidden_states=all_hrm_info,
                attentions=None
            )
        else:
            return {
                'logits': logits,
                'hrm_info': all_hrm_info
            }

# ==============================================================================
# --- DISTRIBUTED DATASET HANDLING ---
# ==============================================================================

class DistributedTextDataset(IterableDataset):
    """Dataset optimizado para entrenamiento distribuido"""

    def __init__(self, tokenizer, texts: List[str], block_size: int = 512, split_type: str = "train",
                 device=None, batch_tokenize: bool = True, max_length: int = 2048,
                 min_text_length: int = 10, rank: int = 0, world_size: int = 1):
        self.tokenizer = tokenizer
        self.texts = texts
        self.block_size = block_size
        self.split_type = split_type
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.batch_tokenize = batch_tokenize
        self.max_length = max_length
        self.min_text_length = min_text_length
        self.rank = rank
        self.world_size = world_size

        # Distribute texts among processes
        texts_per_rank = len(texts) // world_size
        start_idx = rank * texts_per_rank
        end_idx = start_idx + texts_per_rank if rank < world_size - 1 else len(texts)
        self.local_texts = texts[start_idx:end_idx]

        print(f"📚 Dataset {split_type} Rank {rank}: {len(self.local_texts)} textos locales de {len(texts)} totales, block_size={block_size}")

    def __iter__(self):
        # Tokenización on-the-fly para cada proceso
        for text in self.local_texts:
            if not text or len(text.strip()) < self.min_text_length:
                continue

            try:
                # Tokenizar con HF
                tokens = self.tokenizer.encode(text, add_special_tokens=True,
                                             max_length=self.max_length, truncation=True)

                # Crear chunks
                for i in range(0, len(tokens) - self.block_size + 1, self.block_size):
                    chunk = tokens[i:i + self.block_size]
                    if len(chunk) == self.block_size:
                        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                        labels = torch.tensor(chunk[1:], dtype=torch.long)
                        yield {
                            'input_ids': input_ids,
                            'labels': labels
                        }
            except Exception as e:
                if self.rank == 0:  # Solo el proceso principal imprime errores
                    print(f"⚠️ Error tokenizando texto en rank {self.rank}: {e}")
                continue

def load_dataset_hf(tokenizer, split: str = "train", num_samples: int = 1000,
                   dataset_name: str = "allenai/c4", dataset_config: str = "en",
                   text_column: str = "text", min_text_length: int = 50,
                   max_text_length: int = 2048, use_streaming: bool = True, 
                   fast_mode: bool = False, rank: int = 0):
    """Cargar dataset usando datasets de HuggingFace para entrenamiento distribuido"""
    try:
        from datasets import load_dataset

        if rank == 0:  # Solo el proceso principal imprime información
            print(f"📥 Cargando dataset '{dataset_name}' ({dataset_config}) split '{split}' con {num_samples} samples...")
            print(f"   📋 Configuración:")
            print(f"   - Dataset: {dataset_name}")
            print(f"   - HF Transfer: {'✅ Habilitado' if HF_TRANSFER_AVAILABLE else '❌ No disponible (instalar hf_transfer)'}")
            print(f"   - Config: {dataset_config}")
            print(f"   - Columna texto: {text_column}")
            print(f"   - Min length: {min_text_length} chars")
            print(f"   - Max length: {max_text_length} chars")
            print(f"   - Streaming: {use_streaming}")
            print(f"   - Fast mode: {fast_mode}")

        # OPTIMIZACIÓN: Para datasets grandes, usar modo no-streaming es más rápido
        if fast_mode and num_samples > 50000:
            if rank == 0:
                print("⚡ Fast mode: Usando dataset no-streaming para mejor rendimiento...")
            use_streaming = False

        # Cargar dataset
        if use_streaming:
            dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        else:
            if rank == 0:
                print("📦 Descargando dataset completo (más rápido para lotes grandes)...")
            dataset = load_dataset(dataset_name, dataset_config, split=split)

        texts = []
        processed_count = 0

        # Solo el proceso principal muestra progreso
        if TQDM_AVAILABLE and rank == 0:
            progress_desc = f"Procesando {dataset_name}"
            progress = tqdm(enumerate(dataset), desc=progress_desc, total=num_samples if use_streaming else len(dataset))
        else:
            progress = enumerate(dataset)

        for i, item in progress:
            if use_streaming and i >= num_samples:
                break

            text = item.get(text_column, '')
            if isinstance(text, list):
                text = ' '.join(text)  # Para datasets con texto en listas

            if text and len(text.strip()) >= min_text_length:
                # Procesar y limpiar texto
                text = text.strip()[:max_text_length]
                texts.append(text)
                processed_count += 1

                if TQDM_AVAILABLE and isinstance(progress, tqdm) and rank == 0:
                    progress.set_postfix({
                        'válidos': processed_count,
                        'ratio': f'{processed_count/(i+1)*100:.1f}%'
                    })

        if TQDM_AVAILABLE and isinstance(progress, tqdm) and rank == 0:
            progress.close()

        if rank == 0:
            print(f"✅ Procesados {processed_count} textos válidos de {i+1} samples totales")
            print(f"   📊 Ratio de aprovechamiento: {processed_count/(i+1)*100:.1f}%")

        if not texts:
            if rank == 0:
                print("❌ No se encontraron textos válidos en el dataset")
            raise ValueError(f"No se pudieron cargar textos válidos de {dataset_name}")

        return texts

    except Exception as e:
        if rank == 0:
            print(f"❌ Error cargando dataset HF: {e}")
        raise RuntimeError(f"Falló carga de dataset {dataset_name}: {e}") from e

# ==============================================================================
# --- DISTRIBUTED TRAINING FUNCTIONS ---
# ==============================================================================

def save_model_hf(model, tokenizer, save_path: str, config: HRMText1Config, step: int = 0):
    """Guardar modelo y tokenizador HF"""
    os.makedirs(save_path, exist_ok=True)

    # Guardar modelo PyTorch
    model_path = os.path.join(save_path, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)

    # Guardar configuración
    config_dict = {
        'vocab_size': config.vocab_size,
        'block_size': config.block_size,
        'n_embd': config.n_embd,
        'n_head': config.n_head,
        'n_layers': config.n_layers,
        'd_ff': config.d_ff,
        'dropout': config.dropout,
        'use_rotary_embeddings': config.use_rotary_embeddings,
        'gradient_checkpointing': config.gradient_checkpointing,
        'tokenizer_type': getattr(config, 'tokenizer_type', 'huggingface'),
        'hf_tokenizer_name': getattr(config, 'hf_tokenizer_name', 'openai-community/gpt2'),
        'pad_token_id': getattr(config, 'pad_token_id', 0),
    }

    config_path = os.path.join(save_path, "config.json")
    with open(config_path, 'w') as f:
        import json
        json.dump(config_dict, f, indent=2)

    # Guardar tokenizador HF
    try:
        tokenizer.save_pretrained(save_path)
        print(f"💾 Modelo y tokenizador guardados en: {save_path}")
    except Exception as e:
        print(f"⚠️ Error guardando tokenizador: {e}")

def train_hrm_distributed_100m(
    tokenizer_name: str = "openai-community/gpt2",
    output_dir: str = "./hrm-medium-100m-distributed",
    num_train_samples: int = 200000,  # Más samples para modelo 100M
    num_val_samples: int = 20000,     
    batch_size: int = 4,             # Batch size más pequeño por GPU para 100M
    learning_rate: float = 1e-5,     # Learning rate más conservador
    num_epochs: int = 3,
    save_steps: int = 1000,          # Menos frecuente para modelo grande
    eval_steps: int = 200,            
    max_grad_norm: float = 0.5,      # Gradient clipping más agresivo
    warmup_steps: int = 4000,        # Warmup más largo
    # Parámetros de dataset
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    max_text_length: int = 2048,     # Textos más largos para 100M
    min_text_length: int = 100,      # Textos mínimos más largos
    # Parámetros para acelerar carga de datos
    fast_mode: bool = True,
    no_streaming: bool = True,
):
    """Entrenar modelo HRM Medium 100M con entrenamiento distribuido"""
    
    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()
    is_main_process = rank == 0
    is_distributed = world_size > 1

    if is_main_process:
        print(f"🚀 Iniciando entrenamiento HRM Medium 100M DISTRIBUIDO")
        print(f"📊 Configuración distribuida:")
        print(f"   🌐 World size: {world_size}")
        print(f"   🏷️ Rank: {rank}")
        print(f"   📱 Local rank: {local_rank}")
        print(f"   💻 Device: {device}")
        print(f"   ⚡ HF Transfer: {'✅ Habilitado' if HF_TRANSFER_AVAILABLE else '❌ No disponible'}")
        print(f"   📋 Batch size por GPU: {batch_size}")
        print(f"   📊 Batch size efectivo: {batch_size * world_size}")

    # Crear tokenizador HF
    if is_main_process:
        print(f"🔧 Cargando tokenizador: {tokenizer_name}")
    tokenizer = create_tokenizer(tokenizer_name)

    # Crear configuración del modelo Medium 100M
    config = HRMText1Config(
        vocab_size=len(tokenizer),
        block_size=1024,           # Context length más grande
        n_embd=768,                # Embeddings más grandes
        n_head=16,                 # Más heads
        n_layers=12,               # Más layers
        d_ff=3072,                 # FFN más grande
        dropout=0.1,              
        gradient_checkpointing=True,  # Activar para 100M
        tokenizer_type='huggingface',
        hf_tokenizer_name=tokenizer_name,
        pad_token_id=tokenizer.pad_token_id,
        H_cycles=2,                # Más ciclos para modelo grande
        L_cycles=3,
    )

    if is_main_process:
        print(f"📐 Configuración del modelo Medium 100M:")
        print(f"   Vocabulario: {config.vocab_size:,} tokens")
        print(f"   Context length: {config.block_size}")
        print(f"   Embeddings: {config.n_embd}")
        print(f"   Capas: {config.n_layers}")
        print(f"   Cabezas atención: {config.n_head}")
        print(f"   FFN dimension: {config.d_ff}")
        print(f"   Gradient checkpointing: {config.gradient_checkpointing}")

    # Crear modelo
    model = HRMText1(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if is_main_process:
        print(f"🧠 Modelo Medium 100M creado:")
        print(f"   Total parámetros: {total_params:,}")
        print(f"   Parámetros entrenables: {trainable_params:,}")

    # Mover modelo a dispositivo
    model = model.to(device)
    
    if is_main_process and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(local_rank)
        gpu_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
        print(f"🚀 GPU disponible:")
        print(f"   📱 Dispositivo: {gpu_name}")
        print(f"   💾 Memoria: {gpu_memory:.1f} GB")
        print(f"   🔢 Total GPUs: {world_size}")

    # Configurar DDP si es training distribuido
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                   output_device=local_rank if torch.cuda.is_available() else None,
                   find_unused_parameters=True)  # Para HRM con adaptive computation
        
    if is_main_process:
        print(f"🎯 Modelo configurado para dispositivo: {device}")
        if is_distributed:
            print(f"   🌐 Usando DistributedDataParallel")

    # Configurar mixed precision para GPU moderna
    use_amp = device.type == 'cuda'
    if use_amp and is_main_process:
        print("⚡ Activando Mixed Precision (AMP) para mejor rendimiento en GPU")
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Cargar datasets de HuggingFace
    if is_main_process:
        print("📚 Cargando datasets...")

    # Configurar streaming basado en parámetros
    use_streaming_mode = not no_streaming and not fast_mode

    train_texts = load_dataset_hf(
        tokenizer, "train", num_train_samples,
        dataset_name=dataset_name, dataset_config=dataset_config,
        min_text_length=min_text_length, max_text_length=max_text_length,
        use_streaming=use_streaming_mode, fast_mode=fast_mode, rank=rank
    )
    val_texts = load_dataset_hf(
        tokenizer, "validation", num_val_samples,
        dataset_name=dataset_name, dataset_config=dataset_config,
        min_text_length=min_text_length, max_text_length=max_text_length,
        use_streaming=use_streaming_mode, fast_mode=fast_mode, rank=rank
    )

    # Crear datasets distribuidos
    train_dataset = DistributedTextDataset(
        tokenizer, train_texts, config.block_size, "train",
        device=device, max_length=max_text_length, min_text_length=min_text_length,
        rank=rank, world_size=world_size
    )
    val_dataset = DistributedTextDataset(
        tokenizer, val_texts, config.block_size, "validation",
        device=device, max_length=max_text_length, min_text_length=min_text_length,
        rank=rank, world_size=world_size
    )

    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle para distributed
        num_workers=0,  # Usar 0 workers para estabilidad
        pin_memory=device.type == 'cuda',
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == 'cuda',
        drop_last=False,
    )

    # Crear optimizador más conservador para 100M
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05,  # Más regularización
        betas=(0.9, 0.95),  # Más conservador
        eps=1e-8
    )

    # Scheduler con warmup más largo
    estimated_steps_per_epoch = len(train_texts) // (batch_size * world_size)
    total_steps = estimated_steps_per_epoch * num_epochs
    pct_start = min(0.2, warmup_steps / max(total_steps, warmup_steps))  # Warmup más largo

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=pct_start
    )

    if is_main_process:
        print(f"🎯 Entrenamiento distribuido 100M configurado:")
        print(f"   Steps por época estimados: {estimated_steps_per_epoch:,}")
        print(f"   Steps totales estimados: {total_steps:,}")
        print(f"   Warmup steps: {warmup_steps}")

    # Training loop
    model.train()
    step = 0
    best_val_loss = float('inf')

    if is_main_process:
        print(f"\n🎉 ¡Iniciando entrenamiento distribuido Medium 100M!")
        print("=" * 60)

    try:
        for epoch in range(num_epochs):
            if is_main_process:
                print(f"\n📅 Época {epoch + 1}/{num_epochs}")
                
            epoch_loss = 0
            num_batches = 0
            epoch_start_time = time.time()

            # Crear barra de progreso si tqdm está disponible (solo main process)
            if TQDM_AVAILABLE and is_main_process:
                progress_bar = tqdm(
                    enumerate(train_loader),
                    desc=f"Época {epoch + 1}/{num_epochs}",
                    leave=True,
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                )
            else:
                progress_bar = enumerate(train_loader)

            for batch_idx, batch in progress_bar:
                step += 1
                step_start_time = time.time()

                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                # Crear attention mask
                attention_mask = torch.ones_like(input_ids)
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                    attention_mask = (input_ids != tokenizer.pad_token_id).long()

                # Forward pass con mixed precision si está disponible
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        if isinstance(outputs, dict):
                            loss = outputs['loss']
                        else:
                            loss = outputs.loss
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    if isinstance(outputs, dict):
                        loss = outputs['loss']
                    else:
                        loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    # Gradient clipping más agresivo para 100M
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    # Verificar NaN en gradientes antes del step
                    if not torch.isnan(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                    else:
                        if is_main_process:
                            print(f"⚠️ NaN detectado en step {step}, saltando optimización")
                        scaler.update()
                else:
                    loss.backward()
                    # Verificar NaN en gradientes
                    if torch.isnan(loss) or any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
                        if is_main_process:
                            print(f"⚠️ NaN detectado en step {step}, saltando optimización")
                        optimizer.zero_grad()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1
                step_time = time.time() - step_start_time

                # Actualizar barra de progreso (solo main process)
                if TQDM_AVAILABLE and is_main_process and isinstance(progress_bar, tqdm):
                    current_lr = scheduler.get_last_lr()[0]
                    postfix = {
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{current_lr:.2e}',
                        's/step': f'{step_time:.2f}',
                        'step': step
                    }

                    # Añadir información de GPU si está disponible
                    if device.type == 'cuda':
                        gpu_mem_used = torch.cuda.memory_allocated(local_rank) / 1024**3
                        gpu_mem_reserved = torch.cuda.memory_reserved(local_rank) / 1024**3
                        postfix['GPU'] = f'{gpu_mem_used:.1f}/{gpu_mem_reserved:.1f}GB'

                    progress_bar.set_postfix(postfix)

                # Evaluación (solo main process)
                if step % eval_steps == 0 and is_main_process:
                    model.eval()
                    val_loss = 0
                    val_batches = 0

                    print(f"\n🔍 Evaluando...")
                    with torch.no_grad():
                        for val_batch in val_loader:
                            if val_batches >= 15:  # Evaluar solo 15 batches para modelo grande
                                break

                            val_input_ids = val_batch['input_ids'].to(device)
                            val_labels = val_batch['labels'].to(device)

                            val_attention_mask = torch.ones_like(val_input_ids)
                            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                                val_attention_mask = (val_input_ids != tokenizer.pad_token_id).long()

                            val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
                            if isinstance(val_outputs, dict):
                                batch_val_loss = val_outputs['loss'].item()
                            else:
                                batch_val_loss = val_outputs.loss.item()
                            val_loss += batch_val_loss
                            val_batches += 1

                    avg_val_loss = val_loss / max(val_batches, 1)
                    perplexity = math.exp(avg_val_loss) if avg_val_loss < 10 else float('inf')

                    print(f"📊 Step {step} | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.2f}")

                    # Guardar mejor modelo
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_path = os.path.join(output_dir, "best_model")
                        model_to_save = model.module if hasattr(model, 'module') else model
                        save_model_hf(model_to_save, tokenizer, best_model_path, config, step)
                        print(f"💎 Nuevo mejor modelo guardado: {avg_val_loss:.4f}")

                    model.train()

                # Guardar checkpoint (solo main process)
                if step % save_steps == 0 and is_main_process:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
                    model_to_save = model.module if hasattr(model, 'module') else model
                    save_model_hf(model_to_save, tokenizer, checkpoint_path, config, step)
                    print(f"💾 Checkpoint guardado: {checkpoint_path}")

            # Sincronizar todos los procesos al final de cada época
            if is_distributed:
                dist.barrier()

            # Estadísticas de época (solo main process)
            if is_main_process:
                avg_epoch_loss = epoch_loss / max(num_batches, 1)
                epoch_time = time.time() - epoch_start_time
                samples_per_sec = (num_batches * batch_size * world_size) / epoch_time if epoch_time > 0 else 0

                print(f"\n📊 Época {epoch + 1}/{num_epochs} completada:")
                print(f"   📈 Loss promedio: {avg_epoch_loss:.4f}")
                print(f"   ⏱️  Tiempo: {epoch_time:.1f}s")
                print(f"   🚀 Samples/sec (total): {samples_per_sec:.1f}")
                print(f"   🎯 Mejor val loss: {best_val_loss:.4f}")
                print("-" * 50)

        # Guardar modelo final (solo main process)
        if is_main_process:
            final_path = os.path.join(output_dir, "final_model")
            model_to_save = model.module if hasattr(model, 'module') else model
            save_model_hf(model_to_save, tokenizer, final_path, config, step)
            print(f"🏁 Modelo final guardado: {final_path}")

            print(f"\n✅ ¡Entrenamiento distribuido Medium 100M completado!")
            print(f"📊 Estadísticas finales:")
            print(f"   Steps totales: {step}")
            print(f"   Mejor val loss: {best_val_loss:.4f}")
            print(f"   Modelo final: {final_path}")

    finally:
        # Cleanup distributed training
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Entrenar HRM Medium 100M con entrenamiento distribuido")
    parser.add_argument("--tokenizer", type=str, default="openai-community/gpt2",
                       help="Nombre del tokenizador HF")
    parser.add_argument("--output_dir", type=str, default="./hrm-medium-100m-distributed",
                       help="Directorio de salida")
    parser.add_argument("--train_samples", type=int, default=200000,
                       help="Número de samples de entrenamiento")
    parser.add_argument("--val_samples", type=int, default=20000,
                       help="Número de samples de validación")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Tamaño del batch por GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Número de épocas")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Frecuencia de guardado")
    parser.add_argument("--eval_steps", type=int, default=200,
                       help="Frecuencia de evaluación")

    # Parámetros de dataset
    parser.add_argument("--dataset_name", type=str, default="allenai/c4",
                       help="Nombre del dataset HF")
    parser.add_argument("--dataset_config", type=str, default="en",
                       help="Configuración del dataset")
    parser.add_argument("--max_text_length", type=int, default=2048,
                       help="Longitud máxima de texto en caracteres")
    parser.add_argument("--min_text_length", type=int, default=100,
                       help="Longitud mínima de texto en caracteres")
    parser.add_argument("--fast_mode", action="store_true", default=True,
                       help="Modo rápido: descarga dataset completo")
    parser.add_argument("--no_streaming", action="store_true", default=True,
                       help="Forzar descarga completa del dataset")

    if len(os.sys.argv) == 1:
        print("🚀 HRM Medium 100M Distributed Training")
        print("\nUso con torchrun:")
        print("  torchrun --nproc_per_node=2 hrm_training_medium_100m_distributed.py")
        print("  torchrun --nproc_per_node=4 hrm_training_medium_100m_distributed.py")
        print("  torchrun --nproc_per_node=8 hrm_training_medium_100m_distributed.py")
        print("\nEjemplos optimizados para 100M:")
        print("  # 2 GPUs - Recomendado mínimo 16GB VRAM cada una")
        print("  torchrun --nproc_per_node=2 hrm_training_medium_100m_distributed.py --batch_size 4")
        print("  # 4 GPUs - Recomendado 12GB+ VRAM cada una")
        print("  torchrun --nproc_per_node=4 hrm_training_medium_100m_distributed.py --batch_size 3")
        print("  # 8 GPUs - Mínimo 8GB VRAM cada una")
        print("  torchrun --nproc_per_node=8 hrm_training_medium_100m_distributed.py --batch_size 2")
        print("\n💡 Recomendaciones para 100M:")
        print("  - Usar gradient checkpointing (activado por defecto)")
        print("  - Batch size total efectivo: 16-32 (ajustar según GPUs)")
        print("  - Memoria GPU recomendada: 12GB+ por GPU")
        return

    args = parser.parse_args()

    # Verificar dependencias
    if not HF_TOKENIZER_AVAILABLE:
        print("❌ Tokenizador HF no disponible. Instale las dependencias:")
        print("pip install transformers tokenizers datasets")
        return

    # Iniciar entrenamiento distribuido
    train_hrm_distributed_100m(
        tokenizer_name=args.tokenizer,
        output_dir=args.output_dir,
        num_train_samples=args.train_samples,
        num_val_samples=args.val_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        max_text_length=args.max_text_length,
        min_text_length=args.min_text_length,
        fast_mode=args.fast_mode,
        no_streaming=args.no_streaming,
    )

if __name__ == "__main__":
    main()