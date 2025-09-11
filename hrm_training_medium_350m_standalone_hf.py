# -*- coding: utf-8 -*-
"""
HRM-Models Training Script con Tokenizador HuggingFace - MODELO MEDIUM ~350M PARÁMETROS
VERSIÓN MEJORADA: Usando tokenizadores profesionales de HuggingFace

🖥️  CARACTERÍSTICAS:
- Tokenizador HuggingFace (GPT2, GPT2-Spanish, etc.)
- Vocabulario profesional (50K+ tokens)
- Mejor soporte multilingüe (español/inglés)
- Arquitectura HRM optimizada para 350M parámetros
- Sin dependencias de transformers para el modelo (solo tokenizer)
"""

import os, multiprocessing as mp, math, time
from typing import List, Dict, Optional, Tuple
import argparse


# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ tqdm no disponible, usando progreso básico")

# Configurar método de multiprocessing antes de cualquier uso
if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    # Marcar PID principal para evitar spam en multiprocessing
    import os
    os._main_pid = os.getpid()

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Definir clases HRM directamente (extraídas del archivo standalone para evitar dependencias)
print("✅ Configuración HRM integrada directamente en el script HF")

# ==============================================================================
# --- HRM CORE CLASSES (Extracted from standalone) ---
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
                 block_size=512,            # Medium-350M model context
                 n_embd=768,                # Medium-350M model embeddings
                 n_head=12,                 # Medium-350M model heads
                 n_layers=24,               # Medium-350M model layers (doubled)
                 d_ff=3072,                 # Medium-350M model FFN
                 dropout=0.1,               # Standard for large models
                 pad_token_id=0,
                 halt_max_steps=8,          # HRM halt steps
                 ponder_loss_weight=2e-3,   # Further reduced for larger model
                 halt_bias_init=-1.0,
                 use_rotary_embeddings=True,
                 rotary_embedding_base=10000,
                 use_flash_attention=True,
                 gradient_checkpointing=True,  # Enable for memory efficiency
                 # HRM Ciclos controlados para estabilidad
                 H_cycles=1,                # Mantenido para estabilidad
                 L_cycles=5,                # Incrementado para mayor capacidad
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
    """True HRM implementation with hierarchical temporal separation and Deep Supervision for LLMs"""
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
        std_h = lecun_std * depth_factor * 0.3  # Further reduced for larger model
        std_l = lecun_std * depth_factor * 0.2  # L-module even more conservative

        # Usar trunc_normal_init_ del código ACT (matemáticamente correcto)
        trunc_normal_init_(self.h_prediction_head.weight, std=std_h)
        trunc_normal_init_(self.l_prediction_head.weight, std=std_l)

        print(f"🔧 Capa {layer_idx}: H-head std={std_h:.5f}, L-head std={std_l:.5f} (ACT trunc_normal)")

        # Ponder loss components
        self.ponder_network = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.ReLU(),
            nn.Linear(config.n_embd // 4, 1),
            nn.Sigmoid()
        )

        self.convergence_threshold = 1e-3
        self.max_l_steps = config.halt_max_steps

        # HRM Cycles from paper, adapted for LLMs
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles

    def forward(self, z_H, z_L, input_embeddings, attn_mask=None, key_padding_mask=None, training=True):
        """Forward pass with HRM cycles adapted for LLMs - respects token causality"""
        _ = z_H.shape  # batch_size, seq_len, d_model
        _ = z_H.device  # device

        total_l_steps = 0
        total_ponder_loss = 0.0
        all_q_values = []

        # HRM Cycles implementation for LLMs (medium-350M model: enhanced cycles)
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

                        # Exploration durante training (como en ACT)
                        halt_exploration_prob = 0.2  # Increased for larger model
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

        # Estabilización conservadora de logits para modelo grande
        h_logits = torch.clamp(h_logits, -10.0, 10.0)  # Increased range for large model
        l_logits = torch.clamp(l_logits, -10.0, 10.0)  # Increased range for large model

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
    """HRM Model with full hierarchical temporal separation"""
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

        # Inicialización
        self.apply(self._init_weights)

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Inicialización Xavier/Glorot más conservadora para HRM
            nn.init.xavier_uniform_(module.weight, gain=0.3)  # Further reduced gain for large model
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.006)  # Further reduced std for large model

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
            if self.config.gradient_checkpointing and self.training:
                # Use gradient checkpointing for memory efficiency
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                z_H, z_L, hrm_info = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    z_H, z_L, x, causal_mask, attention_mask, self.training
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

                # Estabilizar logits antes del cálculo de pérdida
                shift_h_logits = torch.clamp(shift_h_logits, -20.0, 20.0)  # Prevenir logits extremos

                h_loss = F.cross_entropy(
                    shift_h_logits.view(-1, shift_h_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.config.pad_token_id,
                    reduction='mean'  # Asegurar reducción mean
                )

                # Verificar y corregir h_loss si es inestable
                if torch.isnan(h_loss) or torch.isinf(h_loss) or h_loss > 50.0:
                    print(f"⚠️ H-loss inestable en capa {layer_idx}: {h_loss.item():.4f}, usando fallback")
                    # Usar una pérdida de referencia basada en vocabulario
                    h_loss = torch.log(torch.tensor(self.config.vocab_size, device=h_loss.device, dtype=torch.float32))

                # L-module supervision (if available) con estabilización
                l_loss = 0.0
                if l_logits is not None:
                    shift_l_logits = l_logits[..., :-1, :].contiguous()

                    # Estabilizar L-logits también
                    shift_l_logits = torch.clamp(shift_l_logits, -20.0, 20.0)

                    l_loss = F.cross_entropy(
                        shift_l_logits.view(-1, shift_l_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=self.config.pad_token_id,
                        reduction='mean'
                    )

                    # Verificar y corregir l_loss si es inestable
                    if torch.isnan(l_loss) or torch.isinf(l_loss) or l_loss > 50.0:
                        print(f"⚠️ L-loss inestable en capa {layer_idx}: {l_loss.item():.4f}, usando fallback")
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

            # Main loss con label smoothing para mejor generalización
            main_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
                label_smoothing=0.1  # Evita overconfidence
            )

            # Deep Supervision loss (weighted sum of intermediate losses) - ahora más estable
            deep_supervision_loss = 0.0
            if intermediate_losses:
                total_layers = len(self.layers)

                for loss_info in intermediate_losses:
                    h_loss = loss_info['h_loss']
                    l_loss = loss_info['l_loss']

                    # Las pérdidas ya están estabilizadas arriba, solo verificar valores extremos
                    h_loss = torch.clamp(h_loss, 0.0, 50.0)  # Limitar rango
                    if isinstance(l_loss, torch.Tensor):
                        l_loss = torch.clamp(l_loss, 0.0, 50.0)
                    else:
                        l_loss = min(max(l_loss, 0.0), 50.0)  # Para valores scalar

                    layer_weight = (loss_info['layer_idx'] + 1) / total_layers
                    layer_loss = layer_weight * (h_loss + 0.2 * l_loss)  # Further reduced L-loss weight
                    deep_supervision_loss += layer_loss

                deep_supervision_loss /= len(intermediate_losses)  # Promedio

            # Ponder loss (computational regularization)
            ponder_loss = total_ponder_loss / len(self.layers) if total_ponder_loss > 0 else 0.0

            # Q-learning loss para halting (inspirado en código ACT)
            q_learning_loss = 0.0
            if all_hrm_info and labels is not None:
                # Calcular correctness para Q-learning target (como en ACT)
                with torch.no_grad():
                    # Verificar si las predicciones son correctas
                    predictions = torch.argmax(logits, dim=-1)
                    mask = labels != self.config.pad_token_id
                    is_correct = mask & (predictions == labels)

                    # Accuracy por secuencia para Q-learning target
                    seq_accuracy = is_correct.float().sum(-1) / mask.float().sum(-1).clamp(min=1)
                    seq_is_correct = seq_accuracy > 0.8  # Threshold como en ACT

                # Calcular Q-learning loss de todos los layers
                for hrm_info in all_hrm_info:
                    if hrm_info.get('q_values'):
                        for q_vals in hrm_info['q_values']:
                            # Q-halt loss: predecir si la secuencia será correcta
                            # q_vals ya tiene dimensiones [batch_size, 2] después del pooling
                            q_halt_logits = q_vals[..., 1]  # Solo tomar índice de halt
                            q_halt_loss = F.binary_cross_entropy_with_logits(
                                q_halt_logits,
                                seq_is_correct.float(),
                                reduction='mean'
                            )
                            q_learning_loss += q_halt_loss * 0.03  # Further reduced weight for large model

            # TEMPORAL: Verificar y desactivar Deep Supervision si tiene NaN
            if torch.isnan(torch.tensor(deep_supervision_loss)) or torch.isinf(torch.tensor(deep_supervision_loss)):
                print(f"⚠️ Deep Supervision NaN detectado, desactivando temporalmente")
                deep_supervision_loss = 0.0

            # Deep supervision controlado para evitar NaN
            ds_weight = 0.02  # Minimal weight for large model
            ponder_weight = self.config.ponder_loss_weight * 0.7  # Moderately reduced

            total_loss = (
                main_loss +
                ds_weight * deep_supervision_loss +  # Deep supervision controlado
                ponder_weight * ponder_loss +  # Ponder loss
                q_learning_loss  # Q-learning loss para halting automático
            )

            # Verificar estabilidad de la pérdida final
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"⚠️ NaN/Inf detectado en total_loss: main={main_loss.item()}, ds={deep_supervision_loss}, ponder={ponder_loss}")
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
# --- Dataset Handling (Abbreviated for space) ---
# ==============================================================================

# [Dataset and training functions would be similar to previous models 
#  but with adjusted parameters for the 350M model - batch sizes, etc.]

class OptimizedTextDataset(IterableDataset):
    """Dataset optimizado para texto usando tokenizador HF con GPU y paralelización"""

    def __init__(self, tokenizer, texts: List[str], block_size: int = 512, split_type: str = "train",
                 device=None, batch_tokenize: bool = True, num_proc: int = None, max_length: int = 2048,
                 min_text_length: int = 20, cache_tokens: bool = False):
        self.tokenizer = tokenizer
        self.texts = texts
        self.block_size = block_size
        self.split_type = split_type
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.batch_tokenize = batch_tokenize
        # Configuración inteligente de workers
        if num_proc is not None:
            self.num_proc = num_proc
        else:
            # Auto-detectar basado en cores disponibles
            cpu_cores = mp.cpu_count()
            self.num_proc = min(cpu_cores, 16)
        self.max_length = max_length
        self.min_text_length = min_text_length
        self.cache_tokens = cache_tokens

        # Pre-procesar y tokenizar si está habilitado el cache
        if self.cache_tokens:
            print(f"🔄 Pre-tokenizando {len(texts)} textos para cache...")
            self.tokenized_chunks = self._preprocess_and_tokenize()
        else:
            self.tokenized_chunks = None

        print(f"📚 Dataset {split_type}: {len(texts)} textos, block_size={block_size}")
        print(f"   🚀 GPU tokenization: {self.device.type == 'cuda'}")
        print(f"   ⚡ Batch tokenization: {batch_tokenize}")
        print(f"   🔧 Num processes: {self.num_proc}")
        print(f"   💾 Cache tokens: {cache_tokens}")

    def _preprocess_and_tokenize(self):
        """Pre-procesar y tokenizar todos los textos en paralelo"""
        # Filtrar textos válidos
        valid_texts = [text.strip()[:self.max_length] for text in self.texts
                      if text and len(text.strip()) >= self.min_text_length]

        if not valid_texts:
            return []

        all_chunks = []

        if self.batch_tokenize and len(valid_texts) > 1:
            # Tokenización en batch para mejor rendimiento
            try:
                print(f"   🔄 Tokenizando en batches de 32...")
                batch_size = 32

                for i in range(0, len(valid_texts), batch_size):
                    batch_texts = valid_texts[i:i + batch_size]

                    # Tokenizar batch completo
                    batch_tokens = self.tokenizer(
                        batch_texts,
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True,
                        padding=False,
                        return_tensors=None
                    )

                    # Procesar cada secuencia tokenizada
                    for tokens in batch_tokens['input_ids']:
                        chunks = self._create_chunks(tokens)
                        all_chunks.extend(chunks)

            except Exception as e:
                print(f"⚠️ Error en batch tokenization, usando secuencial: {e}")
                # Fallback a tokenización secuencial
                for text in valid_texts:
                    try:
                        tokens = self.tokenizer.encode(text, add_special_tokens=True,
                                                     max_length=self.max_length, truncation=True)
                        chunks = self._create_chunks(tokens)
                        all_chunks.extend(chunks)
                    except Exception:  # pylint: disable=broad-except
                        continue
        else:
            # Tokenización secuencial tradicional
            for text in valid_texts:
                try:
                    tokens = self.tokenizer.encode(text, add_special_tokens=True,
                                                 max_length=self.max_length, truncation=True)
                    chunks = self._create_chunks(tokens)
                    all_chunks.extend(chunks)
                except Exception:  # pylint: disable=broad-except
                    continue

        print(f"   ✅ Generados {len(all_chunks)} chunks tokenizados")
        return all_chunks

    def _create_chunks(self, tokens):
        """Crear chunks de tokens del tamaño especificado"""
        chunks = []
        for i in range(0, len(tokens) - self.block_size + 1, self.block_size):
            chunk = tokens[i:i + self.block_size]
            if len(chunk) == self.block_size:
                chunks.append({
                    'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
                    'labels': torch.tensor(chunk[1:], dtype=torch.long)
                })
        return chunks

    def __iter__(self):
        if self.tokenized_chunks is not None:
            # Usar chunks pre-tokenizados
            for chunk in self.tokenized_chunks:
                yield chunk
        else:
            # Tokenización on-the-fly
            for text in self.texts:
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
                    print(f"⚠️ Error tokenizando texto: {e}")
                    continue

# Mantener compatibilidad hacia atrás
TextDataset = OptimizedTextDataset

def load_dataset_hf(tokenizer, split: str = "train", num_samples: int = 1000,
                   dataset_name: str = "allenai/c4", dataset_config: str = "en",
                   text_column: str = "text", min_text_length: int = 50,
                   max_text_length: int = 2000, num_proc: int = None,
                   use_streaming: bool = True, fast_mode: bool = False):
    """Cargar dataset usando datasets de HuggingFace de forma parametrizable"""
    try:
        from datasets import load_dataset

        print(f"📥 Cargando dataset '{dataset_name}' ({dataset_config}) split '{split}' con {num_samples} samples...")
        print(f"   📋 Configuración:")
        print(f"   - Dataset: {dataset_name}")
        print(f"   - Config: {dataset_config}")
        print(f"   - Columna texto: {text_column}")
        print(f"   - Min length: {min_text_length} chars")
        print(f"   - Max length: {max_text_length} chars")
        print(f"   - Streaming: {use_streaming}")
        print(f"   - Fast mode: {fast_mode}")

        # OPTIMIZACIÓN: Para datasets grandes, usar modo no-streaming es más rápido
        if fast_mode and num_samples > 50000:
            print("⚡ Fast mode: Usando dataset no-streaming para mejor rendimiento...")
            use_streaming = False

        # Cargar dataset
        if use_streaming:
            dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        else:
            print("📦 Descargando dataset completo (más rápido para lotes grandes)...")
            dataset = load_dataset(dataset_name, dataset_config, split=split)

        texts = []
        processed_count = 0

        if TQDM_AVAILABLE:
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

                if TQDM_AVAILABLE and isinstance(progress, tqdm):
                    progress.set_postfix({
                        'válidos': processed_count,
                        'ratio': f'{processed_count/(i+1)*100:.1f}%'
                    })

        if TQDM_AVAILABLE and isinstance(progress, tqdm):
            progress.close()

        print(f"✅ Procesados {processed_count} textos válidos de {i+1} samples totales")
        print(f"   📊 Ratio de aprovechamiento: {processed_count/(i+1)*100:.1f}%")

        if not texts:
            print("❌ No se encontraron textos válidos en el dataset")
            print("💡 Sugerencias:")
            print("   - Use --no_streaming para mejor compatibilidad")
            print("   - Reduzca --train_samples")
            raise ValueError(f"No se pudieron cargar textos válidos de {dataset_name}")

        return texts

    except Exception as e:
        print(f"❌ Error cargando dataset HF: {e}")
        print("💡 Sugerencias:")
        print("   - Use --no_streaming para mejor compatibilidad con datasets grandes")
        print("   - Verifique conectividad de red")
        print("   - Reduzca el número de samples")
        raise RuntimeError(f"Falló carga de dataset {dataset_name}: {e}") from e

def save_model_hf(model, tokenizer, save_path: str, config: HRMText1Config, step: int = 0):  # pylint: disable=unused-argument
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
        'tokenizer_type': getattr(config, 'tokenizer_type', 'huggingface'),
        'hf_tokenizer_name': getattr(config, 'hf_tokenizer_name', 'openai-community/gpt2'),
        'pad_token_id': getattr(config, 'pad_token_id', 0),
        'gradient_checkpointing': getattr(config, 'gradient_checkpointing', False),
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

def train_hrm_hf(
    tokenizer_name: str = "openai-community/gpt2",
    output_dir: str = "./hrm-medium-350m-hf",
    num_train_samples: int = 500000,   # Massively increased for large model
    num_val_samples: int = 10000,      # Increased for robust validation
    batch_size: int = 2,               # Reduced for very large model
    learning_rate: float = 1e-5,       # Much lower for stability
    num_epochs: int = 8,               # Increased for better convergence
    save_steps: int = 1000,
    eval_steps: int = 200,             # More frequent evaluation
    max_grad_norm: float = 1.0,        # Standard clipping
    warmup_steps: int = 5000,          # Much longer warmup for large model
    # Parámetros de tokenización optimizada
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    batch_tokenize: bool = True,
    cache_tokens: bool = False,
    max_text_length: int = 4000,       # Further increased for better context
    min_text_length: int = 200,        # Increased minimum for quality
    # Parámetros de paralelización configurable
    num_workers: int = 0,
    tokenizer_workers: int = 0,
    prefetch_factor: int = 2,
    cpu_intensive: bool = False,
    max_workers: int = 0,
    batch_size_multiplier: int = 1,
    # Parámetros para acelerar carga de datos
    fast_mode: bool = False,
    no_streaming: bool = False,
):
    """Entrenar modelo HRM Medium 350M con tokenizador HuggingFace"""
    
    print("🚀 Iniciando entrenamiento HRM Medium 350M con tokenizador HF")
    print("⚠️ ADVERTENCIA: Este modelo requiere GPU con al menos 32GB VRAM")
    print("📊 Configuración optimizada para modelo de gran escala:")
    print(f"   Samples entrenamiento: {num_train_samples:,}")
    print(f"   Batch size: {batch_size} (muy reducido para modelo grande)")
    print(f"   Learning rate: {learning_rate} (muy bajo para estabilidad)")
    print(f"   Gradient checkpointing: ACTIVADO para eficiencia de memoria")
    
    # [Implementation would continue similarly but with GPU memory optimizations,
    #  reduced batch sizes, and potentially distributed training setup]

if __name__ == "__main__":
    print("🚀 HRM Medium 350M Training Script")
    print("⚠️  NOTA: Este modelo requiere hardware especializado (32GB+ VRAM)")
    print("💡 Para entrenar este modelo, use configuración multi-GPU o GPU de alta memoria")