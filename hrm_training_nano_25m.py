# -*- coding: utf-8 -*-
"""
HRM-Text1 Training Script - MODELO ULTRA-PEQUEÑO ~25M PARÁMETROS
VERSIÓN ULTRA-COMPACTA: Configuración para ~25M parámetros con contexto ultra-reducido (128 tokens)
- Arquitectura HRM ultra-eficiente (6 capas, 256 dim)
- Rotary Position Embeddings (RoPE) para mejor extrapolación
- Optimizaciones extremas de memoria para recursos muy limitados
- Configuración optimizada para entrenamiento rápido en hardware básico
"""

import os, random, contextlib, multiprocessing as mp, atexit, math
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import T5Tokenizer, PreTrainedModel, PretrainedConfig, GenerationMixin, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from datasets import load_dataset, concatenate_datasets

from huggingface_hub import HfFolder, HfApi

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
                 n_embd=512,                # Para ~100M params
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
                 h_update_period=4,          # NUEVO: H-module se actualiza cada 4 pasos
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
        self.h_update_period = getattr(config, 'h_update_period', 4)  # T steps

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
# --- CONFIGURACIÓN DEL SCRIPT PARA ~100M PARÁMETROS (MODELO PEQUEÑO) ---
# ==============================================================================

# --- CONFIGURACIÓN DE PORCENTAJES DE DATASETS ---
# Porcentaje del dataset completo a usar (1-100)
DATASET_SUBSET_PERCENT = 20.0   # Usar más datos para modelo pequeño (más eficiente)

# CONFIGURACIÓN PERSONALIZADA DE MEZCLAS
# Puedes crear tus propias combinaciones aquí o modificar las existentes
CUSTOM_MIX_RATIOS = {
    # Ejemplo de mezcla personalizada enfocada en calidad para modelo extra pequeño
    "high_quality_small": {
        "c4": 0.5,             # 50% C4 (base sólida)
        "fineweb": 0.3,        # 30% FineWeb (alta calidad)
        "openwebtext": 0.2     # 20% OpenWebText (diversidad)
    },

    # Ejemplo de mezcla balanceada para modelo extra pequeño
    "balanced_small": {
        "c4": 0.4,             # 40% C4 (multilingüe)
        "slimpajama_en": 0.3,  # 30% SlimPajama inglés
        "fineweb": 0.2,        # 20% FineWeb
        "openwebtext": 0.1     # 10% OpenWebText
    },

    # Mezcla rápida para pruebas y desarrollo
    "dev_small": {
        "c4": 0.6,             # 60% C4 (rápido de cargar)
        "openwebtext": 0.4     # 40% OpenWebText
    },

    # Mezcla enfocada en conversaciones para modelo extra pequeño
    "conversation_small": {
        "human_conversations": 0.5,  # 50% Conversaciones humanas
        "c4": 0.3,                   # 30% C4 base
        "fineweb": 0.2               # 20% Contenido de calidad
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
            "slimpajama": 0.35,
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
        "train_samples": 25_000_000,   # Estimación reducida para modelo extra pequeño (50M)
        "val_samples": 125_000,
        "repo_suffix": f"Custom-{custom_name.replace('_', '-').title()}",
        "description": f"Mezcla personalizada para 50M: {custom_name.replace('_', ' ').title()}",
        "mix_ratios": mix_ratios
    }

# Mostrar datasets disponibles
print("=== DATASETS DISPONIBLES PARA MODELO ULTRA-PEQUEÑO (25M) ===")
for key, config in DATASETS_CONFIG.items():
    marker = " ← SELECCIONADO" if key == ACTIVE_DATASET else ""
    print(f"• {key}: {config['description']}{marker}")
print("=" * 40)

# Configuración del dataset activo
DATASET_INFO = DATASETS_CONFIG[ACTIVE_DATASET]
DATASET_NAME = DATASET_INFO["name"]
DATASET_CONFIG = DATASET_INFO["config"]

HF_REPO_ID = f"dreamwar/HRM-Text1-{DATASET_INFO['repo_suffix']}-25M"
SEED = 42
NUM_EPOCHS = 2             # Menos épocas para modelo extra pequeño
BLOCK_SIZE = 128         # Contexto ultra-reducido para minimizar memoria (128 tokens)

# Configuración de entrenamiento para modelo extra pequeño (~50M parámetros)
BATCH_SIZE = 650           # Batch mínimo para reducir memoria drasticamente
GRAD_ACCUM_STEPS = 2     # Batch efectivo de 2 mantenido
EVAL_STEPS = 500         # Evaluar más frecuentemente para modelo pequeño

# Learning rate schedule optimizado para modelos grandes
LEARNING_RATE_MAX = 2e-3  # Reducido para estabilidad
LEARNING_RATE_MIN = 1e-6
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1        # 10% de warmup

# Optimizaciones
MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 3
USE_GRADIENT_CHECKPOINTING = False  # Disabled for small model - dynamic HRM computation incompatible with checkpointing

# --- CONFIGURACIÓN PARA MODELO EXTRA PEQUEÑO (~50M PARÁMETROS) ---
# Configuración ultra-compacta para recursos muy limitados
# Fórmula aproximada: params ≈ vocab_size * n_embd + n_layers * (4 * n_embd² + 3 * n_embd * d_ff)
MODEL_PARAMS = {
    "n_embd": 256,                     # Dimensión ultra-reducida (256)
    "n_head": 4,                       # 4 cabezas de atención (256/4 = 64 dim por cabeza)
    "n_layers": 6,                     # Solo 6 capas HRM (ultra-compacto)
    "d_ff": 1024,                      # 4 * n_embd para FFN (256 * 4)
    "dropout": 0.1,
    "halt_max_steps": 4,               # Mínimos pasos para modelo ultra-pequeño
    "ponder_loss_weight": 1e-2,
    "halt_bias_init": -2.2,
    "use_rotary_embeddings": True,     # RoPE para mejor extrapolación
    "use_flash_attention": True,       # Flash Attention si está disponible
    "gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
    "h_update_period": 2,              # H-module se actualiza cada 2 pasos
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

# Configurar rutas finales
OUTPUT_BASE = determine_output_base()
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "hrm_text1_c4_tiny_25m_output")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.bin")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")

print(f"📁 Ruta base configurada: {OUTPUT_BASE}")
print(f"📁 Directorio de salida: {OUTPUT_DIR}")

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
# --- FUNCIONES AUXILIARES PARA DATALOADER Y LIMPIEZA ---
# ==============================================================================

def get_num_workers():
    """
    Detecta automáticamente el número óptimo de workers para DataLoader
    """
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
                print(f"⚠️  ADVERTENCIA: Poco espacio libre ({free_gb:.1f} GB). Se recomiendan al menos 2 GB para modelo pequeño (100M)")
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

# Configuración distribuida
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

# Configurar distributed training
is_distributed, rank, world_size, local_rank = setup_distributed()

# Configurar dispositivo
if is_distributed:
    device = torch.device(f"cuda:{local_rank}")
    print(f"Dispositivo distribuido: {device} (rank {rank})")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo detectado: {device}")

# Verificar memoria disponible y mostrar información detallada de GPUs
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

print(f"Loading dataset '{DATASET_NAME}' ({DATASET_INFO['description']}) (streaming).")
print("✅ MODO STREAMING: Carga eficiente de memoria activada.")

if ACTIVE_DATASET == "mixed" or ACTIVE_DATASET in CUSTOM_MIX_RATIOS or "mix_ratios" in DATASET_INFO:
    # Cargar y mezclar múltiples datasets
    print("--- CARGANDO DATASETS PARA MEZCLA (MODELO PEQUEÑO 100M) ---")
    
    try:
        list_of_train_datasets = []
        list_of_val_datasets = []
        mix_ratios = DATASET_INFO["mix_ratios"]

        is_valid, message = validate_mix_ratios(mix_ratios, ACTIVE_DATASET)
        if not is_valid:
            print(f"❌ ERROR EN CONFIGURACIÓN: {message}")
            exit(1)
        else:
            print(f"✅ {message}")

        mix_ratios = normalize_mix_ratios(mix_ratios)
        show_mix_summary(mix_ratios, ACTIVE_DATASET)

        for dataset_key, ratio in mix_ratios.items():
            if ratio > 0:
                ds_config = DATASETS_CONFIG[dataset_key]
                print(f"Cargando {dataset_key} ({ratio*100:.1f}%): {ds_config['description']}")

                try:
                    ds = load_dataset(ds_config["name"], ds_config["config"] or None, streaming=True)
                except Exception as e:
                    print(f"  ❌ Error cargando {dataset_key}: {e}. Omitiendo.")
                    continue

                ds_lang_filter = ds_config.get("language_filter")
                if ds_lang_filter and LANGUAGE_DETECTION_AVAILABLE:
                    print(f"  Aplicando filtro de idioma {ds_lang_filter} a {dataset_key}")
                    lang_filter_func = create_language_filter_function(ds_lang_filter, relaxed="slimpajama" in dataset_key.lower())
                    ds = ds.filter(lang_filter_func, batched=True, batch_size=100)

                # Usar hash absoluto para evitar seeds negativos
                dataset_seed = SEED + abs(hash(dataset_key)) % 1000000
                
                # Sub-muestrear train (para streaming, usar take en lugar de select)
                samples_for_this_ds = int(num_train_samples * ratio)
                train_ds = ds["train"].shuffle(seed=dataset_seed).take(samples_for_this_ds)
                list_of_train_datasets.append(train_ds)

                # Sub-muestrear validation
                val_samples_for_this_ds = int(num_val_samples * ratio)
                if "validation" in ds:
                     val_ds = ds["validation"].shuffle(seed=dataset_seed).take(val_samples_for_this_ds)
                     list_of_val_datasets.append(val_ds)

        if not list_of_train_datasets:
            raise ValueError("No se pudieron cargar datasets válidos para la mezcla.")

        raw_datasets = {
            "train": concatenate_datasets(list_of_train_datasets).shuffle(seed=SEED),
            "validation": concatenate_datasets(list_of_val_datasets).shuffle(seed=SEED) if list_of_val_datasets else None
        }

        if raw_datasets["validation"] is None:
            print("Creando split de validación a partir de la mezcla de entrenamiento.")
            split_ds = raw_datasets["train"].train_test_split(test_size=num_val_samples, seed=SEED)
            raw_datasets = {"train": split_ds["train"], "validation": split_ds["test"]}

        print(f"Dataset mezclado creado con {len(list_of_train_datasets)} fuentes.")
    
    except Exception as e:
        print(f"❌ Error cargando datasets mixtos: {e}")
        print("🔄 Cambiando a dataset individual como fallback: C4")
        # Fallback a C4 cuando falle la mezcla
        raw_datasets = load_dataset("allenai/c4", "multilingual", streaming=True)
        
        # Aplicar sub-muestreo y splits para el fallback
        language_filter = None  # C4 ya es multilingüe
        if language_filter and LANGUAGE_DETECTION_AVAILABLE:
            print(f"--- APLICANDO FILTRO DE IDIOMA: {language_filter.upper()} ---")
            lang_filter_func = create_language_filter_function(language_filter)
            raw_datasets = raw_datasets.filter(lang_filter_func, batched=True, batch_size=100)
        
        # Para datasets streaming con validación separada
        if "validation" in raw_datasets and raw_datasets["validation"] is not None:
            train_ds = raw_datasets["train"].shuffle(seed=SEED).take(num_train_samples)
            val_ds = raw_datasets["validation"].shuffle(seed=SEED).take(num_val_samples)
            raw_datasets = {"train": train_ds, "validation": val_ds}
        else:
            # Sin validación separada - crear split dinámico
            train_ds = raw_datasets["train"].shuffle(seed=SEED)
            val_ds = train_ds.take(num_val_samples)
            train_ds = train_ds.skip(num_val_samples).take(num_train_samples)
            raw_datasets = {"train": train_ds, "validation": val_ds}

else:
    # Cargar dataset único
    if DATASET_INFO.get("type") == "kaggle":
        if not KAGGLE_AVAILABLE:
            print("❌ Error: Dataset de Kaggle seleccionado pero kagglehub no está disponible. Instala con: pip install kagglehub")
            exit(1)
        print(f"📥 Descargando dataset de Kaggle: {DATASET_NAME}")
        try:
            kaggle_path = kagglehub.dataset_download(DATASET_NAME)
            import glob
            data_files = glob.glob(os.path.join(kaggle_path, "*.*"))
            if not data_files: raise FileNotFoundError(f"No se encontraron archivos de datos en {kaggle_path}")
            
            file_ext = data_files[0].split('.')[-1]
            if file_ext in ['json', 'jsonl']:
                raw_datasets = load_dataset('json', data_files={'train': data_files}, streaming=True)
            elif file_ext == 'csv':
                raw_datasets = load_dataset('csv', data_files={'train': data_files}, streaming=True)
            else:
                raise ValueError(f"Formato de archivo no soportado: {file_ext}")
        except Exception as e:
            print(f"❌ Error descargando dataset de Kaggle: {e}. Cambiando a C4 como respaldo.")
            raw_datasets = load_dataset("allenai/c4", "multilingual", streaming=True)
    else:
        raw_datasets = load_dataset(DATASET_NAME, DATASET_CONFIG or None, streaming=True)

    language_filter = DATASET_INFO.get("language_filter")
    if language_filter and LANGUAGE_DETECTION_AVAILABLE:
        print(f"--- APLICANDO FILTRO DE IDIOMA: {language_filter.upper()} ---")
        lang_filter_func = create_language_filter_function(language_filter)
        raw_datasets = raw_datasets.filter(lang_filter_func, batched=True, batch_size=100)
    elif language_filter:
        print(f"⚠️  ADVERTENCIA: Filtro de idioma '{language_filter}' solicitado pero langdetect no disponible.")

# Sub-muestreo y división del dataset
language_filter_info = f" (FILTRADO: {DATASET_INFO.get('language_filter', 'N/A').upper()})" if DATASET_INFO.get("language_filter") else ""
print(f"\n!!! USANDO DATASET: {ACTIVE_DATASET.upper()} - {DATASET_INFO['description']}{language_filter_info} !!!")
print(f"!!! USANDO UN SUBCONJUNTO DEL {DATASET_SUBSET_PERCENT}% DEL DATASET !!!")

# Para datasets streaming, manejamos de forma diferente
if "validation" in raw_datasets and raw_datasets["validation"] is not None:
    # Con streaming, tomamos muestras limitadas directamente
    train_ds = raw_datasets["train"].shuffle(seed=SEED).take(num_train_samples)
    val_ds = raw_datasets["validation"].shuffle(seed=SEED).take(num_val_samples)
    raw_datasets = {"train": train_ds, "validation": val_ds}
    print(f"Dataset streaming con validación separada configurado.")
    print(f"Muestras objetivo para entrenamiento: {num_train_samples:,}")
    print(f"Muestras objetivo para validación: {num_val_samples:,}\n")
else:
    print("Dataset streaming sin validación separada - usando solo entrenamiento...")
    # Para datasets streaming sin validación, usamos solo el dataset de entrenamiento
    # y dividiremos dinámicamente durante el entrenamiento
    train_ds = raw_datasets["train"].shuffle(seed=SEED)
    
    # Para crear validación en streaming, tomamos las primeras muestras para validación
    # y el resto para entrenamiento
    val_ds = train_ds.take(num_val_samples)
    train_ds = train_ds.skip(num_val_samples).take(num_train_samples)
    
    raw_datasets = {"train": train_ds, "validation": val_ds}
    print(f"Dataset streaming dividido dinámicamente.")
    print(f"Muestras objetivo para entrenamiento: {num_train_samples:,}")
    print(f"Muestras objetivo para validación: {num_val_samples:,}\n")

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
    tokenized_splits = raw_datasets.map(
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

safe_num_workers = get_num_workers()
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

# Configurar samplers distribuidos si es necesario
train_sampler = None
val_sampler = None

if is_distributed and not is_iterable:
    train_sampler = DistributedSampler(tokenized_splits["train"], num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(tokenized_splits["validation"], num_replicas=world_size, rank=rank, shuffle=False)
    train_shuffle = False  # Cuando usamos DistributedSampler, no podemos usar shuffle en DataLoader
    print(f"🔗 DistributedSampler configurado para rank {rank}/{world_size}")

train_loader = DataLoader(
    tokenized_splits["train"],
    batch_size=BATCH_SIZE,
    num_workers=safe_num_workers,
    pin_memory=True,
    shuffle=train_shuffle,
    sampler=train_sampler,
    collate_fn=custom_collate_fn
)

val_loader = DataLoader(
    tokenized_splits["validation"],
    batch_size=BATCH_SIZE,
    num_workers=safe_num_workers,
    pin_memory=True,
    shuffle=False,
    sampler=val_sampler,
    collate_fn=custom_collate_fn
)

# Crear modelo
config = HRMText1Config(vocab_size=len(tokenizer), block_size=BLOCK_SIZE, **MODEL_PARAMS)
model = HRMText1(config).to(device)

# Envolver modelo para multi-GPU
if is_distributed:
    if world_size > 1 and 'RANK' in os.environ:
        # Entrenamiento distribuido real con torchrun
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"🔗 Modelo envuelto con DDP en GPU {local_rank}")
    elif world_size > 1:
        # Auto-inicialización multi-GPU con DataParallel
        model = nn.DataParallel(model)
        print(f"🔗 Modelo envuelto con DataParallel usando {torch.cuda.device_count()} GPUs")
        print(f"   🎯 GPU principal: {device}")
        print(f"   📋 GPUs utilizadas: {list(range(torch.cuda.device_count()))}")
    else:
        print(f"📱 Modelo en single-GPU mode: {device}")
else:
    print(f"📱 Modelo en single-GPU mode: {device}")

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
    # Para datasets regulares, usar len(train_loader)
    num_training_steps = len(train_loader) * NUM_EPOCHS

num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
print(f"Total de pasos de entrenamiento (estimado): {num_training_steps:,}")
print(f"Pasos de warmup: {num_warmup_steps:,}")

# Scheduler con warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Mixed precision scaler
scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and device.type == 'cuda'))

# --- CONFIGURACIÓN PARA MODIFICACIÓN DE LEARNING RATE ---
MODIFY_LR_ON_LOAD = False
NEW_LEARNING_RATE = 1e-4

# Checkpoint loading
start_epoch = 0
start_step = 0
best_val_loss = float('inf')
patience_counter = 0
CHECKPOINT_STEPS = 1000

if os.path.exists(CHECKPOINT_PATH):
    print(f"--- Reanudando entrenamiento desde el checkpoint: {CHECKPOINT_PATH} ---")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    # Manejar tanto DDP (_orig_mod) como DataParallel (module)
    if hasattr(model, '_orig_mod'):
        model_to_load = model._orig_mod  # DDP
    elif hasattr(model, 'module'):
        model_to_load = model.module     # DataParallel
    else:
        model_to_load = model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])

    if MODIFY_LR_ON_LOAD:
        print(f"--- Modificando learning rate a {NEW_LEARNING_RATE:.6f} ---")
        for param_group in checkpoint['optimizer_state_dict']['param_groups']:
            param_group['lr'] = NEW_LEARNING_RATE
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    start_step = checkpoint.get('step', 0)
    best_val_loss = checkpoint['best_val_loss']
    patience_counter = checkpoint.get('patience_counter', 0)
    
    print(f"Checkpoint cargado. Reanudando desde la época {start_epoch + 1}, paso {start_step}.")
else:
    print("--- No se encontró checkpoint. Empezando entrenamiento desde cero. ---")

print("torch.compile() deshabilitado para optimizar memoria")

# ==============================================================================
# --- BUCLE DE ENTRENAMIENTO ---
# ==============================================================================

global_step = start_step

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    
    # Configurar epoch para DistributedSampler si está en uso
    if is_distributed and train_sampler is not None:
        train_sampler.set_epoch(epoch)
        print(f"📅 Epoch {epoch} configurado para DistributedSampler en rank {rank}")
    
    progress = tqdm(train_loader, desc=f"Época {epoch+1}/{NUM_EPOCHS}")
    
    for i, batch in enumerate(progress):
        # Para IterableDataset, usamos la estimación de pasos por época
        loader_len = estimated_steps_per_epoch if is_iterable else len(train_loader)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = input_ids.clone()

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=MIXED_PRECISION):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
        
        if loss is not None and torch.isfinite(loss):
            scaler.scale(loss).backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0 or i + 1 == loader_len:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
                current_lr = scheduler.get_last_lr()[0]
                progress.set_postfix({"loss": f"{loss.item()*GRAD_ACCUM_STEPS:.4f}", "lr": f"{current_lr:.2e}", "step": global_step})

    # Validación al final de cada época
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
        
        # Solo rank 0 guarda checkpoints y modelos
        if not is_distributed or rank == 0:
            # Manejar tanto DDP (_orig_mod) como DataParallel (module)
            if hasattr(model, '_orig_mod'):
                model_to_save = model._orig_mod  # DDP
            elif hasattr(model, 'module'):
                model_to_save = model.module     # DataParallel
            else:
                model_to_save = model
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"Nueva mejor pérdida de validación. Guardando modelo en {BEST_MODEL_PATH}")
                torch.save(model_to_save.state_dict(), BEST_MODEL_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                
            print(f"Guardando checkpoint al final de época {epoch+1}...")
            torch.save({
                'epoch': epoch + 1, 'step': global_step, 'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(), 'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
            }, CHECKPOINT_PATH)
        
        # Sincronizar best_val_loss y patience_counter entre todos los procesos
        if is_distributed:
            # Broadcast best_val_loss desde rank 0 a todos los procesos
            best_val_loss_tensor = torch.tensor(best_val_loss, device=device)
            patience_counter_tensor = torch.tensor(patience_counter, device=device, dtype=torch.int)
            dist.broadcast(best_val_loss_tensor, src=0)
            dist.broadcast(patience_counter_tensor, src=0)
            best_val_loss = best_val_loss_tensor.item()
            patience_counter = patience_counter_tensor.item()
    
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("Detención temprana por falta de mejora en la validación.")
        break

print("Entrenamiento finalizado.")

# Solo el proceso principal (rank 0) debe guardar el modelo
if not is_distributed or rank == 0:
    # Manejar tanto DDP (_orig_mod) como DataParallel (module)
    if hasattr(model, '_orig_mod'):
        model_to_save = model._orig_mod  # DDP
    elif hasattr(model, 'module'):
        model_to_save = model.module     # DataParallel
    else:
        model_to_save = model
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Cargando el mejor modelo desde '{BEST_MODEL_PATH}' para el guardado final.")
        model_to_save.load_state_dict(torch.load(BEST_MODEL_PATH))

    # FIX: Added safe_serialization=False to handle the RuntimeError with tied weights
    model_to_save.save_pretrained(OUTPUT_DIR, safe_serialization=False)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Modelo y tokenizador guardados en '{OUTPUT_DIR}'")

    # Subir modelo a Hugging Face Hub
    if HF_TOKEN:
        try:
            print(f"\nSubiendo modelo a Hugging Face Hub: {HF_REPO_ID}")
            model_to_save.push_to_hub(HF_REPO_ID, token=HF_TOKEN, commit_message=f"Upload HRM-Text1 model", safe_serialization=False)
            tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN, commit_message=f"Upload tokenizer")
            print(f"✅ Modelo subido exitosamente a https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            print(f"❌ Error al subir el modelo a Hugging Face: {e}")
    else:
        print("\n⚠️  No se encontró HF_TOKEN. El modelo solo se guardó localmente.")
else:
    print(f"🔇 Rank {rank}: Saltando guardado del modelo (solo rank 0 guarda)")

# ==============================================================================
# --- FUNCIÓN DE CHAT Y PRUEBAS ---
# ==============================================================================

def chat_with_model(prompt_text, model, tokenizer, max_new_tokens=100, temperature=0.7, top_k=50):
    model.eval()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=MIXED_PRECISION):
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature, 
            top_k=top_k, do_sample=True, pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Solo rank 0 ejecuta las pruebas finales
if not is_distributed or rank == 0:
    print("\n--- Probando la Generación del Modelo Final ---")
    try:
        inference_model = HRMText1.from_pretrained(OUTPUT_DIR).to(device)
        prompts = [
            "The cat sat on the", "Artificial intelligence is a field that",
            "To be, or not to be, that is the question:", "In a world where technology advances rapidly,",
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
    print(f"Mejor pérdida de validación: {best_val_loss:.4f}")
    print(f"Modelo guardado en: {OUTPUT_DIR}")
else:
    print(f"🔇 Rank {rank}: Saltando pruebas de generación (solo rank 0 ejecuta)")

# Limpiar procesos distribuidos
if is_distributed:
    print("🧹 Limpiando procesos distribuidos...")
    dist.destroy_process_group()

print("\n--- Script completado exitosamente ---")