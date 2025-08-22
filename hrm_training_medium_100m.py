# -*- coding: utf-8 -*-
"""
HRM-Models Training Script - MODELO MEDIUM ~100M PARÁMETROS  
VERSIÓN MEDIUM: Configuración para ~100M parámetros con contexto extendido (1024 tokens)
- Arquitectura HRM medium-eficiente (12 capas, 512 dim)
- Rotary Position Embeddings (RoPE) para mejor extrapolación
- Optimizaciones extremas de memoria para recursos muy limitados
- Configuración optimizada para hardware avanzado (A100, H100, etc.)
"""

import os, random, multiprocessing as mp, atexit, math, time
from typing import List, Dict, Optional, Tuple

# Configurar método de multiprocessing antes de cualquier uso
if __name__ == '__main__':
    mp.set_start_method('fork', force=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import T5Tokenizer, PreTrainedModel, PretrainedConfig, GenerationMixin, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from datasets import load_dataset

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
    _tied_weights_keys = ["lm_head.weight", "token_embeddings.weight"]
    
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
    
    def _tie_weights(self):
        """Tie the weights between the input and output embeddings"""
        if hasattr(self, 'lm_head') and hasattr(self, 'token_embeddings'):
            self._tie_or_clone_weights(self.lm_head, self.token_embeddings)
    
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
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        """
        Carga modelo desde directorio que contiene config.json y checkpoint.pth o pytorch_model.bin
        Compatible con modelos guardados tanto con save_pretrained() como con checkpoint manual
        """
        import json
        
        # Cargar configuración
        config_path = os.path.join(pretrained_model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = HRMText1Config(**config_dict)
        else:
            raise FileNotFoundError(f"No se encontró config.json en {pretrained_model_path}")
        
        # Crear modelo
        model = cls(config)
        
        # Intentar cargar pesos - prioridad: pytorch_model.bin > checkpoint.pth
        model_files = [
            "pytorch_model.bin",
            "model.safetensors", 
            "checkpoint.pth"
        ]
        
        loaded = False
        for model_file in model_files:
            model_path = os.path.join(pretrained_model_path, model_file)
            if os.path.exists(model_path):
                try:
                    state_dict = torch.load(model_path, map_location='cpu')
                    
                    # Si es un checkpoint completo, extraer model_state_dict
                    if 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']
                    
                    # Cargar pesos
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    
                    if missing_keys:
                        print(f"⚠️ Claves faltantes: {missing_keys}")
                    if unexpected_keys:
                        print(f"⚠️ Claves inesperadas: {unexpected_keys}")
                    
                    print(f"✅ Modelo cargado desde: {model_path}")
                    loaded = True
                    break
                    
                except Exception as e:
                    print(f"⚠️ Error cargando {model_file}: {e}")
                    continue
        
        if not loaded:
            raise FileNotFoundError(f"No se encontraron pesos válidos en {pretrained_model_path}")
        
        return model

# ==============================================================================
# --- CONFIGURACIÓN DEL SCRIPT PARA ~100M PARÁMETROS (MODELO PEQUEÑO) ---
# ==============================================================================

# --- CONFIGURACIÓN DE PORCENTAJES DE DATASETS ---
# Porcentaje del dataset completo a usar (1-100)
DATASET_SUBSET_PERCENT = 100   # Usar más datos para modelo pequeño (más eficiente)

# CONFIGURACIÓN PERSONALIZADA DE MEZCLAS
# Puedes crear tus propias combinaciones aquí o modificar las existentes
CUSTOM_MIX_RATIOS = {
    # Ejemplo de mezcla personalizada enfocada en calidad para modelo micro
    "high_quality_small": {
        "c4": 0.5,             # 50% C4 (base sólida)
        "fineweb": 0.3,        # 30% FineWeb (alta calidad)
        "openwebtext": 0.2     # 20% OpenWebText (diversidad)
    },
    
    # Ejemplo de mezcla balanceada para modelo micro
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
    
    # Mezcla enfocada en conversaciones para modelo micro
    "conversation_small": {
        "human_conversations": 0.5,  # 50% Conversaciones humanas
        "c4": 0.3,                   # 30% C4 base
        "fineweb": 0.2               # 20% Contenido de calidad
    }
}

# --- CONFIGURACIÓN DE DATASETS MÚLTIPLES ---
# Selecciona el dataset a usar cambiando ACTIVE_DATASET
ACTIVE_DATASET = "c4-english"  # Opciones: "c4", "openwebtext", "pile", "spanish", "mixed", "high_quality_1b", etc.

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
    "c4-english": {
        "name": "allenai/c4",
        "config": "en",
        "train_samples": 365_000_000,
        "val_samples": None,  # Se usará split automático
        "repo_suffix": "English",
        "description": "Texto en inglés del dataset C4"
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
    "mixed": {
        "name": "mixed",  # Identificador especial
        "config": None,
        "train_samples": 500_000_000,  # Estimación combinada
        "val_samples": 200_000,
        "repo_suffix": "Mixed",
        "description": "Combinación de múltiples datasets",
        "mix_ratios": {  # Proporción de cada dataset en la mezcla
            "c4-english": 0.30,
            "fineweb": 0.20,
            "slimpajama": 0.30,
            "spanish": 0.20
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
        "train_samples": 25_000_000,   # Estimación expandida para modelo micro H200 (25M)
        "val_samples": 125_000,
        "repo_suffix": f"Custom-{custom_name.replace('_', '-').title()}",
        "description": f"Mezcla personalizada para 50M: {custom_name.replace('_', ' ').title()}",
        "mix_ratios": mix_ratios
    }

# Mostrar datasets disponibles
print("=== DATASETS DISPONIBLES PARA MODELO MICRO OPTIMIZADO H200 (25M) ===")
for key, config in DATASETS_CONFIG.items():
    marker = " ← SELECCIONADO" if key == ACTIVE_DATASET else ""
    print(f"• {key}: {config['description']}{marker}")
print("=" * 40)

# Configuración del dataset activo
DATASET_INFO = DATASETS_CONFIG[ACTIVE_DATASET]
DATASET_NAME = DATASET_INFO["name"]
DATASET_CONFIG = DATASET_INFO["config"]

HF_REPO_ID = f"dreamwar/HRM-Models-Medium-100M"
SEED = 42
NUM_EPOCHS = 3             # Épocas para entrenamiento 100M
CONTINUE_TRAINING = True    # True: añade épocas extra y modifica LR automáticamente
BLOCK_SIZE = 1024        # Contexto extendido para mejor calidad de modelo (1024 tokens)

# Configuración de entrenamiento para modelo 100M optimizada para A100/H100
BATCH_SIZE = 16        # Batch balanceado para modelo 100M (~12GB uso estimado)
GRAD_ACCUM_STEPS = 2     # Batch efectivo de 8192 para entrenamiento súper eficiente
EVAL_STEPS = 500         # Evaluar más frecuentemente para modelo pequeño

# Learning rate schedule optimizado para datasets grandes con decaimiento suave
LEARNING_RATE_MAX = 8e-4  # Reducido significativamente para datasets grandes
LEARNING_RATE_MIN = 2e-6  # Mínimo más alto para evitar estancamiento
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.15       # 15% de warmup más largo para estabilidad inicial

# Optimizaciones
MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 3
USE_GRADIENT_CHECKPOINTING = False  # Disabled for small model - dynamic HRM computation incompatible with checkpointing

# --- CONFIGURACIÓN PARA MODELO MICRO OPTIMIZADO PARA H200 (~25M PARÁMETROS) ---
# Configuración micro expandida para aprovechar mejor hardware potente (H200)
# Fórmula aproximada: params ≈ vocab_size * n_embd + n_layers * (4 * n_embd² + 3 * n_embd * d_ff)
MODEL_PARAMS = {
    "n_embd": 512,                     # Dimensión para ~100M params (512)
    "n_head": 16,                      # 16 cabezas de atención (512/16 = 32 dim por cabeza)
    "n_layers": 12,                    # 12 capas HRM (capacidad 100M)
    "d_ff": 2048,                      # 4 * n_embd para FFN (512 * 4)
    "dropout": 0.1,
    "halt_max_steps": 8,               # Pasos optimizados para modelo 100M
    "ponder_loss_weight": 1e-2,
    "halt_bias_init": -2.2,
    "use_rotary_embeddings": True,     # RoPE para mejor extrapolación
    "use_flash_attention": True,       # Flash Attention si está disponible
    "gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
    "h_update_period": 4,              # H-module se actualiza cada 4 pasos para 100M 
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

def detect_and_setup_colab():
    """Detecta si estamos en Google Colab y configura Google Drive automáticamente"""
    try:
        # Verificar si estamos en Colab
        import google.colab
        print("🔍 Google Colab detectado!")
        
        # Montar Google Drive automáticamente
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✅ Google Drive montado exitosamente en /content/drive")
            
            # Verificar que el directorio existe
            drive_path = "/content/drive/MyDrive"
            if os.path.exists(drive_path):
                print(f"✅ Directorio de Drive confirmado: {drive_path}")
                return drive_path
            else:
                print(f"⚠️  Directorio de Drive no encontrado, usando directorio local")
                return "./HRM_Models"
                
        except Exception as e:
            print(f"⚠️  Error montando Google Drive: {e}")
            print("🔄 Continuando con directorio local")
            return "./HRM_Models"
            
    except ImportError:
        # No estamos en Colab
        print("📱 Entorno local detectado (no es Google Colab)")
        return None

# Determinar ruta base final
def determine_output_base():
    """Determina la ruta base según la configuración"""
    # Prioridad: Variable de entorno > Ruta personalizada > Colab Drive > Ruta por defecto
    if HRM_OUTPUT_BASE_ENV:
        print(f"🌍 Usando ruta desde variable de entorno: {HRM_OUTPUT_BASE_ENV}")
        return HRM_OUTPUT_BASE_ENV
    elif CUSTOM_BASE_PATH:
        print(f"🎯 Usando ruta personalizada: {CUSTOM_BASE_PATH}")
        return CUSTOM_BASE_PATH
    else:
        # Detectar y configurar Google Colab automáticamente
        colab_path = detect_and_setup_colab()
        if colab_path:
            return os.path.join(colab_path, "HRM")
        
        # Rutas por defecto según el entorno
        if os.path.exists(os.path.expanduser("~/Documents")):
            return os.path.expanduser("~/Documents/HRM")  # Sistemas Unix/Mac
        else:
            return "./HRM_Models"  # Directorio actual como fallback

# Configurar rutas finales
OUTPUT_BASE = determine_output_base()
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "hrm_models_medium_100m_output")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.bin")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")

print(f"📁 Ruta base configurada: {OUTPUT_BASE}")
print(f"📁 Directorio de salida: {OUTPUT_DIR}")
print(f"📊 TensorBoard logs: {os.path.join(OUTPUT_DIR, 'tensorboard_logs')}")
print(f"💡 Para ver TensorBoard: tensorboard --logdir {os.path.join(OUTPUT_DIR, 'tensorboard_logs')}")
print()

# Verificar disponibilidad de librerías y mostrar status
libraries_status = []
libraries_status.append(f"✅ TensorBoard: {TENSORBOARD_AVAILABLE}")
libraries_status.append(f"✅ Kagglehub: {KAGGLE_AVAILABLE}")
libraries_status.append(f"✅ LangDetect: {LANGUAGE_DETECTION_AVAILABLE}")

print("🔧 Status de librerías opcionales:")
for status in libraries_status:
    print(f"   {status}")
print()

# Configurar TensorBoard
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, "tensorboard_logs")
if TENSORBOARD_AVAILABLE:
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    print(f"📊 TensorBoard logs: {TENSORBOARD_DIR}")
    print(f"💡 Para ver TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}")

# ==============================================================================
# --- FUNCIONES AUXILIARES PARA DATALOADER ---
# ==============================================================================

def get_system_info():
    """Obtiene información precisa del sistema"""
    try:
        import psutil
        logical_cpus = psutil.cpu_count(logical=True)
        physical_cpus = psutil.cpu_count(logical=False)
        cpu_info = f"CPUs: {physical_cpus} físicos, {logical_cpus} lógicos"
        print(f"🖥️  Sistema detectado - {cpu_info}")
        return logical_cpus, physical_cpus
    except ImportError:
        logical_cpus = mp.cpu_count()
        physical_cpus = logical_cpus // 2
        print(f"🖥️  Sistema detectado - CPUs: ~{physical_cpus} físicos, {logical_cpus} lógicos (estimado)")
        return logical_cpus, physical_cpus

def get_dataloader_workers():
    """Determina workers conservativos para DataLoader considerando hf_transfer"""
    logical_cpus, physical_cpus = get_system_info()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    hf_transfer_enabled = os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0') == '1'
    
    # Para entrenamiento distribuido
    if is_distributed and world_size > 1:
        workers_per_process = 2 if hf_transfer_enabled else 3
        optimal_workers = min(workers_per_process, physical_cpus // world_size)
        print(f"🚀 Modo distribuido: {optimal_workers} workers por proceso")
        return optimal_workers
    
    # Detección de entornos especiales
    try:
        if 'google.colab' in str(get_ipython()):
            print("🔧 Google Colab: usando 2 workers conservativos")
            return 2
    except:
        pass

    try:
        get_ipython()
        print("🔧 Jupyter/IPython: usando 2 workers conservativos")
        return 2
    except:
        pass

    # Cálculo conservativo basado en hf_transfer
    if hf_transfer_enabled:
        # Con hf_transfer, usar menos workers para evitar rate limits
        if num_gpus > 1:
            optimal_workers = min(2, max(1, physical_cpus // 4))
            print(f"🚀 Multi-GPU + hf_transfer: {optimal_workers} workers conservativos")
        else:
            optimal_workers = min(2, max(1, physical_cpus // 2))
            print(f"🔧 Single-GPU + hf_transfer: {optimal_workers} workers conservativos")
    else:
        # Sin hf_transfer, algo más de workers pero conservativo
        if num_gpus > 1:
            optimal_workers = min(4, max(2, physical_cpus // 2))
            print(f"🚀 Multi-GPU sin hf_transfer: {optimal_workers} workers moderados")
        else:
            optimal_workers = min(3, max(2, physical_cpus // 2))
            print(f"🔧 Single-GPU sin hf_transfer: {optimal_workers} workers moderados")
    
    return optimal_workers

def get_tokenization_workers():
    """Workers específicos para tokenización (independiente de DataLoader)"""
    logical_cpus, physical_cpus = get_system_info()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Tokenización necesita más workers que DataLoader
    if num_gpus > 1:
        max_workers = min(6, physical_cpus)
    else:
        max_workers = min(4, physical_cpus)
    
    gpu_info = f"{num_gpus} GPU{'s' if num_gpus > 1 else ''}"
    print(f"Tokenización sistema ({gpu_info}): {max_workers} workers")
    
    return max_workers

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

def get_optimized_buffer_size(num_gpus, dataset_size_hint='medium'):
    """Calcula buffer size óptimo conservativo para evitar saturación de memoria"""
    # Buffer base más conservativo
    if dataset_size_hint == 'large':  # Streaming datasets masivos
        base_buffer = 32  # Reducido de 256 a 32
    elif dataset_size_hint == 'medium':  # Datasets normales
        base_buffer = 16  # Reducido de 128 a 16
    else:  # small datasets
        base_buffer = 8   # Reducido de 64 a 8
    
    # Escalado conservativo por número de GPUs
    gpu_multiplier = min(max(1, num_gpus), 4)  # Máximo 4x
    buffer_size = base_buffer * gpu_multiplier
    
    # Límites muy conservativos para evitar saturar memoria
    return min(max(buffer_size, 8), 128)  # Máximo 128, mínimo 8

def get_optimized_prefetch_factor(num_workers, is_multi_gpu=False):
    """Calcula prefetch_factor ultra-conservativo para evitar rate limits y saturación"""
    if num_workers == 0:
        return None
    
    hf_transfer_enabled = os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0') == '1'
    
    if hf_transfer_enabled:
        # Con hf_transfer, usar prefetch mínimo para evitar rate limits
        if is_multi_gpu:
            # Multi-GPU: máximo 2-3 items por worker
            return min(2, max(1, num_workers // 2))
        else:
            # Single-GPU: 1-2 items por worker
            return min(2, max(1, num_workers))
    else:
        # Sin hf_transfer, mantener conservativo pero algo más alto
        if is_multi_gpu:
            # Multi-GPU: 2-4 items por worker máximo
            return min(4, max(2, num_workers))
        else:
            # Single-GPU: 2-3 items por worker máximo
            return min(3, max(2, num_workers))

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

# get_num_workers() ya definida arriba - función duplicada eliminada

def balance_gpu_memory():
    """Optimizar distribución de memoria entre GPUs para DataParallel"""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # Limpiar cache de todas las GPUs
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        
        # Configurar memory fraction para balancear mejor
        total_memory = []
        for i in range(torch.cuda.device_count()):
            total_memory.append(torch.cuda.get_device_properties(i).total_memory)
        
        print(f"💾 Balanceando memoria en {torch.cuda.device_count()} GPUs")
        print(f"   📊 Memoria por GPU: {[f'{m/1e9:.1f}GB' for m in total_memory]}")
        
        # Activar optimizaciones de memoria
        torch.cuda.set_per_process_memory_fraction(0.95)  # Usar 95% de VRAM disponible
        return True
    return False

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

def save_complete_model_for_inference(model, tokenizer, output_dir):
    """
    Guarda el modelo completo en formato compatible con hrm_llm_inference.py
    Crea config.json, pytorch_model.bin y archivos del tokenizer
    """
    try:
        # Obtener el modelo sin wrapper DDP/DataParallel
        model_to_save = model
        if hasattr(model, '_orig_mod'):
            model_to_save = model._orig_mod  # DDP
        elif hasattr(model, 'module'):
            model_to_save = model.module     # DataParallel
        
        print(f"\n💾 Guardando modelo completo para inferencia en: {output_dir}")
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Guardar config.json
        config_dict = model_to_save.config.to_dict() if hasattr(model_to_save.config, 'to_dict') else vars(model_to_save.config)
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ config.json guardado")
        
        # 2. Guardar pytorch_model.bin
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), model_path)
        print(f"✅ pytorch_model.bin guardado")
        
        # 3. Guardar tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
            print(f"✅ Tokenizer guardado")
        
        # 4. Guardar generation_config.json
        generation_config = {
            'max_length': model_to_save.config.block_size,
            'do_sample': True,
            'temperature': 0.8,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'pad_token_id': getattr(tokenizer, 'pad_token_id', None) if tokenizer else None,
            'eos_token_id': getattr(tokenizer, 'eos_token_id', None) if tokenizer else None,
        }
        gen_config_path = os.path.join(output_dir, "generation_config.json")
        with open(gen_config_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(generation_config, f, indent=2, ensure_ascii=False)
        print(f"✅ generation_config.json guardado")
        
        print(f"✅ Modelo completo guardado exitosamente para hrm_llm_inference.py")
        return True
        
    except Exception as e:
        print(f"❌ Error al guardar modelo completo: {e}")
        return False

def save_checkpoint_distributed(model, optimizer, scheduler, scaler, epoch, global_step, 
                               best_val_loss, patience_counter, num_training_steps, 
                               checkpoint_path, is_distributed=False, rank=0):
    """
    Guarda checkpoint de manera compatible con entrenamiento distribuido y single GPU
    Solo el proceso de rank 0 guarda el checkpoint para evitar conflictos
    """
    # Solo el proceso principal (rank 0) debe guardar checkpoints
    if is_distributed and 'RANK' in os.environ and rank != 0:
        # Los demás procesos esperan a que rank 0 termine
        if hasattr(dist, 'is_initialized') and dist.is_initialized():
            dist.barrier()
        return
    
    try:
        # Obtener el modelo sin wrapper DDP/DataParallel
        model_to_save = model
        if hasattr(model, '_orig_mod'):
            model_to_save = model._orig_mod  # DDP
        elif hasattr(model, 'module'):
            model_to_save = model.module     # DataParallel
        
        print(f"\n💾 Guardando checkpoint en paso {global_step}...")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Crear checkpoint temporal primero para atomicidad
        temp_path = checkpoint_path + '.tmp'
        
        checkpoint_data = {
            'epoch': epoch,
            'step': global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'num_training_steps': num_training_steps,
            'distributed_training': is_distributed,
            'timestamp': time.time(),
            # Datos adicionales para inferencia
            'model_config': model_to_save.config.to_dict() if hasattr(model_to_save.config, 'to_dict') else vars(model_to_save.config),
            'tokenizer_info': {
                'tokenizer_class': 'T5Tokenizer',
                'pretrained_model_name': T5_TOKENIZER_REPO,
                'vocab_size': model_to_save.config.vocab_size,
                'pad_token_id': getattr(tokenizer, 'pad_token_id', None) if 'tokenizer' in globals() else None,
                'eos_token_id': getattr(tokenizer, 'eos_token_id', None) if 'tokenizer' in globals() else None,
                'bos_token_id': getattr(tokenizer, 'bos_token_id', None) if 'tokenizer' in globals() else None,
                'unk_token_id': getattr(tokenizer, 'unk_token_id', None) if 'tokenizer' in globals() else None,
            },
            'generation_config': {
                'max_length': model_to_save.config.block_size,
                'do_sample': True,
                'temperature': 0.8,
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.1,
                'pad_token_id': getattr(tokenizer, 'pad_token_id', None) if 'tokenizer' in globals() else None,
                'eos_token_id': getattr(tokenizer, 'eos_token_id', None) if 'tokenizer' in globals() else None,
            },
            'training_metadata': {
                'dataset_name': ACTIVE_DATASET if 'ACTIVE_DATASET' in globals() else None,
                'block_size': model_to_save.config.block_size,
                'learning_rate': LEARNING_RATE_MAX if 'LEARNING_RATE_MAX' in globals() else None,
                'batch_size': BATCH_SIZE if 'BATCH_SIZE' in globals() else None,
                'grad_accumulation_steps': GRAD_ACCUM_STEPS if 'GRAD_ACCUM_STEPS' in globals() else None,
                'seed': SEED if 'SEED' in globals() else None,
            }
        }
        
        # Guardar en archivo temporal
        torch.save(checkpoint_data, temp_path)
        
        # Mover archivo temporal al final (operación atómica)
        os.rename(temp_path, checkpoint_path)
        
        print(f"✅ Checkpoint guardado exitosamente en {checkpoint_path}")
        
        # Sincronizar con otros procesos si es distribuido
        if is_distributed and 'RANK' in os.environ and hasattr(dist, 'is_initialized') and dist.is_initialized():
            dist.barrier()
            
    except Exception as e:
        print(f"❌ Error al guardar checkpoint: {e}")
        
        # Limpiar archivo temporal si existe
        temp_path = checkpoint_path + '.tmp'
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        
        # Sincronizar con otros procesos incluso si hay error
        if is_distributed and 'RANK' in os.environ and hasattr(dist, 'is_initialized') and dist.is_initialized():
            dist.barrier()
        
        raise e

def load_checkpoint_distributed(checkpoint_path, model, optimizer, scheduler, scaler, 
                               device, is_distributed=False, rank=0):
    """
    Carga checkpoint de manera compatible con entrenamiento distribuido y single GPU
    """
    if not os.path.exists(checkpoint_path):
        print("--- No se encontró checkpoint. Empezando entrenamiento desde cero. ---")
        return False, 0, 0, float('inf'), 0
    
    try:
        print(f"--- Reanudando entrenamiento desde el checkpoint: {checkpoint_path} ---")
        
        # Sincronizar antes de cargar si es distribuido
        if is_distributed and 'RANK' in os.environ and hasattr(dist, 'is_initialized') and dist.is_initialized():
            dist.barrier()
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Obtener el modelo sin wrapper para cargar estado
        model_to_load = model
        if hasattr(model, '_orig_mod'):
            model_to_load = model._orig_mod  # DDP
        elif hasattr(model, 'module'):
            model_to_load = model.module     # DataParallel
        
        # Cargar estados
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Extraer información del checkpoint
        start_epoch = checkpoint['epoch']
        start_step = checkpoint.get('step', 0)
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint.get('patience_counter', 0)
        
        print(f"✅ Checkpoint cargado exitosamente")
        print(f"   📊 Época: {start_epoch + 1}, Paso: {start_step}")
        print(f"   🏆 Mejor pérdida de validación: {best_val_loss:.4f}")
        
        return True, start_epoch, start_step, best_val_loss, patience_counter
        
    except Exception as e:
        print(f"❌ Error al cargar checkpoint: {e}")
        print("--- Empezando entrenamiento desde cero. ---")
        return False, 0, 0, float('inf'), 0

# Configurar distributed training
is_distributed, rank, world_size, local_rank = setup_distributed()

# Configurar dispositivo
if is_distributed and 'RANK' in os.environ:
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

def balance_gpu_memory():
    """Balancear memoria GPU antes de crear modelo"""
    if torch.cuda.is_available():
        # Limpiar cache
        torch.cuda.empty_cache()
        
        # Balancear memoria entre GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            for i in range(num_gpus):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
        
        print(f"🧹 Memoria GPU balanceada entre {num_gpus} GPU(s)")

# Balancear memoria GPU antes de crear modelo
balance_gpu_memory()

# Autenticación con Hugging Face Hub (solo si no es import-only)
if not os.environ.get('HRM_IMPORT_ONLY'):
    try:
        from huggingface_hub import login
        
        # Intentar obtener token de variable de entorno
        HF_TOKEN = os.environ.get('HF_TOKEN')
        
        if HF_TOKEN:
            login(token=HF_TOKEN)
            print("✅ Hugging Face token loaded from environment variable.")
        else:
            # Intentar login interactivo (útil para desarrollo local)
            try:
                login()
                print("✅ Hugging Face authentication successful.")
            except Exception as e:
                print(f"⚠️  HF authentication failed: {e}")
                print("💡 Para usar HF Pro, configura HF_TOKEN o ejecuta: huggingface-cli login")
                HF_TOKEN = None
    except ImportError:
        print("⚠️  huggingface_hub login not available")
        HF_TOKEN = os.environ.get('HF_TOKEN')
        if HF_TOKEN:
            HfFolder.save_token(HF_TOKEN)
            print("Hugging Face token loaded (legacy method).")
        else:
            print("HF_TOKEN secret not found.")
            HF_TOKEN = None
else:
    # Solo para imports, no hacer login
    HF_TOKEN = None

# Tokenizer se carga solo cuando se ejecuta el script directamente
# Verificar si solo se está importando para usar las clases
if not os.environ.get('HRM_IMPORT_ONLY'):
    print("Loading tokenizer (T5 slow)...")
    tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_REPO, use_fast=False, legacy=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
else:
    # Solo definir variable para imports, el tokenizer se carga después
    tokenizer = None

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
    print("--- CARGANDO DATASETS PARA MEZCLA (MODELO PEQUEÑO 100M) ---")
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
    """Función de tokenización optimizada para C4 streaming masivo"""
    texts = []
    
    # Manejar diferentes campos de texto según el dataset
    if "text" in examples:
        # Formato estándar (C4, OpenWebText, Pile)
        text_field = examples["text"]
    elif "content" in examples:
        # Algunos datasets usan 'content'
        text_field = examples["content"]
    elif "document" in examples:
        # Algunos datasets usan 'document'
        text_field = examples["document"]
    else:
        # Intentar encontrar el primer campo que parezca texto
        for key in examples.keys():
            if isinstance(examples[key][0], str) and len(examples[key][0]) > 50:
                text_field = examples[key]
                print(f"Usando campo '{key}' como texto")
                break
        else:
            raise ValueError(f"No se encontró campo de texto válido en el dataset. Campos disponibles: {list(examples.keys())}")
    
    # Optimización para C4: procesar textos con filtro eficiente
    for text in text_field:
        if isinstance(text, str) and len(text) > 100:  # Filtro optimizado para calidad
            texts.append(str(text) + tokenizer.eos_token)
    
    # Tokenización optimizada para streaming masivo
    # Sin padding para mayor eficiencia en memoria y procesamiento
    return tokenizer(
        texts, 
        truncation=True, 
        max_length=BLOCK_SIZE, 
        padding=False,  # Eliminado padding para streaming efficiency
        add_special_tokens=False,  # Ya agregamos EOS token manualmente
        return_attention_mask=False  # Sin attention mask para optimizar memoria
    )

print("Applying tokenization function (on-the-fly)...")
tokenized_splits = {}

# Configuración para multi-GPU (necesario antes del loop de tokenización)
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
is_multi_gpu = num_gpus > 1
safe_num_workers = get_dataloader_workers() if not os.environ.get('HRM_IMPORT_ONLY') else 0
tokenization_workers = get_tokenization_workers() if not os.environ.get('HRM_IMPORT_ONLY') else 0

# Función para verificar si es IterableDataset (necesaria en el loop)
def is_iterable_dataset(dataset):
    return isinstance(dataset, IterableDataset)

# Detectar columnas a eliminar dinámicamente
sample = next(iter(raw_datasets["train"]))
columns_to_remove = [col for col in sample.keys() if col not in ["input_ids", "attention_mask"]]
print(f"Columnas detectadas en el dataset: {list(sample.keys())}")
print(f"Columnas a eliminar después de tokenización: {columns_to_remove}")

for split_name in ["train", "validation"]:
    # Optimización para C4 streaming: batch size más grande y configuración eficiente
    if ACTIVE_DATASET == "c4" and is_multi_gpu:
        # Configuración optimizada para C4 streaming masivo
        batch_size_tokenization = 2000  # Batch más grande para C4
        num_proc = min(tokenization_workers, 8)  # Paralelización limitada
        print(f"🚀 Tokenización optimizada C4: batch_size={batch_size_tokenization}, num_proc={num_proc}")
    else:
        batch_size_tokenization = 1000
        num_proc = min(tokenization_workers, 4)
    
    # Para IterableDataset no usar num_proc ni desc (no soportados)
    if is_iterable_dataset(raw_datasets[split_name]):
        print(f"🚀 Tokenizando {split_name} para C4 streaming (IterableDataset)")
        tokenized_splits[split_name] = raw_datasets[split_name].map(
            tokenize_function, 
            batched=True,
            batch_size=batch_size_tokenization,
            remove_columns=columns_to_remove
        ).with_format("torch")
    else:
        tokenized_splits[split_name] = raw_datasets[split_name].map(
            tokenize_function, 
            batched=True,
            batch_size=batch_size_tokenization,
            num_proc=num_proc,
            remove_columns=columns_to_remove,
            desc=f"Tokenizando {split_name} para C4 streaming"
        ).with_format("torch")

# ### FIX DATALOADER ###: Variables ya definidas arriba


# Función is_iterable_dataset ya definida arriba

# Detectar si los datasets son iterables
train_is_iterable = is_iterable_dataset(tokenized_splits["train"])
val_is_iterable = is_iterable_dataset(tokenized_splits["validation"])

def custom_collate_fn(batch):
    """
    Collate function personalizada para manejar tensores de diferentes longitudes
    Hace padding a la longitud máxima del batch
    """
    # Extraer input_ids del batch
    input_ids = [item['input_ids'] for item in batch]
    
    # Encontrar la longitud máxima en el batch
    max_length = max(len(ids) for ids in input_ids)
    
    # Hacer padding con tokenizer.pad_token_id
    padded_input_ids = []
    for ids in input_ids:
        if len(ids) < max_length:
            # Pad con el pad_token_id
            padding_length = max_length - len(ids)
            padded_ids = torch.cat([ids, torch.full((padding_length,), tokenizer.pad_token_id, dtype=ids.dtype)])
        else:
            padded_ids = ids
        padded_input_ids.append(padded_ids)
    
    # Crear attention_mask
    attention_mask = []
    for ids in input_ids:
        mask = torch.ones(max_length, dtype=torch.long)
        if len(ids) < max_length:
            mask[len(ids):] = 0  # Marcar padding como 0
        attention_mask.append(mask)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(attention_mask)
    }

print(f"Creando DataLoaders optimizados con {safe_num_workers} workers...")

# Configuración optimizada para multi-GPU (variables ya definidas arriba)

# Configuración optimizada para C4 streaming con multi-GPU
# Configuración de DataLoader basada en workers disponibles
if safe_num_workers > 0:
    prefetch_factor = get_optimized_prefetch_factor(safe_num_workers, is_multi_gpu)
    persistent_workers = True
    
    if is_multi_gpu:
        # Para DataParallel usar GPU 0, para distribuido usar LOCAL_RANK
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        pin_memory_device = f"cuda:{local_rank}"  # Pin a GPU específica
        multiprocessing_context = "fork"  # Compatible sin __main__ guard
        
        hf_status = "🚀 hf_transfer optimizado" if os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') == '1' else "🐌 standard I/O"
        print(f"🚀 Configuración Multi-GPU ({hf_status}): prefetch_factor={prefetch_factor}, workers={safe_num_workers}")
        print(f"   📊 Buffer optimizado para streaming dataset")
    else:
        pin_memory_device = None
        multiprocessing_context = None
        
        hf_status = "hf_transfer" if os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') == '1' else "standard"
        print(f"🔧 Configuración Single-GPU ({hf_status}): prefetch_factor={prefetch_factor}, workers={safe_num_workers}")
else:
    # Sin workers - modo import o configuración incorrecta
    prefetch_factor = None
    persistent_workers = False
    pin_memory_device = None
    multiprocessing_context = None
    print(f"⚠️  Sin workers disponibles: prefetch_factor=None, workers={safe_num_workers}")
    print(f"   💡 Verifica que HRM_IMPORT_ONLY no esté activado para entrenamiento")

# Configurar argumentos del DataLoader condicionalmente
train_kwargs = {
    "batch_size": BATCH_SIZE,
    "num_workers": safe_num_workers,
    "pin_memory": True,
    "persistent_workers": persistent_workers,
    "shuffle": False,  # False para datasets iterables
    "collate_fn": custom_collate_fn,
    "drop_last": is_multi_gpu,  # Drop last para consistency en multi-GPU
}

# Solo agregar argumentos no-None
if prefetch_factor is not None:
    train_kwargs["prefetch_factor"] = prefetch_factor
    print(f"   ✅ DataLoader configurado con prefetch_factor={prefetch_factor}")
else:
    print(f"   ⚠️  DataLoader SIN prefetch_factor (workers={safe_num_workers})")
if pin_memory_device is not None:
    train_kwargs["pin_memory_device"] = pin_memory_device
if multiprocessing_context is not None:
    train_kwargs["multiprocessing_context"] = multiprocessing_context

train_loader = DataLoader(tokenized_splits["train"], **train_kwargs)

# Configurar argumentos del validation DataLoader condicionalmente
val_kwargs = {
    "batch_size": BATCH_SIZE,
    "num_workers": safe_num_workers,
    "pin_memory": True,
    "persistent_workers": persistent_workers,
    "shuffle": False,
    "collate_fn": custom_collate_fn,
    "drop_last": False,  # No drop last en validación
}

# Solo agregar argumentos no-None
if prefetch_factor is not None:
    val_kwargs["prefetch_factor"] = prefetch_factor
if pin_memory_device is not None:
    val_kwargs["pin_memory_device"] = pin_memory_device
if multiprocessing_context is not None:
    val_kwargs["multiprocessing_context"] = multiprocessing_context

val_loader = DataLoader(tokenized_splits["validation"], **val_kwargs)

# Sistema de buffer inteligente para C4 streaming
class StreamingBufferWrapper:
    """Buffer inteligente para datasets streaming masivos como C4"""
    def __init__(self, dataloader, buffer_size=None, min_buffer_ratio=0.4):
        self.dataloader = dataloader
        # Buffer adaptativo basado en número de GPUs
        self.buffer_size = buffer_size or get_optimized_buffer_size(num_gpus, 'medium')
        self.min_buffer_ratio = min_buffer_ratio
        self.buffer = []
        self.iterator = iter(dataloader)
        self._fill_initial_buffer()
        
    def _fill_initial_buffer(self):
        """Llenar buffer inicial para evitar GPU starvation"""
        target_size = int(self.buffer_size * 0.8)  # 80% inicial
        try:
            for _ in range(target_size):
                batch = next(self.iterator)
                self.buffer.append(batch)
            print(f"🔋 Buffer inicial llenado: {len(self.buffer)} batches")
        except StopIteration:
            print(f"⚠️  Dataset agotado durante llenado inicial: {len(self.buffer)} batches")
    
    def _maintain_buffer(self):
        """Mantener buffer mínimo para streaming continuo"""
        min_size = int(self.buffer_size * self.min_buffer_ratio)
        while len(self.buffer) < min_size:
            try:
                batch = next(self.iterator)
                self.buffer.append(batch)
            except StopIteration:
                break
    
    def __iter__(self):
        while self.buffer:
            # Mantener buffer antes de entregar batch
            self._maintain_buffer()
            if self.buffer:
                yield self.buffer.pop(0)

# Aplicar buffer wrapper solo en multi-GPU para C4 streaming
if is_multi_gpu and ACTIVE_DATASET == "c4":
    print(f"🚀 Activando buffer inteligente para C4 streaming multi-GPU")
    # Buffer más grande para mejor utilización de CPU en paralelo
    buffer_size = get_optimized_buffer_size(num_gpus, 'large')  # Optimizado conservativo
    train_loader = StreamingBufferWrapper(train_loader, buffer_size=buffer_size)
    print(f"📦 Buffer streaming: {buffer_size} batches para {num_gpus} GPUs")

# Balancear memoria GPU antes de crear modelo
balance_gpu_memory()

# Crear modelo solo si no es import-only
if not os.environ.get('HRM_IMPORT_ONLY'):
    config = HRMText1Config(vocab_size=len(tokenizer), block_size=BLOCK_SIZE, **MODEL_PARAMS)
    model = HRMText1(config).to(device)

# Envolver modelo para multi-GPU (solo si no es import-only)
if not os.environ.get('HRM_IMPORT_ONLY') and is_distributed:
    if world_size > 1 and 'RANK' in os.environ:
        # Entrenamiento distribuido real con torchrun
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"🔗 Modelo envuelto con DDP en GPU {local_rank}")
    elif world_size > 1:
        # Auto-inicialización multi-GPU con DataParallel optimizado
        device_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=device_ids, output_device=0)
        
        # Optimizaciones para mejor balanceo
        torch.backends.cudnn.benchmark = True  # Optimizar para tamaños fijos
        torch.backends.cuda.matmul.allow_tf32 = True  # Acelerar matmul
        torch.backends.cudnn.allow_tf32 = True  # Acelerar convs
        
        print(f"🔗 Modelo envuelto con DataParallel usando {len(device_ids)} GPUs")
        print(f"   🎯 GPU principal: {device}")
        print(f"   📋 GPUs utilizadas: {device_ids}")
        print(f"   ⚡ Optimizaciones activadas: cuDNN benchmark, TF32")
    else:
        print(f"📱 Modelo en single-GPU mode: {device}")
else:
    print(f"📱 Modelo en single-GPU mode: {device}")

# Inicializar variables globales de entrenamiento
start_epoch = 0
start_step = 0
best_val_loss = float('inf')
patience_counter = 0
CHECKPOINT_STEPS = 1000
global_step = 0

# Variables para tracking de velocidad y throughput
step_times = []
epoch_start_time = None
samples_processed = 0

# Contar parámetros (solo si el modelo fue creado)
if not os.environ.get('HRM_IMPORT_ONLY'):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Número de parámetros del modelo: {total_params:,}")
    print(f"Estimación de VRAM necesaria: {total_params * 4 / 1e9:.1f} GB (solo parámetros)")
else:
    total_params = 0  # Valor por defecto para imports

# Solo crear optimizador y continuar con entrenamiento si no es import-only
if not os.environ.get('HRM_IMPORT_ONLY'):
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

    # Scheduler coseno con warmup para decaimiento más suave

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
    if TENSORBOARD_AVAILABLE and (not is_distributed or rank == 0):
        writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
        print(f"📊 TensorBoard Writer inicializado")
        
        # Log hyperparameters
        hyperparams = {
            'model/n_embd': MODEL_PARAMS['n_embd'],
            'model/n_layers': MODEL_PARAMS['n_layers'],
            'model/n_head': MODEL_PARAMS['n_head'],
            'model/d_ff': MODEL_PARAMS['d_ff'],
            'model/block_size': BLOCK_SIZE,
            'train/batch_size': BATCH_SIZE,
            'train/grad_accum_steps': GRAD_ACCUM_STEPS,
            'train/learning_rate_max': LEARNING_RATE_MAX,
            'train/warmup_ratio': WARMUP_RATIO,
            'train/num_epochs': NUM_EPOCHS,
            'model/total_params': total_params,
        }
        
        # Log hyperparams como texto
        hyperparams_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
        writer.add_text("Hyperparameters", hyperparams_text, 0)

    # --- CONFIGURACIÓN PARA MODIFICACIÓN DE LEARNING RATE ---
    # Configuración unificada para entrenamiento continuo
    # NEW_LEARNING_RATE se usa automáticamente cuando CONTINUE_TRAINING=True
    NEW_LEARNING_RATE = 8e-4   # LR reducido para fine-tuning con nuevo dataset

    # Checkpoint loading (variables ya inicializadas globalmente)

    # Cargar checkpoint usando la función distribuida
    checkpoint_loaded, start_epoch, start_step, best_val_loss, patience_counter = load_checkpoint_distributed(
        checkpoint_path=CHECKPOINT_PATH,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        is_distributed=is_distributed,
        rank=rank if is_distributed else 0
    )

    # Manejar modificación de learning rate si se cargó checkpoint
    if checkpoint_loaded and CONTINUE_TRAINING:
        print(f"--- Modificando learning rate de {optimizer.param_groups[0]['lr']:.6f} a {NEW_LEARNING_RATE:.6f} ---")
        for param_group in optimizer.param_groups:
            param_group['lr'] = NEW_LEARNING_RATE
        print(f"✅ Learning rate modificado exitosamente a: {NEW_LEARNING_RATE:.6f}")

    # Verificar y ajustar scheduler si el dataset cambió
    if checkpoint_loaded:
        # Recargar checkpoint para verificar num_training_steps (se podría optimizar guardándolo en la función)
        if os.path.exists(CHECKPOINT_PATH):
            temp_checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            checkpoint_training_steps = temp_checkpoint.get('num_training_steps', 0)
            
            if checkpoint_training_steps != num_training_steps:
                print(f"Dataset cambió. Reajustando scheduler: {checkpoint_training_steps} -> {num_training_steps}")
                scheduler = get_cosine_schedule_with_warmup(
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
                # CRITICAL FIX: Actualizar start_step al nuevo valor calculado
                start_step = new_step
                print(f"🔄 start_step actualizado de checkpoint para nuevo dataset: {start_step}")
    # Actualizar global_step con el valor cargado (ahora potencialmente ajustado)
    global_step = start_step

def main_training():
    """Función principal de entrenamiento con métricas avanzadas de TensorBoard"""
    global global_step, writer, step_times, epoch_start_time, samples_processed, tokenizer
    global best_val_loss, patience_counter, start_epoch, start_step, model, optimizer
    
    # Cargar tokenizer solo cuando se ejecuta entrenamiento
    print("Loading tokenizer (T5 slow)...")
    tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_REPO, use_fast=False, legacy=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    # Configurar HuggingFace settings para mejor compatibilidad con Colab
    try:
        # Configurar variables de entorno para HuggingFace
        import os
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')  # Evitar warnings en Jupyter/Colab
        os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')  # Habilitar transferencias rápidas de HF
        if 'google.colab' in str(type(get_ipython() if 'get_ipython' in globals() else '')):
            print("🔧 Configuración optimizada para Google Colab detectada")
    except:
        pass
    
    # Métricas de velocidad
    step_start_time = time.time()
    
    # Determinar épocas finales para entrenamiento continuo
    if CONTINUE_TRAINING and start_epoch >= NUM_EPOCHS:
        final_epochs = start_epoch + 2  # Entrenar 2 épocas adicionales
        print(f"🔄 Modo continuo: entrenando épocas {start_epoch+1} a {final_epochs}")
    else:
        final_epochs = NUM_EPOCHS
    
    for epoch in range(start_epoch, final_epochs):
        epoch_start_time = time.time()
        print(f"\\n🚀 Iniciando Época {epoch+1}/{final_epochs}")
        
        model.train()
        optimizer.zero_grad()
        
        progress = tqdm(train_loader, desc=f"Época {epoch+1}/{final_epochs}")
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        for i, batch in enumerate(progress):
            step_start_time = time.time()
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = input_ids.clone()
            
            # Contar muestras procesadas
            samples_processed += input_ids.size(0)
            
            # Mixed precision forward pass
            with torch.amp.autocast(
                device_type=device.type, 
                dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, 
                enabled=MIXED_PRECISION
            ):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRAD_ACCUM_STEPS
                
                # Para DataParallel, loss puede ser un tensor con múltiples valores
                if hasattr(model, 'module') and loss.dim() > 0:
                    loss = loss.mean()
            
            # Backward pass
            if loss is not None and torch.isfinite(loss):
                scaler.scale(loss).backward()
                epoch_loss += loss.item() * GRAD_ACCUM_STEPS
                epoch_steps += 1
                
                if (i + 1) % GRAD_ACCUM_STEPS == 0:
                    # Gradient clipping y update
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    
                    # Métricas de tiempo y velocidad
                    step_end_time = time.time()
                    step_time = step_end_time - step_start_time
                    step_times.append(step_time)
                    
                    # Mantener solo últimos 100 tiempos para rolling average
                    if len(step_times) > 100:
                        step_times.pop(0)
                    
                    # Monitoreo específico para C4 streaming - detectar GPU starvation
                    if ACTIVE_DATASET == "c4" and len(step_times) >= 10:
                        recent_avg_time = sum(step_times[-10:]) / 10
                        if recent_avg_time > 2.0:  # Si los steps toman más de 2 segundos
                            print(f"⚠️  Posible GPU starvation detectado: {recent_avg_time:.2f}s por step")
                            print(f"   💡 Considera aumentar prefetch_factor o buffer size")
                        elif recent_avg_time < 0.1:  # Steps muy rápidos pueden indicar datos insuficientes
                            print(f"🚀 GPU utilización óptima: {recent_avg_time:.3f}s por step")
                    
                    # Guardar checkpoint cada CHECKPOINT_STEPS
                    if global_step % CHECKPOINT_STEPS == 0:
                        save_checkpoint_distributed(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            epoch=epoch,
                            global_step=global_step,
                            best_val_loss=best_val_loss,
                            patience_counter=patience_counter,
                            num_training_steps=num_training_steps,
                            checkpoint_path=CHECKPOINT_PATH,
                            is_distributed=is_distributed,
                            rank=rank if is_distributed else 0
                        )
                        
                        # Guardar modelo completo para inferencia también
                        if rank == 0 or not is_distributed:
                            save_complete_model_for_inference(
                                model=model,
                                tokenizer=tokenizer,
                                output_dir=OUTPUT_DIR
                            )
                    
                    # Actualizar progress bar
                    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else LEARNING_RATE_MAX
                    progress.set_postfix({
                        "loss": f"{loss.item()*GRAD_ACCUM_STEPS:.4f}", 
                        "lr": f"{current_lr:.2e}", 
                        "step": global_step,
                        "s/step": f"{step_time:.2f}"
                    })
                    
                    # TensorBoard logging expandido (incluyendo código anterior)
                    if writer is not None and global_step % 10 == 0:
                        train_loss = loss.item() * GRAD_ACCUM_STEPS
                        
                        # Métricas básicas
                        writer.add_scalar('Loss/Train', train_loss, global_step)
                        writer.add_scalar('Learning_Rate/Current', current_lr, global_step)
                        
                        # Métricas de velocidad y throughput
                        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
                        writer.add_scalar('Performance/Avg_Step_Time_Sec', avg_step_time, global_step)
                        writer.add_scalar('Performance/Steps_Per_Second', 1.0 / (avg_step_time + 1e-8), global_step)
                        
                        # Throughput en muestras por segundo
                        samples_per_sec = (input_ids.size(0) * GRAD_ACCUM_STEPS) / (avg_step_time + 1e-8)
                        writer.add_scalar('Performance/Samples_Per_Second', samples_per_sec, global_step)
                        writer.add_scalar('Performance/Tokens_Per_Second', samples_per_sec * BLOCK_SIZE, global_step)
                        
                        # Tiempo transcurrido desde inicio de época
                        if epoch_start_time is not None:
                            epoch_elapsed = time.time() - epoch_start_time
                            writer.add_scalar('Performance/Epoch_Time_Minutes', epoch_elapsed / 60.0, global_step)
                        
                        # [Resto del código TensorBoard anterior se mantiene]
        
        # Validación al final de cada época
        print(f"\\n📊 Ejecutando validación para época {epoch+1}")
        model.eval()
        
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            # Evaluar en una muestra representativa de validación
            for i, batch in enumerate(val_loader):
                if i >= 100:  # Limitar evaluación para eficiencia
                    break
                    
                input_ids = batch['input_ids'].to(device)
                
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
                    enabled=MIXED_PRECISION
                ):
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / max(val_steps, 1)
        print(f"📊 Pérdida de validación: {avg_val_loss:.4f}")
        
        # Guardar mejor modelo si es necesario
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Guardar mejor modelo
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(model_to_save.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 Nuevo mejor modelo guardado! Pérdida: {best_val_loss:.4f}")
            
            # Guardar modelo completo para inferencia cuando es el mejor
            if rank == 0 or not is_distributed:
                save_complete_model_for_inference(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=OUTPUT_DIR
                )
        else:
            patience_counter += 1
            print(f"⏳ Paciencia: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
        # Log validación en TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/Validation', avg_val_loss, global_step)
            writer.add_scalar('Model/Best_Val_Loss', best_val_loss, global_step)
            writer.add_scalar('Training/Patience_Counter', patience_counter, global_step)
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping activado. Mejor pérdida: {best_val_loss:.4f}")
            break
        
        model.train()  # Volver a modo entrenamiento
        
        # Log tiempo total de época
        if writer is not None and epoch_start_time is not None:
            total_epoch_time = time.time() - epoch_start_time
            writer.add_scalar('Performance/Total_Epoch_Time_Minutes', total_epoch_time / 60.0, global_step)
            writer.add_scalar('Performance/Avg_Loss_Per_Epoch', epoch_loss / max(epoch_steps, 1), global_step)
    
    print("Entrenamiento completado en main_training()!")

def save_final_model():
    """Guarda el modelo final y lo sube a Hugging Face Hub si está configurado"""
    # Guardar modelo final
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

    if os.path.exists(BEST_MODEL_PATH):
        print(f"Cargando el mejor modelo desde '{BEST_MODEL_PATH}' para el guardado final.")
        model_to_save.load_state_dict(torch.load(BEST_MODEL_PATH))

    model_to_save.save_pretrained(OUTPUT_DIR, safe_serialization=False)
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
                commit_message=f"Upload HRM-Models 100M medium model"
            )

            # Subir el tokenizador
            tokenizer.push_to_hub(
                HF_REPO_ID,
                token=HF_TOKEN,
                commit_message=f"Upload tokenizer for HRM-Models 100M medium model"
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

def test_model_and_summary():
    """Prueba el modelo final y muestra el resumen del entrenamiento"""
    print("\n--- Probando la Generación del Modelo Final ---")
    try:
        inference_model = HRMText1.from_pretrained(OUTPUT_DIR).to(device)
        # torch.compile deshabilitado para ahorrar memoria
        # if torch.__version__.startswith("2") and hasattr(torch, 'compile'):
        #     inference_model = torch.compile(inference_model)
        
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

# Ejecutar el entrenamiento principal
if __name__ == "__main__" and not os.environ.get('HRM_IMPORT_ONLY'):
    main_training()
    save_final_model()
    test_model_and_summary()
