# -*- coding: utf-8 -*-
"""
HRM-Models Training Script con Tokenizador HuggingFace - MODELO MICRO ~10M PARÁMETROS  
VERSIÓN MEJORADA: Usando tokenizadores profesionales de HuggingFace

🖥️  CARACTERÍSTICAS:
- Tokenizador HuggingFace (GPT2, GPT2-Spanish, etc.)
- Vocabulario profesional (50K+ tokens)
- Mejor soporte multilingüe (español/inglés)
- Arquitectura HRM optimizada
- Sin dependencias de transformers para el modelo (solo tokenizer)
"""

import os, random, multiprocessing as mp, atexit, math, time
from typing import List, Dict, Optional, Tuple
import argparse

# Configurar método de multiprocessing antes de cualquier uso
if __name__ == '__main__':
    mp.set_start_method('fork', force=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from torch.optim import AdamW

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

class HRMText1Config(SimpleConfig):
    model_type = "hrm_text1"
    
    def __init__(self, 
                 vocab_size=50257,          # HF tokenizer default
                 block_size=128,            # Micro model context
                 n_embd=256,                # Micro model embeddings
                 n_head=8,                  # Micro model heads
                 n_layers=6,                # Micro model layers
                 d_ff=1024,                 # Micro model FFN
                 dropout=0.1,
                 pad_token_id=0,            
                 halt_max_steps=4,          # HRM halt steps
                 ponder_loss_weight=1e-2,
                 halt_bias_init=-0.5,
                 use_rotary_embeddings=True,
                 rotary_embedding_base=10000,
                 use_flash_attention=True,
                 gradient_checkpointing=False,
                 h_update_period=2,         # H-module update period
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
        self.h_update_period = h_update_period

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
    
    def forward(self, x):
        device = x.device
        norm = x.norm(dim=-1, keepdim=True, dtype=torch.float32).clamp(min=self.eps)
        return (x / norm.to(device)) * self.weight

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
        
        self.convergence_threshold = 1e-3
        self.max_l_steps = config.halt_max_steps
        self.h_update_period = getattr(config, 'h_update_period', 4)
    
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

class HRMText1(nn.Module):
    """HRM Model with full hierarchical temporal separation"""
    
    def __init__(self, config: HRMText1Config):
        super().__init__()
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
        
        # Compartir pesos entre token embeddings y lm_head
        self.lm_head.weight = self.token_embeddings.weight
        
        # Inicialización
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
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
        
        # HRM layers con estados separados
        for step_count, layer in enumerate(self.layers):
            z_H, z_L, hrm_info = layer(z_H, z_L, step_count, attn_mask=causal_mask, key_padding_mask=attention_mask)
        
        # Final norm y lm_head
        output = self.final_norm(z_H)
        logits = self.lm_head(output)
        
        # Si se proporcionan labels, calcular loss
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
            return (loss, logits)
        
        return logits

# Hugging Face Hub imports
try:
    from huggingface_hub import HfApi
    HF_API_AVAILABLE = True
except ImportError:
    HF_API_AVAILABLE = False
    print("⚠️ WARNING: huggingface_hub no está disponible. No se podrá subir al Hub.")

# ==============================================================================
# --- Dataset Handling ---
# ==============================================================================

class TextDataset(IterableDataset):
    """Dataset para texto usando tokenizador HF"""
    
    def __init__(self, tokenizer, texts: List[str], block_size: int = 128, split_type: str = "train"):
        self.tokenizer = tokenizer
        self.texts = texts
        self.block_size = block_size
        self.split_type = split_type
        
        print(f"📚 Dataset {split_type}: {len(texts)} textos, block_size={block_size}")
    
    def __iter__(self):
        for text in self.texts:
            if not text or len(text.strip()) < 10:
                continue
            
            try:
                # Tokenizar con HF - limitar longitud máxima para evitar errores
                tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=1024, truncation=True)
                
                # Dividir en chunks del tamaño del bloque
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

def load_dataset_hf(tokenizer, split: str = "train", num_samples: int = 1000):
    """Cargar dataset usando datasets de HuggingFace"""
    try:
        from datasets import load_dataset
        
        print(f"📥 Cargando dataset '{split}' con {num_samples} samples...")
        
        # Usar C4 en inglés como dataset principal
        dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
        
        texts = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            text = item.get('text', '')
            if len(text.strip()) > 50:  # Filtrar textos muy cortos
                # Limitar longitud del texto para evitar secuencias muy largas
                text = text.strip()[:2000]  # Máximo 2000 caracteres
                texts.append(text)
        
        print(f"✅ Cargados {len(texts)} textos válidos")
        return texts
        
    except Exception as e:
        print(f"⚠️ Error cargando dataset HF: {e}")
        return create_synthetic_dataset(num_samples)

def create_synthetic_dataset(num_samples: int = 1000):
    """Crear dataset sintético para testing"""
    print(f"🔧 Creando dataset sintético con {num_samples} samples...")
    
    templates = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "To be or not to be, that is the question.",
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael.",
        "En un lugar de la Mancha, de cuyo nombre no quiero acordarme.",
        "Había una vez en un reino muy lejano.",
        "def function(x): return x * 2",
        "import torch\nimport numpy as np\n\nclass Model(nn.Module):",
        "The weather today is sunny and warm.",
    ]
    
    texts = []
    for i in range(num_samples):
        # Crear textos variados combinando plantillas
        text = random.choice(templates)
        if random.random() > 0.5:
            text += " " + random.choice(templates)
        texts.append(text)
    
    print(f"✅ Dataset sintético creado: {len(texts)} textos")
    return texts

# ==============================================================================
# --- Training Functions ---
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

def train_hrm_hf(
    tokenizer_name: str = "openai-community/gpt2",
    output_dir: str = "./hrm-micro-10m-hf",
    num_train_samples: int = 10000,
    num_val_samples: int = 1000,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    num_epochs: int = 3,
    save_steps: int = 500,
    eval_steps: int = 200,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 100,
):
    """Entrenar modelo HRM con tokenizador HuggingFace"""
    
    print(f"🚀 Iniciando entrenamiento HRM con tokenizador HF")
    print(f"📊 Configuración:")
    print(f"   Tokenizador: {tokenizer_name}")
    print(f"   Directorio salida: {output_dir}")
    print(f"   Samples entrenamiento: {num_train_samples}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Épocas: {num_epochs}")
    
    # Crear tokenizador HF
    print(f"🔧 Cargando tokenizador: {tokenizer_name}")
    tokenizer = create_tokenizer(tokenizer_name)
    
    # Crear configuración del modelo
    config = HRMText1Config(
        vocab_size=len(tokenizer),
        block_size=128,
        n_embd=256,
        n_head=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
        tokenizer_type='huggingface',
        hf_tokenizer_name=tokenizer_name,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    print(f"📐 Configuración del modelo:")
    print(f"   Vocabulario: {config.vocab_size:,} tokens")
    print(f"   Embeddings: {config.n_embd}")
    print(f"   Capas: {config.n_layers}")
    print(f"   Cabezas atención: {config.n_head}")
    
    # Crear modelo
    model = HRMText1(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🧠 Modelo creado:")
    print(f"   Total parámetros: {total_params:,}")
    print(f"   Parámetros entrenables: {trainable_params:,}")
    
    # Cargar datasets
    print("📚 Cargando datasets...")
    train_texts = load_dataset_hf(tokenizer, "train", num_train_samples)
    val_texts = load_dataset_hf(tokenizer, "validation", num_val_samples)
    
    train_dataset = TextDataset(tokenizer, train_texts, config.block_size, "train")
    val_dataset = TextDataset(tokenizer, val_texts, config.block_size, "validation")
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Crear optimizador
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler con warmup
    total_steps = len(train_texts) // batch_size * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps/total_steps
    )
    
    print(f"🎯 Entrenamiento configurado:")
    print(f"   Steps totales estimados: {total_steps:,}")
    print(f"   Warmup steps: {warmup_steps}")
    
    # Training loop
    model.train()
    step = 0
    best_val_loss = float('inf')
    
    print(f"\n🎉 ¡Iniciando entrenamiento!")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\n📅 Época {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            step += 1
            
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            # Forward pass usando interfaz HRM completa
            # Crear attention mask si no existe
            attention_mask = torch.ones_like(input_ids)
            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
            
            # El modelo HRM devuelve (loss, logits, ...) cuando se pasan labels
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Logging
            if step % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")
            
            # Evaluación
            if step % eval_steps == 0:
                model.eval()
                val_loss = 0
                val_batches = 0
                
                print("🔍 Evaluando...")
                with torch.no_grad():
                    for val_batch in val_loader:
                        if val_batches >= 50:  # Evaluar solo 50 batches
                            break
                        
                        val_input_ids = val_batch['input_ids']
                        val_labels = val_batch['labels']
                        
                        # Crear attention mask para validación
                        val_attention_mask = torch.ones_like(val_input_ids)
                        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                            val_attention_mask = (val_input_ids != tokenizer.pad_token_id).long()
                        
                        # Usar interfaz HRM completa
                        val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
                        val_loss += (val_outputs[0] if isinstance(val_outputs, tuple) else val_outputs.loss).item()
                        val_batches += 1
                
                avg_val_loss = val_loss / max(val_batches, 1)
                print(f"📊 Step {step} | Val Loss: {avg_val_loss:.4f}")
                
                # Guardar mejor modelo
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(output_dir, "best_model")
                    save_model_hf(model, tokenizer, best_model_path, config, step)
                    print(f"💎 Nuevo mejor modelo guardado: {avg_val_loss:.4f}")
                
                model.train()
            
            # Guardar checkpoint
            if step % save_steps == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
                save_model_hf(model, tokenizer, checkpoint_path, config, step)
                print(f"💾 Checkpoint guardado: {checkpoint_path}")
            
            # Early stopping en caso de pérdida muy baja
            if loss.item() < 0.01:
                print("🎯 Loss muy bajo, finalizando entrenamiento temprano")
                break
        
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"📊 Época {epoch + 1} completada | Loss promedio: {avg_epoch_loss:.4f}")
    
    # Guardar modelo final
    final_path = os.path.join(output_dir, "final_model")
    save_model_hf(model, tokenizer, final_path, config, step)
    print(f"🏁 Modelo final guardado: {final_path}")
    
    print(f"\n✅ ¡Entrenamiento completado!")
    print(f"📊 Estadísticas finales:")
    print(f"   Steps totales: {step}")
    print(f"   Mejor val loss: {best_val_loss:.4f}")
    print(f"   Modelo final: {final_path}")

def main():
    parser = argparse.ArgumentParser(description="Entrenar HRM con tokenizador HuggingFace")
    parser.add_argument("--tokenizer", type=str, default="openai-community/gpt2",
                       help="Nombre del tokenizador HF")
    parser.add_argument("--output_dir", type=str, default="./hrm-micro-10m-hf",
                       help="Directorio de salida")
    parser.add_argument("--train_samples", type=int, default=10000,
                       help="Número de samples de entrenamiento")
    parser.add_argument("--val_samples", type=int, default=1000,
                       help="Número de samples de validación")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Tamaño del batch")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Número de épocas")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Frecuencia de guardado")
    parser.add_argument("--eval_steps", type=int, default=200,
                       help="Frecuencia de evaluación")
    
    if len(os.sys.argv) == 1:
        print("🚀 HRM Training con Tokenizador HuggingFace")
        print("\nUso:")
        print("  python hrm_training_micro_10m_hf.py [opciones]")
        print("\nEjemplos:")
        print("  # Entrenar con GPT2 inglés")
        print("  python hrm_training_micro_10m_hf.py --tokenizer openai-community/gpt2")
        print("  # Entrenar con GPT2 español")
        print("  python hrm_training_micro_10m_hf.py --tokenizer DeepESP/gpt2-spanish")
        print("  # Configuración personalizada")
        print("  python hrm_training_micro_10m_hf.py --tokenizer openai-community/gpt2 --batch_size 16 --epochs 5")
        return
    
    args = parser.parse_args()
    
    # Verificar dependencias
    if not HF_TOKENIZER_AVAILABLE:
        print("❌ Tokenizador HF no disponible. Instale las dependencias:")
        print("pip install transformers tokenizers datasets")
        return
    
    # Iniciar entrenamiento
    train_hrm_hf(
        tokenizer_name=args.tokenizer,
        output_dir=args.output_dir,
        num_train_samples=args.train_samples,
        num_val_samples=args.val_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )

if __name__ == "__main__":
    main()