#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat HRM para modelos HuggingFace
Interfaz de chat simple para modelos HRM entrenados con HuggingFace
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import json
from typing import Optional, List, Dict, Any
from transformers import AutoTokenizer

# Import HF tokenizer wrapper
try:
    from hf_tokenizer_wrapper_simple import HuggingFaceTokenizerWrapper
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ HF tokenizer wrapper no disponible")

class HRMConfig:
    """Configuración HRM compatible con HF"""
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 50257)
        self.block_size = kwargs.get('block_size', 128)  
        self.n_embd = kwargs.get('n_embd', 256)
        self.n_head = kwargs.get('n_head', 8)
        self.n_layers = kwargs.get('n_layers', 6)
        self.d_ff = kwargs.get('d_ff', 1024)
        self.dropout = kwargs.get('dropout', 0.1)
        self.halt_max_steps = kwargs.get('halt_max_steps', 4)
        self.ponder_loss_weight = kwargs.get('ponder_loss_weight', 0.01)
        self.halt_bias_init = kwargs.get('halt_bias_init', -0.5)
        self.use_rotary_embeddings = kwargs.get('use_rotary_embeddings', True)
        self.rotary_embedding_base = kwargs.get('rotary_embedding_base', 10000)
        self.use_flash_attention = kwargs.get('use_flash_attention', True)
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', False)
        self.h_update_period = kwargs.get('h_update_period', 2)

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding simplificado"""
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.register_buffer('freq_cis', self._compute_freq_cis(dim, base))
    
    def _compute_freq_cis(self, dim, base):
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        return freqs
    
    def forward(self, x, seq_len):
        device = x.device
        freqs = self.freq_cis.to(device)
        t = torch.arange(seq_len, device=device).type_as(freqs)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

def apply_rotary_emb(x, freqs_cis):
    """Aplicar rotary embedding simplificado"""
    # Para simplificar, por ahora retornamos x sin modificación
    # En una implementación completa, aquí iría la rotación compleja
    return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention con soporte para rotary embeddings"""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        self.use_rotary = config.use_rotary_embeddings
        if self.use_rotary:
            self.rotary = RotaryEmbedding(self.head_dim, config.rotary_embedding_base)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply rotary embeddings
        if self.use_rotary:
            freqs_cis = self.rotary(x, T)
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)
        
        # Attention
        att = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(y)

class HaltingUnit(nn.Module):
    """Unidad de parada para ACT"""
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.n_embd, 1)
        self.linear.bias.data.fill_(config.halt_bias_init)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class HRMBlock(nn.Module):
    """Bloque HRM con ACT"""
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.n_embd),
            nn.Dropout(config.dropout)
        )
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.halt_unit = HaltingUnit(config)
        self.max_steps = config.halt_max_steps
    
    def forward(self, x, mask=None):
        # ACT simplificado - usar solo 1 step para inferencia
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class HRMModelHF(nn.Module):
    """Modelo HRM compatible con HuggingFace"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([HRMBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Compartir pesos entre embeddings y output projection
        self.lm_head.weight = self.token_embeddings.weight
    
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(torch.arange(T, device=device))
        x = self.dropout(tok_emb + pos_emb)
        
        # Crear causal mask
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(T, T, device=device))
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm y output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

class ChatHRM:
    """Interfaz de chat para modelos HRM"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = self._get_device(device)
        self.model_path = model_path
        
        print(f"🤖 Cargando modelo HRM desde: {model_path}")
        print(f"💻 Dispositivo: {self.device}")
        
        # Cargar configuración
        self.config = self._load_config()
        
        # Cargar tokenizador
        self.tokenizer = self._load_tokenizer()
        
        # Cargar modelo
        self.model = self._load_model()
        
        print("✅ Modelo HRM cargado exitosamente")
        
    def _get_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device_str)
    
    def _load_config(self) -> HRMConfig:
        config_path = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return HRMConfig(**config_dict)
        else:
            print("⚠️ config.json no encontrado, usando configuración por defecto")
            return HRMConfig()
    
    def _load_tokenizer(self):
        # Primero intentar cargar desde config si especifica tokenizer
        config_path = os.path.join(self.model_path, "config.json")
        hf_tokenizer_name = None
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                hf_tokenizer_name = config_dict.get('hf_tokenizer_name')
            except:
                pass
        
        # Intentar cargar el tokenizador desde el directorio del modelo
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print(f"✅ Tokenizador HF cargado desde modelo: {tokenizer.__class__.__name__}")
            print(f"   📊 Vocabulario: {len(tokenizer):,} tokens")
            return tokenizer
        except Exception as e:
            print(f"⚠️ Error cargando tokenizador desde modelo: {e}")
            
            # Fallback al tokenizador especificado en config
            if hf_tokenizer_name:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
                    print(f"✅ Tokenizador HF cargado desde config: {hf_tokenizer_name}")
                    print(f"   📊 Vocabulario: {len(tokenizer):,} tokens")
                    return tokenizer
                except Exception as e2:
                    print(f"⚠️ Error cargando tokenizador desde config: {e2}")
            
            # Último fallback
            if HF_AVAILABLE:
                print("🔧 Usando wrapper HF por defecto...")
                return HuggingFaceTokenizerWrapper("openai-community/gpt2")
            else:
                raise RuntimeError("No se pudo cargar ningún tokenizador")
    
    def _load_model(self) -> HRMModelHF:
        model = HRMModelHF(self.config)
        
        # Cargar pesos del modelo
        model_path = os.path.join(self.model_path, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            print("✅ Pesos del modelo cargados")
        else:
            print("⚠️ No se encontraron pesos del modelo, usando inicialización aleatoria")
        
        model.to(self.device)
        model.eval()
        return model
    
    @torch.no_grad()
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.8, 
                top_k: int = 50, top_p: float = 0.9) -> str:
        """Generar texto usando el modelo HRM"""
        
        # Tokenizar prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=self.config.block_size)
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generar
        generated_ids = input_ids.clone()
        
        for _ in range(max_length):
            # Forward pass
            logits = self.model(generated_ids)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Check max length
            if generated_ids.size(1) >= self.config.block_size:
                break
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()
    
    def chat_loop(self):
        """Loop principal de chat interactivo"""
        print("\n🤖 Chat HRM - ¡Empezemos a conversar!")
        print("Comandos: /quit para salir, /clear para limpiar historial")
        print("=" * 50)
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\n👤 Usuario: ").strip()
                
                if user_input.lower() in ['/quit', '/exit', '/q']:
                    print("👋 ¡Hasta luego!")
                    break
                
                if user_input.lower() == '/clear':
                    conversation_history = []
                    print("🧹 Historial limpiado")
                    continue
                
                if not user_input:
                    continue
                
                # Crear prompt con contexto
                if conversation_history:
                    context = "\n".join(conversation_history[-6:])  # Últimas 3 interacciones
                    prompt = f"{context}\nUsuario: {user_input}\nAsistente:"
                else:
                    prompt = f"Usuario: {user_input}\nAsistente:"
                
                # Generar respuesta
                print("🤖 Asistente: ", end="", flush=True)
                response = self.generate(prompt, max_length=80, temperature=0.8)
                print(response)
                
                # Actualizar historial
                conversation_history.append(f"Usuario: {user_input}")
                conversation_history.append(f"Asistente: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description="Chat con modelo HRM-HuggingFace")
    parser.add_argument("--model_path", type=str, default="./hrm-micro-10m-hf/final_model",
                       help="Ruta al modelo entrenado")
    parser.add_argument("--device", type=str, default="auto",
                       help="Dispositivo (auto, cpu, cuda, mps)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Prompt único (sin modo interactivo)")
    parser.add_argument("--max_length", type=int, default=100,
                       help="Longitud máxima de generación")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Temperatura para sampling")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k filtering")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling")
    
    args = parser.parse_args()
    
    # Verificar que existe el modelo
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Modelo no encontrado en {args.model_path}")
        print("\n💡 Modelos disponibles:")
        for model_dir in ["hrm-micro-10m-hf", "hrm-small-50m-hf", "hrm-medium-100m-hf"]:
            if os.path.exists(model_dir):
                subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
                print(f"   📁 {model_dir}/: {', '.join(subdirs)}")
        sys.exit(1)
    
    try:
        # Crear chat
        chat = ChatHRM(args.model_path, args.device)
        
        if args.prompt:
            # Modo prompt único
            print(f"📝 Prompt: {args.prompt}")
            response = chat.generate(args.prompt, args.max_length, args.temperature, 
                                   args.top_k, args.top_p)
            print(f"🤖 Respuesta: {response}")
        else:
            # Modo interactivo
            chat.chat_loop()
            
    except Exception as e:
        print(f"❌ Error iniciando chat: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()