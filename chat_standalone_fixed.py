#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat Standalone Completamente Independiente para HRM-Models
No depende de importar el script de entrenamiento completo
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import sys
import argparse
import math
from typing import Optional, Tuple, Dict, Any

class HRMText1Config:
    """Configuración para el modelo HRM"""
    def __init__(self, **kwargs):
        # Configuración por defecto para modelo micro 10M
        self.vocab_size = kwargs.get('vocab_size', 205)
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
        self.use_flash_attention = kwargs.get('use_flash_attention', False)
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', False)
        self.h_update_period = kwargs.get('h_update_period', 2)

class SimpleTokenizer:
    """Tokenizer simple standalone"""
    
    def __init__(self):
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3,
            '<mask>': 4
        }
        
        self.word_to_id = self.special_tokens.copy()
        self.id_to_word = {v: k for k, v in self.special_tokens.items()}
        
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.mask_token = '<mask>'
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.mask_token_id = 4
        
        self.vocab_size = 205
        self._built = False
    
    def load_from_file(self, vocab_path: str):
        """Cargar vocabulario desde archivo JSON"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word_to_id = vocab_data['word_to_id']
        self.id_to_word = {int(k): v for k, v in vocab_data.get('id_to_word', {}).items()}
        self.vocab_size = vocab_data.get('vocab_size', len(self.word_to_id))
        
        if 'special_tokens' in vocab_data:
            self.special_tokens = vocab_data['special_tokens']
        
        # Asegurar tokens especiales
        self.pad_token_id = self.special_tokens.get('<pad>', 0)
        self.unk_token_id = self.special_tokens.get('<unk>', 1)
        self.bos_token_id = self.special_tokens.get('<s>', 2)
        self.eos_token_id = self.special_tokens.get('</s>', 3)
        self.mask_token_id = self.special_tokens.get('<mask>', 4)
        
        self._built = True
        print(f"✅ Vocabulario cargado: {self.vocab_size} tokens")
    
    def encode(self, text: str) -> list:
        """Tokenizar texto simple"""
        if not self._built:
            # Vocabulario mínimo si no está construido
            words = text.lower().split()
            return [self.unk_token_id] * min(len(words), 10)
        
        words = text.lower().split()
        tokens = []
        for word in words:
            # Limpiar puntuación básica
            cleaned_word = word.strip('.,!?;:"()[]{}')
            if cleaned_word in self.word_to_id:
                tokens.append(self.word_to_id[cleaned_word])
            else:
                tokens.append(self.unk_token_id)
        return tokens
    
    def decode(self, tokens: list) -> str:
        """Decodificar tokens a texto"""
        words = []
        for token in tokens:
            if isinstance(token, torch.Tensor):
                token = token.item()
            
            if token in self.id_to_word:
                word = self.id_to_word[token]
                # Filtrar tokens especiales
                if not word.startswith('<') or word in ['<s>', '</s>']:
                    if word not in ['<s>', '</s>', '<pad>', '<mask>']:
                        words.append(word)
        return ' '.join(words)
    
    def __len__(self):
        return self.vocab_size

# Versión simplificada del modelo para chat
class SimpleChatModel(nn.Module):
    """Modelo simplificado para chat que puede cargar pesos del HRM"""
    
    def __init__(self, config: HRMText1Config):
        super().__init__()
        self.config = config
        
        # Componentes básicos
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_encoding = nn.Embedding(config.block_size, config.n_embd)
        
        # Capas del transformer
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.n_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # Inicialización
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeddings = self.embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.pos_encoding(position_ids)
        
        x = token_embeddings + position_embeddings
        
        # Transformer layers (simplified)
        for layer in self.layers:
            # Para decoder, necesitamos máscara causal
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            x = layer(x, x, tgt_mask=causal_mask.to(input_ids.device))
        
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.9):
        """Generación simple de texto"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Tomar solo los últimos tokens si excede block_size
                current_input = input_ids[:, -self.config.block_size:]
                
                # Forward pass
                logits = self.forward(current_input)
                logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float('-inf')
                
                # Samplear
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Concatenar
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Parar si token de fin
                if next_token.item() == 3:  # </s>
                    break
        
        return input_ids

def load_model_and_tokenizer(model_path: str):
    """Cargar modelo y tokenizer desde directorio"""
    
    # Cargar tokenizer
    tokenizer = SimpleTokenizer()
    vocab_path = os.path.join(model_path, "vocab.json")
    
    if os.path.exists(vocab_path):
        tokenizer.load_from_file(vocab_path)
    else:
        print("⚠️ No se encontró vocab.json, usando tokenizer básico")
        tokenizer._built = True
    
    # Cargar configuración
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = HRMText1Config(**config_dict)
    else:
        print("⚠️ Usando configuración por defecto")
        config = HRMText1Config()
    
    # Crear modelo
    model = SimpleChatModel(config)
    
    # Intentar cargar pesos
    checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(model_path, "best_model.bin")
    
    if os.path.exists(checkpoint_path):
        try:
            print(f"🔄 Cargando pesos desde: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Si el state_dict viene de un checkpoint, extraer model_state_dict
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # Intentar cargar pesos compatibles
            model_state = model.state_dict()
            loaded_keys = []
            
            for key in model_state.keys():
                if key in state_dict and model_state[key].shape == state_dict[key].shape:
                    model_state[key] = state_dict[key]
                    loaded_keys.append(key)
            
            model.load_state_dict(model_state, strict=False)
            print(f"✅ Cargados {len(loaded_keys)}/{len(model_state)} tensores")
            
        except Exception as e:
            print(f"⚠️ Error cargando pesos: {e}")
            print("💡 Usando modelo con pesos aleatorios")
    else:
        print("⚠️ No se encontraron pesos, usando modelo aleatorio")
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Chat con HRM-Models Standalone")
    parser.add_argument("--model", type=str, required=True, help="Directorio del modelo")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperatura de generación")
    parser.add_argument("--max_tokens", type=int, default=50, help="Tokens máximos a generar")
    
    if len(sys.argv) == 1:
        print("Uso:")
        print("  python chat_standalone_fixed.py --model /path/to/model")
        return
    
    args = parser.parse_args()
    
    print("🚀 HRM Chat Standalone Mejorado")
    print("=" * 50)
    
    # Cargar modelo y tokenizer
    model_path = os.path.expanduser(args.model)
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    print(f"📊 Modelo: {sum(p.numel() for p in model.parameters()):,} parámetros")
    print(f"🔤 Vocabulario: {len(tokenizer)} tokens")
    print(f"🌡️ Temperatura: {args.temperature}")
    print(f"📝 Max tokens: {args.max_tokens}")
    
    print("\\n💬 ¡Chat iniciado! (escribe 'quit' para salir)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\\n👤 Tú: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'salir']:
                print("👋 ¡Hasta luego!")
                break
            
            if not user_input:
                continue
            
            # Tokenizar entrada
            input_tokens = tokenizer.encode(user_input)
            if not input_tokens:
                print("⚠️ No se pudo tokenizar la entrada")
                continue
            
            input_tensor = torch.tensor([input_tokens], dtype=torch.long)
            print(f"🔢 Tokens: {len(input_tokens)}")
            
            # Generar respuesta
            print("🤖 HRM:", end=" ", flush=True)
            
            with torch.no_grad():
                output = model.generate(
                    input_tensor,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature
                )
                
                # Extraer solo los tokens nuevos
                new_tokens = output[0, len(input_tokens):].tolist()
                response = tokenizer.decode(new_tokens)
                
                print(response)
        
        except KeyboardInterrupt:
            print("\\n👋 Chat interrumpido. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\\n❌ Error: {e}")

if __name__ == "__main__":
    main()