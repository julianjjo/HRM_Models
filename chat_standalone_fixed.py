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
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Optional, Tuple, Dict, Any, List, Union, Set

class HRMText1Config:
    """Configuración para el modelo HRM"""
    def __init__(self, **kwargs):
        # Configuración por defecto para modelo micro 10M
        self.vocab_size = kwargs.get('vocab_size', 5000)
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

class AdaptiveBPETokenizer:
    """Tokenizador BPE avanzado con vocabulario dinámico y optimizaciones"""
    
    def __init__(self, vocab_size=5000, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Tokens especiales mejorados
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<s>': 2,
            '</s>': 3,
            '<mask>': 4,
            '<cls>': 5,
            '<sep>': 6,
            '<newline>': 7,
            '<tab>': 8,
            '<url>': 9,
            '<email>': 10,
            '<number>': 11
        }
        
        # Mapeos de vocabulario
        self.vocab = {}  # token -> id
        self.inverse_vocab = {}  # id -> token
        self.byte_pairs = {}  # par -> merge_id
        self.merge_ranks = {}  # merge_id -> rank
        
        # Inicializar tokens especiales
        for token, token_id in self.special_tokens.items():
            self.vocab[token] = token_id
            self.inverse_vocab[token_id] = token
        
        # Aliases para compatibilidad
        self.word_to_id = self.vocab
        self.id_to_word = self.inverse_vocab
        
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.mask_token = '<mask>'
        self.cls_token = '<cls>'
        self.sep_token = '<sep>'
        
        self.pad_token_id = self.special_tokens['<pad>']
        self.unk_token_id = self.special_tokens['<unk>']
        self.bos_token_id = self.special_tokens['<s>']
        self.eos_token_id = self.special_tokens['</s>']
        self.mask_token_id = self.special_tokens['<mask>']
        self.cls_token_id = self.special_tokens['<cls>']
        self.sep_token_id = self.special_tokens['<sep>']
        
        # Cache y estadísticas
        self.encoding_cache = {}
        self.token_frequencies = Counter()
        self._built = False
        
        # Regex mejorados para preprocessing
        self.patterns = {
            'url': re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE),
            'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            'number': re.compile(r'\b\d+(?:\.\d+)?\b'),
            'whitespace': re.compile(r'\s+'),
            'punctuation': re.compile(r'[^\w\s]')
        }
        
        print(f"🔧 Inicializado AdaptiveBPETokenizer con vocabulario de {vocab_size:,} tokens")
    
    def _normalize_text(self, text: str) -> str:
        """Normalización avanzada de texto"""
        # Normalizar Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Reemplazar patrones especiales
        text = self.patterns['url'].sub('<url>', text)
        text = self.patterns['email'].sub('<email>', text)
        text = self.patterns['number'].sub('<number>', text)
        
        # Normalizar espacios en blanco
        text = text.replace('\n', '<newline>')
        text = text.replace('\t', '<tab>')
        text = self.patterns['whitespace'].sub(' ', text)
        
        return text.strip()
    
    def _get_byte_pairs(self, word: str) -> List[Tuple[str, str]]:
        """Obtener todos los pares de bytes consecutivos en una palabra"""
        pairs = []
        prev_char = word[0]
        for char in word[1:]:
            pairs.append((prev_char, char))
            prev_char = char
        return pairs
    
    def _get_word_tokens(self, text: str) -> List[str]:
        """Tokenización inicial por palabras con manejo de caracteres especiales"""
        # Pre-tokenizar manteniendo espacios y puntuación
        tokens = re.findall(r'\w+|[^\w\s]|\s+', text)
        # Filtrar tokens vacíos
        return [token for token in tokens if token.strip()]
    
    def _intelligent_word_filtering(self, word_freq: Counter) -> Dict[str, int]:
        """Filtrado inteligente con múltiples criterios de calidad"""
        filtered_words = {}
        
        # Listas de palabras a filtrar
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                    'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 
                    'por', 'son', 'con', 'para', 'como', 'las', 'los', 'del', 'una'}
        
        # Patrones problemáticos
        problematic_patterns = [
            re.compile(r'^[^a-zA-Z0-9]*$'),  # Solo símbolos
            re.compile(r'^.{1}$'),           # Caracteres únicos (ya manejados por char vocab)
            re.compile(r'^\\d+$'),            # Solo números (ya manejado por <number>)
            re.compile(r'^[A-Z]{4,}$'),      # Acrónimos muy largos
        ]
        
        total_words = sum(word_freq.values())
        
        for word, freq in word_freq.items():
            # Criterio 1: Frecuencia mínima
            if freq < self.min_frequency:
                continue
                
            # Criterio 2: Filtrar palabras muy cortas o muy largas
            if len(word) < 2 or len(word) > 50:
                continue
                
            # Criterio 3: Filtrar patrones problemáticos
            is_problematic = any(pattern.match(word) for pattern in problematic_patterns)
            if is_problematic:
                continue
                
            # Criterio 4: Priorizar palabras con buena ratio frecuencia/longitud
            freq_ratio = freq / total_words
            length_penalty = 1.0 / (1.0 + len(word) * 0.1)  # Penalizar palabras muy largas
            quality_score = freq_ratio * length_penalty
            
            # Criterio 5: Dar prioridad a palabras frecuentes
            if freq >= 100 or quality_score >= 1e-6:
                filtered_words[word] = freq
            elif word.lower() not in stopwords and freq >= self.min_frequency * 2:
                # Palabras menos frecuentes pero no stopwords
                filtered_words[word] = freq
        
        # Ordenar por frecuencia para priorizar tokens más útiles
        sorted_words = dict(sorted(filtered_words.items(), key=lambda x: x[1], reverse=True))
        
        # Limitar vocabulario si es muy grande
        max_words = self.vocab_size - len(self.special_tokens) - 1000  # Reservar espacio para caracteres
        if len(sorted_words) > max_words:
            sorted_words = dict(list(sorted_words.items())[:max_words])
        
        print(f"📊 Filtrado inteligente: {len(word_freq):,} -> {len(sorted_words):,} palabras")
        print(f"   Criterios aplicados: frecuencia ≥ {self.min_frequency}, longitud 2-50, patrones válidos")
        
        return sorted_words
    
    def build_vocab(self, texts: List[str], verbose: bool = True):
        """Construir vocabulario BPE desde corpus de textos"""
        if verbose:
            print(f"🔧 Construyendo vocabulario BPE desde {len(texts):,} textos...")
        
        # Fase 1: Recopilar caracteres únicos
        char_freq = Counter()
        word_freq = Counter()
        
        for i, text in enumerate(texts):
            if verbose and i % 10000 == 0:
                print(f"   Procesando texto {i+1:,}/{len(texts):,}")
            
            normalized_text = self._normalize_text(text)
            words = self._get_word_tokens(normalized_text)
            
            for word in words:
                word_freq[word] += 1
                for char in word:
                    char_freq[char] += 1
        
        # Filtrado inteligente de palabras con múltiples criterios
        filtered_words = self._intelligent_word_filtering(word_freq)
        
        if verbose:
            print(f"   Palabras únicas: {len(word_freq):,} -> {len(filtered_words):,} (freq >= {self.min_frequency})")
            print(f"   Caracteres únicos: {len(char_freq):,}")
        
        # Fase 2: Inicializar vocabulario con caracteres
        current_id = len(self.special_tokens)
        
        # Agregar caracteres más frecuentes primero
        for char, freq in char_freq.most_common():
            if char not in self.vocab:
                self.vocab[char] = current_id
                self.inverse_vocab[current_id] = char
                current_id += 1
        
        # Fase 3: Aplicar BPE iterativamente
        vocab_words = {word: list(word) for word, freq in filtered_words.items()}
        
        num_merges = self.vocab_size - len(self.vocab)
        if verbose:
            print(f"   Realizando {num_merges:,} merges de BPE...")
        
        for merge_step in range(num_merges):
            if verbose and merge_step % 1000 == 0 and merge_step > 0:
                print(f"   Merge {merge_step:,}/{num_merges:,}")
            
            # Contar pares de tokens adyacentes
            pair_counts = Counter()
            for word, word_tokens in vocab_words.items():
                word_pairs = self._get_pairs(word_tokens)
                word_frequency = filtered_words[word]
                for pair in word_pairs:
                    pair_counts[pair] += word_frequency
            
            if not pair_counts:
                break
                
            # Encontrar el par más frecuente
            best_pair = pair_counts.most_common(1)[0][0]
            
            # Crear nuevo token combinando el par
            new_token = best_pair[0] + best_pair[1]
            
            # Actualizar vocabulario
            if new_token not in self.vocab:
                self.vocab[new_token] = current_id
                self.inverse_vocab[current_id] = new_token
                current_id += 1
                
                # Guardar información del merge
                self.byte_pairs[best_pair] = merge_step
                self.merge_ranks[merge_step] = len(self.merge_ranks)
            
            # Actualizar representaciones de palabras
            vocab_words = self._merge_vocab(vocab_words, best_pair, new_token)
        
        # Actualizar estadísticas
        self.token_frequencies.update(filtered_words)
        self._built = True
        
        if verbose:
            print(f"✅ Vocabulario BPE construido:")
            print(f"   Total tokens: {len(self.vocab):,}")
            print(f"   Merges realizados: {len(self.byte_pairs):,}")
            print(f"   Cobertura estimada: {self._estimate_coverage(texts[:1000]):.1%}")
        
        return self
    
    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """Obtener pares de tokens adyacentes"""
        pairs = set()
        prev_token = word_tokens[0]
        for token in word_tokens[1:]:
            pairs.add((prev_token, token))
            prev_token = token
        return pairs
    
    def _merge_vocab(self, vocab_words: Dict[str, List[str]], pair: Tuple[str, str], new_token: str) -> Dict[str, List[str]]:
        """Aplicar merge a todo el vocabulario"""
        new_vocab_words = {}
        for word, word_tokens in vocab_words.items():
            new_word_tokens = []
            i = 0
            while i < len(word_tokens):
                if (i < len(word_tokens) - 1 and 
                    word_tokens[i] == pair[0] and 
                    word_tokens[i + 1] == pair[1]):
                    new_word_tokens.append(new_token)
                    i += 2
                else:
                    new_word_tokens.append(word_tokens[i])
                    i += 1
            new_vocab_words[word] = new_word_tokens
        return new_vocab_words
    
    def _estimate_coverage(self, texts: List[str]) -> float:
        """Estimar cobertura del vocabulario"""
        total_tokens = 0
        unknown_tokens = 0
        
        for text in texts:
            tokens = self._bpe_encode_word(self._normalize_text(text))
            total_tokens += len(tokens)
            unknown_tokens += tokens.count(self.unk_token_id)
        
        return 1.0 - (unknown_tokens / max(total_tokens, 1))
    
    def _bpe_encode_word(self, word: str) -> List[int]:
        """Codificar palabra usando BPE"""
        if not word:
            return []
        
        # Usar cache si disponible
        if word in self.encoding_cache:
            return self.encoding_cache[word]
        
        # Tokenizar palabra a nivel de caracteres
        word_tokens = list(word)
        
        # Aplicar merges en orden de ranking
        while len(word_tokens) > 1:
            pairs = self._get_pairs(word_tokens)
            if not pairs:
                break
                
            # Encontrar el mejor par para merger según el ranking
            best_pair = None
            best_rank = float('inf')
            
            for pair in pairs:
                if pair in self.byte_pairs:
                    rank = self.merge_ranks.get(self.byte_pairs[pair], float('inf'))
                    if rank < best_rank:
                        best_pair = pair
                        best_rank = rank
            
            if best_pair is None:
                break
                
            # Realizar merge
            new_token = best_pair[0] + best_pair[1]
            new_word_tokens = []
            i = 0
            while i < len(word_tokens):
                if (i < len(word_tokens) - 1 and 
                    word_tokens[i] == best_pair[0] and 
                    word_tokens[i + 1] == best_pair[1]):
                    new_word_tokens.append(new_token)
                    i += 2
                else:
                    new_word_tokens.append(word_tokens[i])
                    i += 1
            word_tokens = new_word_tokens
        
        # Convertir tokens a IDs
        token_ids = []
        for token in word_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Manejar tokens desconocidos a nivel de caracteres
                for char in token:
                    if char in self.vocab:
                        token_ids.append(self.vocab[char])
                    else:
                        token_ids.append(self.unk_token_id)
        
        # Guardar en cache
        self.encoding_cache[word] = token_ids
        return token_ids
    
    def encode(self, text: str, max_length: Optional[int] = None, 
               truncation: bool = True, padding: bool = False, 
               return_tensors: Optional[str] = None, 
               add_special_tokens: bool = True, 
               return_attention_mask: bool = True) -> Union[List[int], Dict]:
        """Codificar texto usando BPE"""
        if not self._built:
            print("⚠️ Vocabulario no construido. Usando vocabulario básico.")
            self.build_vocab([text])
        
        # Normalizar texto
        normalized_text = self._normalize_text(text)
        
        # Tokenizar por palabras y espacios
        words = self._get_word_tokens(normalized_text)
        
        # Codificar cada palabra con BPE
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for word in words:
            word_tokens = self._bpe_encode_word(word)
            token_ids.extend(word_tokens)
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        # Truncar si es necesario
        if max_length and truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.eos_token_id] if add_special_tokens else token_ids[:max_length]
        
        # Padding si es necesario
        if padding and max_length:
            if len(token_ids) < max_length:
                token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
        
        if return_tensors == "pt":
            result = {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
            if return_attention_mask:
                attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in token_ids]
                result["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)
            return result
        
        return token_ids
    
    def __call__(self, text: Union[str, List[str]], **kwargs) -> Dict:
        """Hacer el tokenizer callable como HuggingFace"""
        if isinstance(text, list):
            # Procesar batch de textos
            all_input_ids = []
            all_attention_masks = []
            
            for t in text:
                result = self.encode(t, **kwargs)
                if isinstance(result, dict):
                    all_input_ids.append(result["input_ids"][0].tolist() if hasattr(result["input_ids"], 'tolist') else result["input_ids"])
                    if "attention_mask" in result:
                        all_attention_masks.append(result["attention_mask"][0].tolist() if hasattr(result["attention_mask"], 'tolist') else result["attention_mask"])
                else:
                    all_input_ids.append(result)
            
            result_dict = {"input_ids": all_input_ids}
            if all_attention_masks:
                result_dict["attention_mask"] = all_attention_masks
            return result_dict
        else:
            # Procesar texto individual
            result = self.encode(text, **kwargs)
            if isinstance(result, dict):
                return result
            else:
                return {"input_ids": result}
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], 
               skip_special_tokens: bool = True, 
               clean_up_tokenization_spaces: bool = True) -> str:
        """Decodificar tokens a texto"""
        if hasattr(token_ids, 'tolist'):  # Es un tensor
            if len(token_ids.shape) > 1:
                token_ids = token_ids.squeeze()
            token_ids = token_ids.tolist()
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        # Reconstruir texto
        text = "".join(tokens)
        
        if clean_up_tokenization_spaces:
            # Restaurar espacios normalizados
            text = text.replace('<newline>', '\n')
            text = text.replace('<tab>', '\t')
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text
    
    def __len__(self) -> int:
        return len(self.vocab)
    
    def get_vocab_size(self) -> int:
        """Obtener tamaño actual del vocabulario"""
        return len(self.vocab)
    
    def add_tokens(self, new_tokens: List[str], special: bool = False) -> int:
        """Agregar nuevos tokens dinámicamente al vocabulario"""
        added = 0
        current_id = len(self.vocab)
        
        for token in new_tokens:
            if token not in self.vocab:
                self.vocab[token] = current_id
                self.inverse_vocab[current_id] = token
                
                if special:
                    self.special_tokens[token] = current_id
                
                current_id += 1
                added += 1
        
        if added > 0:
            print(f"➕ Agregados {added} nuevos tokens al vocabulario (total: {len(self.vocab):,})")
        
        return added
    
    def save_pretrained(self, save_directory: str):
        """Guardar tokenizer con todas las mejoras"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Guardar vocabulario principal
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'inverse_vocab': {str(k): v for k, v in self.inverse_vocab.items()},
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens,
                'min_frequency': self.min_frequency
            }, f, indent=2, ensure_ascii=False)
        
        # Guardar merges de BPE
        merges_file = os.path.join(save_directory, "merges.txt")
        with open(merges_file, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
            # Ordenar merges por ranking
            sorted_merges = sorted(self.byte_pairs.items(), key=lambda x: self.merge_ranks.get(x[1], 0))
            for (token1, token2), merge_id in sorted_merges:
                f.write(f"{token1} {token2}\n")
        
        # Guardar estadísticas
        stats_file = os.path.join(save_directory, "tokenizer_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_tokens': len(self.vocab),
                'bpe_merges': len(self.byte_pairs),
                'cache_size': len(self.encoding_cache),
                'token_frequencies': dict(self.token_frequencies.most_common(1000))
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 AdaptiveBPETokenizer guardado en: {save_directory}")
        print(f"   📊 Estadísticas: {len(self.vocab):,} tokens, {len(self.byte_pairs):,} merges")
    
    def load_pretrained(self, load_directory: str):
        """Cargar tokenizer guardado"""
        # Cargar vocabulario
        vocab_file = os.path.join(load_directory, "vocab.json")
        if os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.vocab = data['vocab']
                self.inverse_vocab = {int(k): v for k, v in data['inverse_vocab'].items()}
                self.vocab_size = data['vocab_size']
                self.special_tokens = data['special_tokens']
                self.min_frequency = data.get('min_frequency', 2)
        
        # Cargar merges
        merges_file = os.path.join(load_directory, "merges.txt")
        if os.path.exists(merges_file):
            with open(merges_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]  # Skip version line
                for i, line in enumerate(lines):
                    tokens = line.strip().split()
                    if len(tokens) == 2:
                        self.byte_pairs[(tokens[0], tokens[1])] = i
                        self.merge_ranks[i] = i
        
        # Actualizar aliases
        self.word_to_id = self.vocab
        self.id_to_word = self.inverse_vocab
        self._built = True
        
        print(f"📁 AdaptiveBPETokenizer cargado desde: {load_directory}")
        print(f"   📊 Estadísticas: {len(self.vocab):,} tokens, {len(self.byte_pairs):,} merges")
        
        return self

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
    tokenizer = AdaptiveBPETokenizer()
    
    if os.path.exists(model_path):
        try:
            tokenizer.load_pretrained(model_path)
        except Exception as e:
            print(f"⚠️ Error cargando tokenizer: {e}")
            print("💡 Usando tokenizer básico")
            tokenizer._built = True
    else:
        print("⚠️ No se encontró directorio del modelo, usando tokenizer básico")
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

# Alias para compatibilidad
SimpleTokenizer = AdaptiveBPETokenizer

if __name__ == "__main__":
    main()