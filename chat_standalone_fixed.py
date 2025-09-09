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

# Import HF tokenizer wrapper
try:
    from hf_tokenizer_wrapper_simple import HuggingFaceTokenizerWrapper, create_tokenizer
    HF_TOKENIZER_AVAILABLE = True
    print("✅ HuggingFace tokenizer wrapper disponible")
except ImportError:
    HF_TOKENIZER_AVAILABLE = False
    print("⚠️ HuggingFace tokenizer wrapper no disponible, usando tokenizador legacy")

class HRMText1Config:
    """Configuración para el modelo HRM"""
    def __init__(self, **kwargs):
        # Configuración por defecto para modelo micro 10M
        # Actualizado para soportar tokenizadores de HF
        self.vocab_size = kwargs.get('vocab_size', 50257)  # GPT2 default
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
        
        # Configuración específica del tokenizador
        self.tokenizer_type = kwargs.get('tokenizer_type', 'huggingface')  # 'huggingface' o 'legacy'
        self.hf_tokenizer_name = kwargs.get('hf_tokenizer_name', 'openai-community/gpt2')
        self.pad_token_id = kwargs.get('pad_token_id', 0)

class AdaptiveBPETokenizer:
    """Tokenizador BPE avanzado con vocabulario dinámico y optimizaciones"""
    
    def __init__(self, vocab_size=5000, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Tokens especiales optimizados para modelo 10M (solo tokens existentes)
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<s>': 2,
            '</s>': 3,
            '<mask>': 4,
            '<cls>': 5,
            '<sep>': 6,
            '<newline>': 7,      # ✅ Disponible en modelo
            '<tab>': 8,          # ✅ Disponible en modelo
            '<space>': 9,        # ✅ Disponible - Para exactamente 2 espacios
            '<spaces>': 10,      # ✅ Disponible - Para 3+ espacios
            '<url>': 11,
            '<email>': 12,
            '<number>': 13
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
        self.space_token = '<space>'      # Para 2 espacios
        self.spaces_token = '<spaces>'    # Para 3+ espacios
        self.newline_token = '<newline>'  # Para \n
        self.tab_token = '<tab>'          # Para \t
        
        self.pad_token_id = self.special_tokens['<pad>']
        self.unk_token_id = self.special_tokens['<unk>']
        self.bos_token_id = self.special_tokens['<s>']
        self.eos_token_id = self.special_tokens['</s>']
        self.mask_token_id = self.special_tokens['<mask>']
        self.cls_token_id = self.special_tokens['<cls>']
        self.sep_token_id = self.special_tokens['<sep>']
        self.newline_token_id = self.special_tokens['<newline>']
        self.tab_token_id = self.special_tokens['<tab>']
        self.space_token_id = self.special_tokens['<space>']
        self.spaces_token_id = self.special_tokens['<spaces>']
        self.url_token_id = self.special_tokens['<url>']
        self.email_token_id = self.special_tokens['<email>']
        self.number_token_id = self.special_tokens['<number>']
        
        # Cache y estadísticas
        self.encoding_cache = {}
        self.token_frequencies = Counter()
        self._built = False
        
        # Regex mejorados para preprocessing con mejor manejo de espacios
        self.patterns = {
            'url': re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE),
            'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            'number': re.compile(r'\b\d+(?:\.\d+)?\b'),
            'multiple_spaces': re.compile(r'  +'),  # Para espacios múltiples
            'whitespace': re.compile(r'\s+'),
            'punctuation': re.compile(r'[^\w\s]')
        }
        
        print(f"🔧 Inicializado AdaptiveBPETokenizer con vocabulario de {vocab_size:,} tokens")
    
    def _normalize_text(self, text: str) -> str:
        """Normalización avanzada de texto con mejor preservación de espacios en blanco"""
        if not text:
            return ""
            
        # Normalizar Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Reemplazar patrones especiales ANTES de procesar espacios
        text = self.patterns['url'].sub('<url>', text)
        text = self.patterns['email'].sub('<email>', text) 
        text = self.patterns['number'].sub('<number>', text)
        
        return self._advanced_space_processing(text)
    
    def _advanced_space_processing(self, text: str) -> str:
        """Procesamiento optimizado de espacios para modelo 10M (usando solo tokens disponibles)"""
        if not text:
            return ""
        
        # Procesar newlines y tabs primero
        text = text.replace('\n', '<newline>')
        text = text.replace('\t', '<tab>')
        
        # Procesar espacios múltiples (optimizado para tokens disponibles)
        # 4+ espacios consecutivos -> <spaces> (se normaliza a 3)
        text = re.sub(r'    +', '<spaces>', text)
        # Exactamente 3 espacios -> <spaces>
        text = text.replace('   ', '<spaces>')
        # Exactamente 2 espacios -> <space>
        text = text.replace('  ', '<space>')
        # Un solo espacio permanece como ' '
        
        # Manejar líneas vacías (simular <blank> con <newline>)
        # Reemplazar secuencias de newlines múltiples
        text = re.sub(r'<newline><newline><newline>+', '<newline><newline>', text)
        
        # Procesar indentación al inicio (usando tokens disponibles)
        lines = text.split('<newline>')
        processed_lines = []
        
        for line in lines:
            # Detectar indentación al inicio
            if line.startswith('<spaces>'):
                # 3+ espacios al inicio -> mantener <spaces> (simula indentación)
                processed_lines.append(line)
            elif line.startswith('<space>'):
                # 2 espacios al inicio -> mantener <space>
                processed_lines.append(line)
            else:
                processed_lines.append(line)
        
        result = '<newline>'.join(processed_lines)
        
        # Limpiar espacios extremos solo si no son significativos
        original_starts_with_space = text.startswith((' ', '\n', '\t'))
        original_ends_with_space = text.endswith((' ', '\n', '\t'))
        
        if not (original_starts_with_space or original_ends_with_space):
            result = result.strip()
        
        return result
    
    def _get_byte_pairs(self, word: str) -> List[Tuple[str, str]]:
        """Obtener todos los pares de bytes consecutivos en una palabra"""
        pairs = []
        prev_char = word[0]
        for char in word[1:]:
            pairs.append((prev_char, char))
            prev_char = char
        return pairs
    
    def _get_word_tokens(self, text: str) -> List[str]:
        """Tokenización inicial por palabras con mejor preservación de espacios"""
        if not text:
            return []
            
        # Pre-tokenizar manteniendo espacios, puntuación y tokens especiales
        # Separar tokens especiales (solo tokens disponibles en modelo 10M)
        special_pattern = r'(<newline>|<tab>|<space>|<spaces>|<url>|<email>|<number>|<pad>|<unk>|<s>|</s>|<mask>|<cls>|<sep>)'
        tokens = []
        
        # Dividir por tokens especiales
        parts = re.split(special_pattern, text)
        
        for part in parts:
            if not part:
                continue
            elif part in self.special_tokens:
                # Es un token especial, mantenerlo como está
                tokens.append(part)
            else:
                # Tokenizar parte normal manteniendo espacios simples
                part_tokens = re.findall(r'\w+|[^\w\s]| ', part)
                tokens.extend([t for t in part_tokens if t])  # Mantener espacios simples
        
        return tokens
    
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
            print("⚠️ Vocabulario no construido. Usando tokenización básica.")
            # Fallback simple tokenization
            tokens = []
            if add_special_tokens:
                tokens.append(self.bos_token_id)
            
            # Tokenización básica por caracteres
            for char in text:
                if char in self.vocab:
                    tokens.append(self.vocab[char])
                else:
                    tokens.append(self.unk_token_id)
            
            if add_special_tokens:
                tokens.append(self.eos_token_id)
            
            return tokens
        
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
            # MEJORADO: Restaurar espacios en blanco de manera más precisa
            text = self._restore_blank_spaces(text)
        
        return text
    
    def _restore_blank_spaces(self, text: str) -> str:
        """Restaurar espacios desde tokens especiales (optimizado para modelo 10M)"""
        if not text:
            return text
        
        # Restaurar en orden específico para evitar conflictos
        
        # 1. Restaurar newlines
        text = text.replace('<newline>', '\n')
        
        # 2. Restaurar tabs
        text = text.replace('<tab>', '\t')
        
        # 3. Restaurar espacios múltiples (disponibles en modelo 10M)
        text = text.replace('<spaces>', '   ')  # 3 espacios (normalización)
        
        # 4. Restaurar espacios dobles
        text = text.replace('<space>', '  ')    # 2 espacios
        
        # 5. Limpiar múltiples newlines consecutivos (máximo 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 6. Solo hacer strip si no hay espacios/newlines significativos
        if not (text.startswith((' ', '\n', '\t')) or text.endswith((' ', '\n', '\t'))):
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

# ==============================================================================
# --- ARQUITECTURA HRM PARA CHAT ---
# ==============================================================================

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("⚠️ Flash Attention no disponible, usando atención estándar")

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding optimizado para multi-GPU y mejor extrapolación"""
    def __init__(self, dim, max_position_embeddings=4096, base=10000):
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

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
    
    def forward(self, x):
        # Fix para estabilidad numérica y compatibilidad multi-GPU
        device = x.device
        
        # Calcular RMS norm con tensors en el device correcto
        mean_square = torch.mean(x**2, dim=-1, keepdim=True)
        eps_tensor = torch.tensor(self.eps, device=device, dtype=x.dtype)
        rms = torch.rsqrt(mean_square + eps_tensor)
        
        # Asegurar que weight esté en el device correcto
        weight = self.weight.to(device)
        
        return weight * (x * rms)

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
    """Simplified HRM implementation for chat"""
    def __init__(self, config):
        super().__init__()
        self.block = HRMBlock(config)
        self.config = config
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """Simplified forward pass for chat inference"""
        return self.block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

class HRMText1(nn.Module):
    """HRM Model for Chat - Simplified from training version"""
    
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
    
    def forward(self, input_ids, attention_mask=None):
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
        
        # HRM layers
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, key_padding_mask=attention_mask)
        
        # Final norm y lm_head
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.9):
        """Generación de texto con HRM"""
        self.eval()
        
        with torch.no_grad():
            generated_tokens = []
            
            for step in range(max_new_tokens):
                # Tomar solo los últimos tokens si excede block_size
                current_input = input_ids[:, -self.config.block_size:]
                
                # Forward pass
                logits = self.forward(current_input)
                logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    values, indices = torch.topk(logits, top_k)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(-1, indices, values)
                    logits = logits_filtered
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create a mask for indices to remove
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                    indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Samplear
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Concatenar
                input_ids = torch.cat([input_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())
                
                # Parar si token de fin, pero solo después de generar al menos algunos tokens
                if next_token.item() == 3 and step > 2:  # </s>, allow some generation first
                    break
                
                # También parar si se encuentra un token de parada natural
                if next_token.item() in [0, 1]:  # <pad> o <unk> también pueden indicar fin
                    if step > 5:  # Solo si ya hemos generado algo
                        break
        
        return input_ids

def load_model_and_tokenizer(model_path: str, use_hf_tokenizer: bool = None, hf_model_name: str = None):
    """
    Cargar modelo y tokenizer desde directorio
    
    Args:
        model_path: Ruta al modelo guardado
        use_hf_tokenizer: Si usar tokenizador HF (None = auto-detectar)
        hf_model_name: Nombre del modelo HF a usar (None = usar de config)
    """
    
    # Cargar configuración primero
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = HRMText1Config(**config_dict)
    else:
        print("⚠️ Usando configuración por defecto")
        config = HRMText1Config()
    
    # Determinar qué tokenizador usar
    if use_hf_tokenizer is None:
        use_hf_tokenizer = (
            HF_TOKENIZER_AVAILABLE and 
            getattr(config, 'tokenizer_type', 'huggingface') == 'huggingface'
        )
    
    # Cargar tokenizer
    if use_hf_tokenizer and HF_TOKENIZER_AVAILABLE:
        try:
            tokenizer_name = hf_model_name or getattr(config, 'hf_tokenizer_name', 'openai-community/gpt2')
            print(f"🔧 Cargando tokenizador HuggingFace: {tokenizer_name}")
            tokenizer = create_tokenizer(tokenizer_name)
            
            # Actualizar config con el tamaño real del vocabulario
            config.vocab_size = len(tokenizer)
            config.pad_token_id = tokenizer.pad_token_id
            
            print(f"✅ Tokenizador HF cargado: {len(tokenizer):,} tokens")
            
        except Exception as e:
            print(f"⚠️ Error cargando tokenizador HF: {e}")
            print("🔄 Fallback a tokenizador legacy...")
            use_hf_tokenizer = False
    
    if not use_hf_tokenizer:
        # Usar tokenizador legacy
        tokenizer = AdaptiveBPETokenizer()
        
        if os.path.exists(model_path):
            try:
                tokenizer.load_pretrained(model_path)
            except Exception as e:
                print(f"⚠️ Error cargando tokenizer legacy: {e}")
                print("💡 Usando tokenizer básico")
                tokenizer._built = True
        else:
            print("⚠️ No se encontró directorio del modelo, usando tokenizer básico")
            tokenizer._built = True
        
        # Ajustar configuración para tokenizador legacy
        config.vocab_size = len(tokenizer)
    
    # Crear modelo
    model = HRMText1(config)
    
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
            
            # Intentar cargar pesos compatibles con mapeo de claves
            model_state = model.state_dict()
            loaded_keys = []
            vocab_size_mismatch = False
            
            # Crear mapeo de claves del modelo original al chat
            key_mapping = {}
            
            for model_key in model_state.keys():
                # Verificar compatibilidad de tamaño de vocabulario
                if 'token_embeddings.weight' in model_key or 'lm_head.weight' in model_key:
                    if model_key in state_dict:
                        if model_state[model_key].shape != state_dict[model_key].shape:
                            print(f"⚠️ Incompatibilidad en vocabulario: {model_state[model_key].shape} vs {state_dict[model_key].shape}")
                            vocab_size_mismatch = True
                            continue
                
                # Intentar mapeos comunes
                mapped_keys = [
                    model_key,  # Exact match first
                    model_key.replace('block.', 'H_module.'),  # Training to chat mapping
                    model_key.replace('H_module.', 'block.'),  # Chat to training mapping
                ]
                
                for mapped_key in mapped_keys:
                    if mapped_key in state_dict and model_state[model_key].shape == state_dict[mapped_key].shape:
                        key_mapping[model_key] = mapped_key
                        break
            
            # Cargar los pesos usando el mapeo
            for model_key in model_state.keys():
                if model_key in key_mapping:
                    checkpoint_key = key_mapping[model_key]
                    model_state[model_key] = state_dict[checkpoint_key]
                    loaded_keys.append(model_key)
            
            model.load_state_dict(model_state, strict=False)
            
            if vocab_size_mismatch:
                print(f"⚠️ Cargados {len(loaded_keys)}/{len(model_state)} tensores (embeddings omitidos por incompatibilidad)")
                print("💡 El modelo necesita entrenamiento desde cero para el nuevo vocabulario")
            else:
                print(f"✅ Cargados {len(loaded_keys)}/{len(model_state)} tensores")
            
        except Exception as e:
            print(f"⚠️ Error cargando pesos: {e}")
            print("💡 Usando modelo con pesos aleatorios")
    else:
        print("⚠️ No se encontraron pesos, usando modelo aleatorio")
        print("💡 Perfecto para entrenamiento desde cero")
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Chat con HRM-Models Standalone")
    parser.add_argument("--model", type=str, required=True, help="Directorio del modelo")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperatura de generación")
    parser.add_argument("--max_tokens", type=int, default=50, help="Tokens máximos a generar")
    parser.add_argument("--tokenizer", type=str, default="auto", 
                       choices=["auto", "hf", "legacy"], 
                       help="Tipo de tokenizador: auto, hf (Hugging Face), legacy")
    parser.add_argument("--hf_model", type=str, default=None,
                       help="Modelo de HF para el tokenizador (ej: openai-community/gpt2, DeepESP/gpt2-spanish)")
    
    if len(sys.argv) == 1:
        print("Uso:")
        print("  python chat_standalone_fixed.py --model /path/to/model [--tokenizer hf] [--hf_model openai-community/gpt2]")
        print("\nEjemplos:")
        print("  # Usar tokenizador HF automático")
        print("  python chat_standalone_fixed.py --model ./model --tokenizer hf")
        print("  # Usar GPT2 específico")  
        print("  python chat_standalone_fixed.py --model ./model --tokenizer hf --hf_model openai-community/gpt2")
        print("  # Usar GPT2 español")
        print("  python chat_standalone_fixed.py --model ./model --tokenizer hf --hf_model DeepESP/gpt2-spanish")
        return
    
    args = parser.parse_args()
    
    print("🚀 HRM Chat Standalone con Tokenizador HuggingFace")
    print("=" * 60)
    
    # Determinar configuración del tokenizador
    use_hf = None
    if args.tokenizer == "hf":
        use_hf = True
    elif args.tokenizer == "legacy":
        use_hf = False
    # Si es "auto", dejar que load_model_and_tokenizer decida
    
    # Cargar modelo y tokenizer
    model_path = os.path.expanduser(args.model)
    model, tokenizer = load_model_and_tokenizer(
        model_path, 
        use_hf_tokenizer=use_hf,
        hf_model_name=args.hf_model
    )
    
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
            
            # Agregar BOS token si no está presente
            if input_tokens[0] != tokenizer.bos_token_id:
                input_tokens = [tokenizer.bos_token_id] + input_tokens
            
            input_tensor = torch.tensor([input_tokens], dtype=torch.long)
            print(f"🔢 Tokens: {len(input_tokens)}")
            
            # Generar respuesta
            print("🤖 HRM:", end=" ", flush=True)
            
            with torch.no_grad():
                output = model.generate(
                    input_tensor,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=50,
                    top_p=0.9
                )
                
                # Extraer solo los tokens nuevos
                new_tokens = output[0, len(input_tokens):].tolist()
                
                if new_tokens:
                    response = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    print(response if response.strip() else "[Sin respuesta generada]")
                else:
                    print("[No se generaron tokens nuevos]")
        
        except KeyboardInterrupt:
            print("\\n👋 Chat interrumpido. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\\n❌ Error: {e}")

# Alias para compatibilidad
SimpleTokenizer = AdaptiveBPETokenizer

if __name__ == "__main__":
    main()