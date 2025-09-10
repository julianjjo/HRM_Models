#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper Simple de Tokenizador HuggingFace para HRM
Solo HuggingFace tokenizers, sin AdaptiveBPE fallback
GPT-2 maneja espacios nativamente, sin procesamiento adicional
"""

import torch
from typing import Optional, Union, List, Dict
from transformers import AutoTokenizer, GPT2TokenizerFast

class HuggingFaceTokenizerWrapper:
    """
    Wrapper simplificado que usa solo tokenizadores de Hugging Face
    """
    
    def __init__(self, model_name: str = "openai-community/gpt2"):
        """
        Inicializa el wrapper con un tokenizador de HF
        
        Args:
            model_name: Nombre del modelo/tokenizador de HF a usar
        """
        self.model_name = model_name
        
        # Cargar tokenizador de HF
        # Solo imprimir en proceso principal para evitar spam en multiprocessing
        import os
        if os.getpid() == getattr(os, '_main_pid', os.getpid()):
            print(f"🔧 Cargando tokenizador HuggingFace: {model_name}")
        try:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=False
            )
        except Exception as e:
            print(f"⚠️ Error cargando {model_name}, usando GPT2 por defecto: {e}")
            self.hf_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        # Configurar pad token si no existe
        if self.hf_tokenizer.pad_token is None:
            if self.hf_tokenizer.eos_token is not None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            else:
                self.hf_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        # Crear mapeos de compatibilidad
        self._create_compatibility_mappings()
        
        # Solo imprimir en proceso principal
        if os.getpid() == getattr(os, '_main_pid', os.getpid()):
            print(f"✅ Tokenizador inicializado:")
            print(f"   📊 Vocabulario: {len(self.hf_tokenizer):,} tokens")
            print(f"   🔤 Modelo: {self.model_name}")
            print(f"   🔧 Versión rápida: {hasattr(self.hf_tokenizer, 'is_fast') and self.hf_tokenizer.is_fast}")
    
    def _create_compatibility_mappings(self):
        """Crear mapeos para compatibilidad con interfaz anterior"""
        # Aliases para compatibilidad
        self.vocab = self.hf_tokenizer.vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.word_to_id = self.vocab
        self.id_to_word = self.inverse_vocab
        
        # Tokens especiales
        self.pad_token = self.hf_tokenizer.pad_token
        self.unk_token = self.hf_tokenizer.unk_token
        self.bos_token = self.hf_tokenizer.bos_token if hasattr(self.hf_tokenizer, 'bos_token') else None
        self.eos_token = self.hf_tokenizer.eos_token
        
        # IDs de tokens especiales
        self.pad_token_id = self.hf_tokenizer.pad_token_id
        self.unk_token_id = self.hf_tokenizer.unk_token_id
        self.bos_token_id = getattr(self.hf_tokenizer, 'bos_token_id', None)
        self.eos_token_id = self.hf_tokenizer.eos_token_id
        
        # Para GPT2, usar eos como bos si no existe bos
        if self.bos_token_id is None and hasattr(self.hf_tokenizer, 'eos_token_id'):
            self.bos_token_id = self.eos_token_id
            self.bos_token = self.eos_token
    
    def encode(self, 
               text: str, 
               max_length: Optional[int] = None,
               truncation: bool = True,
               padding: bool = False,
               return_tensors: Optional[str] = None,
               add_special_tokens: bool = True,
               return_attention_mask: bool = True) -> Union[List[int], Dict]:
        """
        Codificar texto usando el tokenizador de HF
        """
        if not text:
            if return_tensors == "pt":
                result = {"input_ids": torch.tensor([[]], dtype=torch.long)}
                if return_attention_mask:
                    result["attention_mask"] = torch.tensor([[]], dtype=torch.long)
                return result
            return []
        
        try:
            kwargs = {
                'add_special_tokens': add_special_tokens,
                'return_attention_mask': return_attention_mask,
            }
            
            if max_length is not None:
                kwargs['max_length'] = max_length
                kwargs['truncation'] = truncation
            else:
                kwargs['truncation'] = False
                
            if padding and max_length is not None:
                kwargs['padding'] = 'max_length'
            elif padding:
                kwargs['padding'] = True
            else:
                kwargs['padding'] = False
                
            if return_tensors is not None:
                kwargs['return_tensors'] = return_tensors
            
            result = self.hf_tokenizer(text, **kwargs)
            
            # Si no se solicitan tensors, devolver solo la lista de IDs
            if return_tensors is None:
                if isinstance(result, dict) and "input_ids" in result:
                    input_ids = result["input_ids"]
                    if hasattr(input_ids, 'tolist'):
                        return input_ids.tolist()
                    return input_ids
                elif isinstance(result, list):
                    return result
                else:
                    return self.hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error en encode: {e}")
            tokens = self.hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)
            if return_tensors == "pt":
                result = {"input_ids": torch.tensor([tokens], dtype=torch.long)}
                if return_attention_mask:
                    attention_mask = [1] * len(tokens)
                    result["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)
                return result
            return tokens
    
    def __call__(self, text: Union[str, List[str]], **kwargs) -> Dict:
        """Hacer el tokenizer callable como HuggingFace"""
        return self.hf_tokenizer(text, **kwargs)
    
    def decode(self, 
               token_ids: Union[List[int], torch.Tensor], 
               skip_special_tokens: bool = True,
               clean_up_tokenization_spaces: bool = True) -> str:
        """Decodificar tokens a texto"""
        if hasattr(token_ids, 'tolist'):
            if len(token_ids.shape) > 1:
                token_ids = token_ids.squeeze()
            token_ids = token_ids.tolist()
        
        try:
            return self.hf_tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
        except Exception as e:
            print(f"⚠️ Error en decode: {e}")
            return "<error_decoding>"
    
    def __len__(self) -> int:
        """Retornar tamaño del vocabulario"""
        return len(self.hf_tokenizer)
    
    def get_vocab_size(self) -> int:
        """Obtener tamaño del vocabulario"""
        return len(self.hf_tokenizer)
    
    def save_pretrained(self, save_directory: str):
        """Guardar tokenizador"""
        self.hf_tokenizer.save_pretrained(save_directory)
        print(f"💾 Tokenizador guardado en: {save_directory}")
    
    def load_pretrained(self, load_directory: str):
        """Cargar tokenizador guardado"""
        try:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(load_directory, use_fast=True)
            self._create_compatibility_mappings()
            print(f"📁 Tokenizador cargado desde: {load_directory}")
            return self
        except Exception as e:
            print(f"⚠️ Error cargando tokenizador: {e}")
            return self

# Función de conveniencia para crear instancias
def create_tokenizer(model_name: str = "openai-community/gpt2") -> HuggingFaceTokenizerWrapper:
    """
    Función de conveniencia para crear tokenizadores
    
    Opciones recomendadas:
    - "openai-community/gpt2": Inglés, rápido, 50K vocab
    - "DeepESP/gpt2-spanish": Español, 50K vocab  
    - "bert-base-multilingual-cased": 104 idiomas, 119K vocab
    """
    return HuggingFaceTokenizerWrapper(model_name)

if __name__ == "__main__":
    # Test básico
    print("🧪 Testing Simple HuggingFace Tokenizer")
    
    tokenizer = create_tokenizer("openai-community/gpt2")
    
    test_texts = [
        "Hello world",
        "Hello  world",  # Doble espacio - GPT2 lo maneja nativamente
        "Hola mundo",
        "def func():\n    return True",  # Código con indentación
    ]
    
    for text in test_texts:
        print(f"\n🔸 Input: {repr(text)}")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"   Tokens: {len(tokens)}")
        print(f"   Decoded: {repr(decoded)}")
        print(f"   Match: {'✅' if decoded == text else '⚠️'}")