#!/usr/bin/env python3
"""
Script para convertir modelos HRM a versiones standalone sin dependencias de Transformers
"""

import re
import os

# Clases base standalone que reemplazan las de Transformers
STANDALONE_CLASSES = '''
# ==============================================================================
# --- CLASES BASE STANDALONE (REEMPLAZAN TRANSFORMERS) ---
# ==============================================================================

class SimpleConfig:
    """Configuración base simple para reemplazar PretrainedConfig"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class SimplePreTrainedModel(nn.Module):
    """Modelo base simple para reemplazar PreTrainedModel"""
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, do_sample=True, pad_token_id=0, **kwargs):
        """Generación simple de texto"""
        self.eval()
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids)
                logits = outputs.logits[:, -1, :] / temperature
                
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == pad_token_id:
                    break
        
        return input_ids

class SimpleModelOutput:
    """Output simple para reemplazar transformers ModelOutput"""
    def __init__(self, loss=None, logits=None, **kwargs):
        self.loss = loss
        self.logits = logits
        for key, value in kwargs.items():
            setattr(self, key, value)

class SimpleGenerationMixin:
    """Mixin simple para reemplazar GenerationMixin"""
    pass

# ==============================================================================
# --- TOKENIZER SIMPLE STANDALONE ---
# ==============================================================================

import re
import json
from collections import Counter

class SimpleTokenizer:
    """Tokenizer simple sin dependencias de HuggingFace"""
    
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<s>': 2,
            '</s>': 3,
            '<mask>': 4
        }
        
        # Inicializar con tokens especiales
        for token, token_id in self.special_tokens.items():
            self.word_to_id[token] = token_id
            self.id_to_word[token_id] = token
        
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.mask_token = '<mask>'
        
        self.pad_token_id = self.special_tokens['<pad>']
        self.unk_token_id = self.special_tokens['<unk>']
        self.bos_token_id = self.special_tokens['<s>']
        self.eos_token_id = self.special_tokens['</s>']
        self.mask_token_id = self.special_tokens['<mask>']
        
        self._built = False
    
    def _tokenize_text(self, text):
        """Tokenización básica por palabras y caracteres"""
        # Limpiar y normalizar
        text = text.lower().strip()
        # Separar por espacios y puntuación
        tokens = re.findall(r'\\w+|[^\\w\\s]', text)
        return tokens
    
    def build_vocab(self, texts):
        """Construir vocabulario desde textos"""
        print(f"🔧 Construyendo vocabulario desde {len(texts)} textos...")
        
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize_text(text)
            word_counts.update(tokens)
        
        # Tomar las palabras más frecuentes
        most_common = word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        # Agregar al vocabulario
        current_id = len(self.special_tokens)
        for word, count in most_common:
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
        
        self._built = True
        print(f"✅ Vocabulario construido: {len(self.word_to_id)} tokens")
        return self
    
    def encode(self, text, max_length=None, truncation=True, padding=False, return_tensors=None):
        """Codificar texto a tokens"""
        if not self._built:
            # Construir vocabulario básico si no está construido
            basic_vocab = text.split()[:1000]  # Usar texto para vocabulario básico
            self.build_vocab([text])
        
        tokens = self._tokenize_text(text)
        token_ids = [self.word_to_id.get(token, self.unk_token_id) for token in tokens]
        
        # Truncar si es necesario
        if max_length and truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Padding si es necesario
        if padding and max_length:
            if len(token_ids) < max_length:
                token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
        
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
        
        return token_ids
    
    def __call__(self, text, **kwargs):
        """Hacer el tokenizer callable como Transformers"""
        return self.encode(text, **kwargs)
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decodificar tokens a texto"""
        if hasattr(token_ids, 'tolist'):  # Es un tensor
            if len(token_ids.shape) > 1:
                token_ids = token_ids.squeeze()
            token_ids = token_ids.tolist()
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                token = self.id_to_word[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        return " ".join(tokens)
    
    def __len__(self):
        return len(self.word_to_id)
    
    def save_pretrained(self, save_directory):
        """Guardar tokenizer"""
        os.makedirs(save_directory, exist_ok=True)
        
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w') as f:
            json.dump({
                'word_to_id': self.word_to_id,
                'id_to_word': {str(k): v for k, v in self.id_to_word.items()},
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens
            }, f, indent=2)
        
        print(f"💾 Tokenizer guardado en: {save_directory}")

'''

def convert_model_to_standalone(input_file, output_file):
    """Convertir un modelo a versión standalone"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"🔄 Convirtiendo {input_file} -> {output_file}")
    
    # 1. Actualizar docstring
    content = re.sub(
        r'("""[^"]*HRM-Models Training Script[^"]*?)(\n- .*?)(\n""")',
        r'\1 [STANDALONE VERSION]\2\n- SIN DEPENDENCIAS DE TRANSFORMERS - Implementación standalone\n- Tokenizer simple personalizado incluido\3',
        content,
        flags=re.DOTALL
    )
    
    # 2. Remover imports problemáticos de transformers
    content = re.sub(
        r'from transformers import.*?get_cosine_schedule_with_warmup\n',
        '# Transformers imports removed - using standalone implementations\n',
        content
    )
    
    # 3. Remover import de datasets
    content = re.sub(
        r'from datasets import load_dataset\n',
        '# datasets import removed - using custom dataset loading\n',
        content
    )
    
    # 4. Remover import de huggingface_hub
    content = re.sub(
        r'from huggingface_hub import.*?\n',
        '# huggingface_hub imports removed - using standalone implementations\n',
        content
    )
    
    # 5. Insertar clases standalone después de los imports de torch
    torch_imports_end = content.find('from torch.optim import AdamW')
    if torch_imports_end != -1:
        torch_imports_end = content.find('\n', torch_imports_end) + 1
        content = content[:torch_imports_end] + STANDALONE_CLASSES + content[torch_imports_end:]
    
    # 6. Reemplazar referencias a PretrainedConfig con SimpleConfig
    content = re.sub(r'PretrainedConfig', 'SimpleConfig', content)
    
    # 7. Reemplazar referencias a PreTrainedModel con SimplePreTrainedModel  
    content = re.sub(r'PreTrainedModel', 'SimplePreTrainedModel', content)
    
    # 8. Reemplazar referencias a GenerationMixin
    content = re.sub(r'GenerationMixin', 'SimpleGenerationMixin', content)
    
    # 9. Reemplazar T5Tokenizer con SimpleTokenizer
    content = re.sub(r'T5Tokenizer', 'SimpleTokenizer', content)
    content = re.sub(r'T5_TOKENIZER_REPO = "t5-small"', 'T5_TOKENIZER_REPO = None  # Using SimpleTokenizer instead', content)
    content = re.sub(r'tokenizer = T5Tokenizer\.from_pretrained.*?\n', 'tokenizer = SimpleTokenizer(vocab_size=32000)\n', content)
    
    # 10. Remover código de get_cosine_schedule_with_warmup y usar SimpleCosineScheduler
    content = re.sub(
        r'scheduler = get_cosine_schedule_with_warmup\([^)]*\)',
        'scheduler = SimpleCosineScheduler(optimizer, num_warmup_steps, num_training_steps)',
        content
    )
    
    # 11. Actualizar función de carga de datasets para ser standalone
    # Este es un cambio más complejo que requeriría análisis específico de cada modelo
    
    # 12. Escribir archivo convertido
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Conversión completada: {output_file}")

def main():
    """Convertir todos los modelos a versiones standalone"""
    
    models = [
        'hrm_training_nano_25m_standalone.py',
        'hrm_training_micro_10m_standalone.py', 
        'hrm_training_small_50m_standalone.py',
        'hrm_training_medium_100m_standalone.py',
        'hrm_training_medium_350m_standalone.py',
        'hrm_training_large_1b_standalone.py'
    ]
    
    for model in models:
        if os.path.exists(model):
            # El archivo ya existe como copia, convertirlo in-place
            convert_model_to_standalone(model, model)
        else:
            print(f"⚠️ Archivo no encontrado: {model}")
    
    print("🎉 ¡Conversión de todos los modelos completada!")

if __name__ == "__main__":
    main()