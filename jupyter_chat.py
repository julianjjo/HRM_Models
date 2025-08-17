# -*- coding: utf-8 -*-
"""
HRM-Text1 Jupyter Chat - VERSIÓN SIMPLIFICADA PARA NOTEBOOKS
Optimizado para ejecutar en Jupyter/IPython/Colab sin problemas.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer
from typing import Generator, Dict, Tuple
from datetime import datetime

# ==============================================================================
# --- CONFIGURACIÓN ---
# ==============================================================================

# Detectar entorno
IN_COLAB = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False
IN_JUPYTER = 'get_ipython' in globals()

print(f"🔍 Entorno detectado: {'Google Colab' if IN_COLAB else 'Jupyter' if IN_JUPYTER else 'Python estándar'}")

# Importar modelo HRM
try:
    # Añadir directorio actual al path
    if '.' not in sys.path:
        sys.path.append('.')
    
    from hrm_llm_training_c4_b import HRMText1, HRMText1Config
    HRM_MODEL_AVAILABLE = True
    print("✅ Modelo HRM-Text1 disponible")
except ImportError as e:
    print(f"⚠️  Modelo HRM no disponible: {e}")
    print("💡 Asegúrate de que hrm_llm_training_c4_b.py esté en el mismo directorio")
    HRM_MODEL_AVAILABLE = False

# Configuración de modelos
COLAB_MODEL_PATH = "/content/drive/MyDrive/HRM/hrm_text1_c4_output-large"
LOCAL_MODEL_PATHS = [
    "hrm_text1_c4_output-large",
    "./hrm_text1_c4_output-large",
    "hrm_text1_mixed_output-large",
    "./hrm_text1_mixed_output-large"
]

# Parámetros por defecto
DEFAULT_PARAMS = {
    "max_new_tokens": 150,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}

# ==============================================================================
# --- FUNCIONES DE UTILIDAD ---
# ==============================================================================

def find_model_path():
    """Encuentra automáticamente la ruta del modelo"""
    # Primero intentar Colab
    if IN_COLAB and os.path.exists(COLAB_MODEL_PATH):
        return COLAB_MODEL_PATH
    
    # Luego intentar rutas locales
    for path in LOCAL_MODEL_PATHS:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
            return path
    
    # Buscar cualquier directorio con patrón hrm_text1
    try:
        for item in os.listdir('.'):
            if (os.path.isdir(item) and 
                'hrm_text1' in item.lower() and 
                'output' in item.lower() and
                os.path.exists(os.path.join(item, "config.json"))):
                return item
    except:
        pass
    
    return None

def load_model_simple(model_path=None, device="auto"):
    """Carga el modelo de forma simple y robusta"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Encontrar modelo si no se especifica
    if model_path is None:
        model_path = find_model_path()
        if model_path is None:
            raise FileNotFoundError("No se encontró ningún modelo entrenado")
    
    print(f"🔄 Cargando modelo desde: {model_path}")
    print(f"📱 Dispositivo: {device}")
    
    # Cargar tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path, use_fast=False, legacy=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    # Cargar modelo
    if HRM_MODEL_AVAILABLE:
        try:
            model = HRMText1.from_pretrained(model_path).to(device)
            print("✅ Modelo HRM cargado exitosamente")
        except Exception as e:
            print(f"⚠️  Error cargando HRM: {e}")
            print("🔄 Usando modelo estándar...")
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            print("✅ Modelo estándar cargado")
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        print("✅ Modelo estándar cargado")
    
    return model, tokenizer, device

# ==============================================================================
# --- GENERACIÓN DE TOKENS ---
# ==============================================================================

def generate_response_streaming(model, tokenizer, prompt, device, **params):
    """Genera respuesta con streaming de tokens"""
    # Parámetros con valores por defecto
    generation_params = DEFAULT_PARAMS.copy()
    generation_params.update(params)
    
    # Preparar entrada
    full_prompt = f"Usuario: {prompt}\nAsistente:"
    inputs = tokenizer(full_prompt, return_tensors="pt", max_length=1024, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    model.eval()
    generated_tokens = []
    start_time = time.time()
    
    with torch.inference_mode():
        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()
        
        for step in range(generation_params["max_new_tokens"]):
            # Forward pass
            outputs = model(input_ids=current_ids, attention_mask=current_mask)
            logits = outputs.logits[:, -1, :]
            
            # Aplicar temperatura
            if generation_params["temperature"] != 1.0:
                logits = logits / generation_params["temperature"]
            
            # Aplicar repetition penalty
            if generation_params["repetition_penalty"] != 1.0:
                for token_id in set(current_ids[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= generation_params["repetition_penalty"]
                    else:
                        logits[0, token_id] /= generation_params["repetition_penalty"]
            
            # Top-k filtering
            if generation_params["top_k"] > 0:
                top_k = min(generation_params["top_k"], logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
            if generation_params["top_p"] < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > generation_params["top_p"]
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Samplear siguiente token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decodificar y añadir
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
            generated_tokens.append(token_text)
            
            # Parar si es EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Actualizar secuencia
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            current_mask = torch.cat([current_mask, torch.ones(1, 1, device=device)], dim=-1)
            
            # Mostrar progreso
            if step % 10 == 0:
                print(".", end="", flush=True)
    
    # Procesar respuesta
    response = "".join(generated_tokens).strip()
    if response.startswith(full_prompt):
        response = response[len(full_prompt):].strip()
    
    # Estadísticas
    generation_time = time.time() - start_time
    stats = {
        "tokens": len(generated_tokens),
        "time": generation_time,
        "tokens_per_sec": len(generated_tokens) / generation_time if generation_time > 0 else 0
    }
    
    print()  # Nueva línea después de los puntos
    return response, stats

# ==============================================================================
# --- CLASE PRINCIPAL DE CHAT ---
# ==============================================================================

class JupyterChat:
    """Chat simplificado para Jupyter"""
    
    def __init__(self, model_path=None):
        self.model, self.tokenizer, self.device = load_model_simple(model_path)
        self.params = DEFAULT_PARAMS.copy()
        self.history = []
        
        print("\n" + "="*50)
        print("🤖 HRM-Text1 JUPYTER CHAT")
        print("="*50)
        print("✅ Modelo cargado y listo")
        print("💡 Usa chat.generate('tu mensaje') para chatear")
        print("💡 Usa chat.config() para ajustar parámetros")
        print("="*50)
    
    def generate(self, prompt, **kwargs):
        """Genera una respuesta"""
        print(f"💬 Tú: {prompt}")
        print("🤖 Generando", end="", flush=True)
        
        # Combinar parámetros
        params = self.params.copy()
        params.update(kwargs)
        
        # Generar respuesta
        response, stats = generate_response_streaming(
            self.model, self.tokenizer, prompt, self.device, **params
        )
        
        # Mostrar resultado
        print(f"🤖 Bot: {response}")
        print(f"📊 {stats['tokens']} tokens en {stats['time']:.2f}s ({stats['tokens_per_sec']:.1f} tok/s)")
        
        # Guardar en historial
        self.history.append({
            "prompt": prompt,
            "response": response,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def config(self, **kwargs):
        """Configura parámetros"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                print(f"✅ {key} = {value}")
            else:
                print(f"❌ Parámetro desconocido: {key}")
        
        print("\n📋 Parámetros actuales:")
        for key, value in self.params.items():
            print(f"   {key}: {value}")
    
    def show_history(self, last_n=5):
        """Muestra historial reciente"""
        print(f"\n📚 HISTORIAL (últimos {last_n}):")
        print("-" * 40)
        for i, conv in enumerate(self.history[-last_n:], 1):
            print(f"{i}. Usuario: {conv['prompt']}")
            print(f"   Bot: {conv['response'][:100]}{'...' if len(conv['response']) > 100 else ''}")
            print(f"   ({conv['stats']['tokens']} tokens)")
            print()

# ==============================================================================
# --- FUNCIONES DE CONVENIENCIA ---
# ==============================================================================

def quick_chat(model_path=None):
    """Crea un chat rápidamente"""
    return JupyterChat(model_path)

def demo():
    """Ejecuta una demostración"""
    try:
        chat = quick_chat()
        
        print("\n🎮 DEMO INTERACTIVA")
        print("Probando algunas preguntas...")
        
        preguntas = [
            "¿Qué es la inteligencia artificial?",
            "Explícame brevemente qué es Python",
            "¿Cuáles son los beneficios de la energía solar?"
        ]
        
        for pregunta in preguntas:
            print(f"\n{'-'*50}")
            chat.generate(pregunta)
            time.sleep(1)  # Pausa entre preguntas
        
        print(f"\n🏁 Demo completada. Usa 'chat.generate()' para continuar.")
        return chat
        
    except Exception as e:
        print(f"❌ Error en demo: {e}")
        return None

# Mensaje de bienvenida
if __name__ != "__main__":  # Solo mostrar si se importa, no si se ejecuta directamente
    print("\n🚀 HRM-Text1 Jupyter Chat cargado!")
    print("📋 Comandos disponibles:")
    print("   chat = quick_chat()           # Crear chat")
    print("   chat.generate('mensaje')      # Enviar mensaje")
    print("   chat.config(temperature=0.8)  # Ajustar parámetros")
    print("   chat.show_history()           # Ver historial")
    print("   demo()                        # Ejecutar demo")