# -*- coding: utf-8 -*-
"""
HRM-Text1 Colab Chat - VERSIÓN OPTIMIZADA PARA GOOGLE COLAB
Diseñado específicamente para ejecutar en Google Colab sin dependencias externas.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, AutoModelForCausalLM
from typing import Generator, Dict, Tuple
from datetime import datetime

# ==============================================================================
# --- CONFIGURACIÓN PARA COLAB ---
# ==============================================================================

# Detectar si estamos en Google Colab
try:
    import google.colab
    IN_COLAB = True
    print("🌟 Google Colab detectado - Usando configuración optimizada")
except ImportError:
    IN_COLAB = False
    print("💻 Entorno local detectado")

# Configuración de modelos para Colab
COLAB_MODEL_PATHS = {
    "c4": "/content/drive/MyDrive/HRM/hrm_text1_c4_output-large",
    "mixed": "/content/drive/MyDrive/HRM/hrm_text1_mixed_output-large",
    "slimpajama": "/content/drive/MyDrive/HRM/hrm_text1_slimpajama_output-large",
    "spanish": "/content/drive/MyDrive/HRM/hrm_text1_spanish_output-large",
}

# Parámetros por defecto optimizados para Colab
DEFAULT_PARAMS = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}

# ==============================================================================
# --- FUNCIONES DE UTILIDAD ---
# ==============================================================================

def mount_drive_if_needed():
    """Monta Google Drive si es necesario"""
    if IN_COLAB:
        try:
            if not os.path.exists('/content/drive'):
                from google.colab import drive
                print("📱 Montando Google Drive...")
                drive.mount('/content/drive')
                print("✅ Google Drive montado correctamente")
            else:
                print("✅ Google Drive ya está montado")
                
        except Exception as e:
            print(f"⚠️  No se pudo montar Google Drive: {e}")
            return False
    return True

def find_available_models():
    """Encuentra modelos disponibles en Colab"""
    available = {}
    
    if IN_COLAB:
        # Buscar en rutas de Google Drive
        for dataset_name, path in COLAB_MODEL_PATHS.items():
            if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
                available[dataset_name] = path
    
    # Buscar en directorio actual
    try:
        for item in os.listdir('.'):
            if (os.path.isdir(item) and 
                'hrm_text1' in item.lower() and 
                'output' in item.lower() and
                os.path.exists(os.path.join(item, "config.json"))):
                # Inferir dataset del nombre
                item_lower = item.lower()
                for dataset in ['c4', 'mixed', 'spanish', 'slimpajama']:
                    if dataset in item_lower:
                        available[f"local_{dataset}"] = item
                        break
                else:
                    available[f"local_{item}"] = item
    except:
        pass
    
    return available

def load_model_colab(model_path, device="auto"):
    """Carga modelo optimizado para Colab"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔄 Cargando modelo desde: {model_path}")
    print(f"📱 Dispositivo: {device}")
    
    try:
        # Cargar tokenizer
        print("📝 Cargando tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(model_path, use_fast=False, legacy=False)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
        # Cargar modelo estándar (más compatible)
        print("🧠 Cargando modelo...")
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        
        print("✅ Modelo cargado exitosamente")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None, None, None

# ==============================================================================
# --- GENERACIÓN DE TOKENS ---
# ==============================================================================

def generate_response_colab(model, tokenizer, prompt, device, **params):
    """Genera respuesta optimizada para Colab"""
    # Parámetros con valores por defecto
    generation_params = DEFAULT_PARAMS.copy()
    generation_params.update(params)
    
    # Preparar entrada
    full_prompt = f"Usuario: {prompt}\nAsistente:"
    inputs = tokenizer(full_prompt, return_tensors="pt", max_length=1024, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print("🤖 Generando respuesta", end="", flush=True)
    
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
            
            # Mostrar progreso
            if step % 5 == 0:
                print(".", end="", flush=True)
            
            # Parar si es EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Actualizar secuencia
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            current_mask = torch.cat([current_mask, torch.ones(1, 1, device=device)], dim=-1)
            
            # Truncar si es muy largo
            if current_ids.size(1) > 2048:
                current_ids = current_ids[:, -1024:]
                current_mask = current_mask[:, -1024:]
    
    print()  # Nueva línea
    
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
    
    return response, stats

# ==============================================================================
# --- CLASE PRINCIPAL ---
# ==============================================================================

class ColabChat:
    """Chat optimizado para Google Colab"""
    
    def __init__(self, model_path=None):
        print("🚀 Inicializando HRM Chat para Google Colab")
        
        # Montar drive si es necesario
        if not mount_drive_if_needed():
            print("⚠️  Continuando sin Google Drive...")
        
        # Encontrar y cargar modelo
        if model_path is None:
            available_models = find_available_models()
            if not available_models:
                print("❌ No se encontraron modelos disponibles")
                print("💡 Asegúrate de que el modelo esté en Google Drive")
                print("💡 Rutas esperadas:")
                for name, path in COLAB_MODEL_PATHS.items():
                    print(f"   {name}: {path}")
                return
            
            # Usar el primer modelo disponible
            model_name = list(available_models.keys())[0]
            model_path = available_models[model_name]
            print(f"🎯 Usando modelo: {model_name} desde {model_path}")
        
        # Cargar modelo
        self.model, self.tokenizer, self.device = load_model_colab(model_path)
        if self.model is None:
            print("❌ No se pudo cargar el modelo")
            return
        
        self.params = DEFAULT_PARAMS.copy()
        self.history = []
        
        print("\n" + "="*50)
        print("🤖 HRM-Text1 COLAB CHAT")
        print("="*50)
        print("✅ Modelo cargado y listo para usar")
        print("💡 Usa chat('tu mensaje') para conversar")
        print("💡 Usa config() para ajustar parámetros")
        print("="*50)
    
    def chat(self, prompt, **kwargs):
        """Función principal de chat"""
        if self.model is None:
            print("❌ Modelo no disponible")
            return
        
        print(f"💬 Tú: {prompt}")
        
        # Combinar parámetros
        params = self.params.copy()
        params.update(kwargs)
        
        # Generar respuesta
        response, stats = generate_response_colab(
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
        """Configura parámetros de generación"""
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
        """Muestra historial de conversación"""
        print(f"\n📚 HISTORIAL (últimos {last_n}):")
        print("-" * 40)
        for i, conv in enumerate(self.history[-last_n:], 1):
            print(f"{i}. Tú: {conv['prompt']}")
            print(f"   Bot: {conv['response'][:100]}{'...' if len(conv['response']) > 100 else ''}")
            print(f"   ({conv['stats']['tokens']} tokens, {conv['stats']['time']:.1f}s)")
            print()
    
    def quick_demo(self):
        """Ejecuta una demo rápida"""
        demos = [
            "Hola, ¿cómo estás?",
            "¿Qué es la inteligencia artificial?",
            "Explícame brevemente qué es Python"
        ]
        
        print("\n🎮 DEMO RÁPIDA")
        for prompt in demos:
            print(f"\n{'-'*40}")
            self.chat(prompt)
            time.sleep(1)
        
        print(f"\n🏁 Demo completada!")

# ==============================================================================
# --- FUNCIONES DE CONVENIENCIA ---
# ==============================================================================

def setup_colab_chat(model_path=None):
    """Configuración rápida para Colab"""
    return ColabChat(model_path)

def quick_demo():
    """Demo rápida"""
    chat = setup_colab_chat()
    if chat.model is not None:
        chat.quick_demo()
    return chat

def list_available_models():
    """Lista modelos disponibles"""
    mount_drive_if_needed()
    models = find_available_models()
    
    print("🔍 MODELOS DISPONIBLES:")
    if models:
        for name, path in models.items():
            print(f"  • {name}: {path}")
    else:
        print("  ❌ No se encontraron modelos")
        print("  💡 Verifica que los modelos estén en Google Drive:")
        for name, path in COLAB_MODEL_PATHS.items():
            print(f"     {name}: {path}")
    
    return models

# ==============================================================================
# --- INSTRUCCIONES DE USO ---
# ==============================================================================

def show_usage():
    """Muestra instrucciones de uso"""
    print("""
🚀 HRM-Text1 COLAB CHAT - INSTRUCCIONES DE USO

📋 CONFIGURACIÓN INICIAL:
1. Ejecuta esta celda para cargar el chat
2. El sistema montará automáticamente Google Drive
3. Detectará modelos disponibles automáticamente

💬 CHAT BÁSICO:
   # Crear chat
   chat = setup_colab_chat()
   
   # Conversar
   chat.chat("Hola, ¿cómo estás?")
   
   # Ver historial
   chat.show_history()

⚙️ CONFIGURACIÓN AVANZADA:
   # Ajustar creatividad
   chat.config(temperature=0.8, max_new_tokens=300)
   
   # Chat con parámetros específicos
   chat.chat("Escribe un poema", temperature=1.2)

🎮 DEMO RÁPIDA:
   # Ejecutar demo automática
   demo_chat = quick_demo()

🔍 UTILIDADES:
   # Ver modelos disponibles
   list_available_models()
   
   # Cargar modelo específico
   chat = setup_colab_chat("/ruta/a/tu/modelo")

📁 RUTAS DE MODELOS EN GOOGLE DRIVE:
   c4: /content/drive/MyDrive/HRM/hrm_text1_c4_output-large
   mixed: /content/drive/MyDrive/HRM/hrm_text1_mixed_output-large
   spanish: /content/drive/MyDrive/HRM/hrm_text1_spanish_output-large
   slimpajama: /content/drive/MyDrive/HRM/hrm_text1_slimpajama_output-large
""")

# Mostrar instrucciones al importar
if __name__ != "__main__":
    print("📚 HRM-Text1 Colab Chat cargado exitosamente!")
    print("💡 Ejecuta show_usage() para ver instrucciones completas")
    print("🚀 Inicio rápido: chat = setup_colab_chat()")