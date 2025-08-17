#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM-Text1 Interactive Chat - VERSIÓN AVANZADA
Incluye soporte para múltiples datasets, generación secuencial de tokens,
gestión de contexto y monitoreo de rendimiento.

FUNCIONALIDADES:
- Detección automática de modelos entrenados
- Generación de tokens secuencial (streaming)
- Gestión de historial de conversación
- Parámetros de generación ajustables
- Estadísticas de rendimiento en tiempo real
- Soporte para modelos HRM-Text1
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Generator
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, GenerationMixin
from tqdm.auto import tqdm

# Importar modelo HRM desde el script de entrenamiento
# Manejar tanto ejecución desde script como desde Jupyter/IPython
try:
    # Intentar obtener el directorio del archivo actual
    if '__file__' in globals():
        current_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        # Fallback para Jupyter/IPython - usar directorio actual
        current_dir = os.getcwd()
    
    sys.path.append(current_dir)
    from hrm_llm_training_c4_b import HRMText1, HRMText1Config
    HRM_MODEL_AVAILABLE = True
    print("✅ Modelo HRM-Text1 disponible")
except ImportError as e:
    print("⚠️  Modelo HRM no disponible. Usando modelo estándar.")
    print(f"   Error: {e}")
    HRM_MODEL_AVAILABLE = False

# ==============================================================================
# --- CONFIGURACIÓN Y CONSTANTES ---
# ==============================================================================

# Configuración de datasets disponibles (sincronizada con el script de entrenamiento)
AVAILABLE_DATASETS = {
    "c4": "hrm_text1_c4_output-large",
    "openwebtext": "hrm_text1_openwebtext_output-large", 
    "pile": "hrm_text1_pile_output-large",
    "spanish": "hrm_text1_spanish_output-large",
    "slimpajama": "hrm_text1_slimpajama_output-large",
    "slimpajama_es": "hrm_text1_slimpajama_es_output-large",
    "slimpajama_en": "hrm_text1_slimpajama_en_output-large",
    "mixed": "hrm_text1_mixed_output-large",
    "mixed_es": "hrm_text1_mixed_es_output-large",
    "high_quality": "hrm_text1_high_quality_output-large",
    "multilingual_balanced": "hrm_text1_multilingual_balanced_output-large",
    "experimental_full": "hrm_text1_experimental_full_output-large",
    # Buscar también en el directorio actual de output existente
    "current": "hrm_text1_c4_output-large"  # Para compatibilidad con el directorio existente
}

# Parámetros de generación por defecto
DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "stream_tokens": True
}

# ==============================================================================
# --- CLASES AUXILIARES ---
# ==============================================================================

class ConversationHistory:
    """Gestiona el historial de conversación y contexto"""
    
    def __init__(self, max_history: int = 10, max_context_length: int = 2048):
        self.max_history = max_history
        self.max_context_length = max_context_length
        self.conversations: List[Dict] = []
        
    def add_exchange(self, prompt: str, response: str, stats: Dict):
        """Añade un intercambio al historial"""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "stats": stats
        }
        self.conversations.append(exchange)
        
        # Mantener solo los últimos max_history intercambios
        if len(self.conversations) > self.max_history:
            self.conversations = self.conversations[-self.max_history:]
    
    def get_context(self, include_last_n: int = 3) -> str:
        """Obtiene el contexto de los últimos n intercambios"""
        if not self.conversations:
            return ""
        
        context_parts = []
        for conv in self.conversations[-include_last_n:]:
            context_parts.append(f"Usuario: {conv['prompt']}")
            context_parts.append(f"Asistente: {conv['response']}")
        
        return "\n".join(context_parts)
    
    def save_to_file(self, filepath: str):
        """Guarda el historial en un archivo JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, filepath: str):
        """Carga el historial desde un archivo JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.conversations = json.load(f)
        except FileNotFoundError:
            print(f"Archivo de historial no encontrado: {filepath}")

class PerformanceMonitor:
    """Monitorea estadísticas de rendimiento"""
    
    def __init__(self):
        self.reset_stats()
    
    def reset_stats(self):
        """Reinicia las estadísticas"""
        self.total_tokens_generated = 0
        self.total_time_spent = 0.0
        self.generation_count = 0
        self.average_tokens_per_second = 0.0
    
    def record_generation(self, tokens_generated: int, time_taken: float):
        """Registra una generación"""
        self.total_tokens_generated += tokens_generated
        self.total_time_spent += time_taken
        self.generation_count += 1
        self.average_tokens_per_second = self.total_tokens_generated / self.total_time_spent
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas actuales"""
        return {
            "total_tokens": self.total_tokens_generated,
            "total_time": self.total_time_spent,
            "generations": self.generation_count,
            "avg_tokens_per_sec": self.average_tokens_per_second,
            "avg_tokens_per_generation": self.total_tokens_generated / max(1, self.generation_count)
        }

# ==============================================================================
# --- FUNCIONES DE UTILIDAD ---
# ==============================================================================

def find_available_models(base_paths: List[str] = None) -> Dict[str, str]:
    """Encuentra todos los modelos disponibles en el sistema"""
    if base_paths is None:
        # Rutas por defecto a buscar
        base_paths = [
            "/content/drive/MyDrive/HRM",  # Google Colab
            ".",  # Directorio actual
            os.path.expanduser("~/HRM"),  # Directorio home del usuario
            "/Users/julianmican/Documents/HRM/HRM-Text",  # Ruta específica del usuario
        ]
    
    available = {}
    
    # Buscar en todas las rutas base
    for base_path in base_paths:
        if os.path.exists(base_path):
            for dataset_name, folder_name in AVAILABLE_DATASETS.items():
                if dataset_name not in available:  # No sobrescribir si ya se encontró
                    model_path = os.path.join(base_path, folder_name)
                    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
                        available[dataset_name] = model_path
    
    # Buscar directorios que coincidan con patrones conocidos en el directorio actual
    try:
        current_files = os.listdir(".")
        for item in current_files:
            if os.path.isdir(item) and "hrm_text1" in item.lower() and "output" in item.lower():
                config_path = os.path.join(item, "config.json")
                if os.path.exists(config_path):
                    # Intentar identificar el dataset por el nombre del directorio
                    item_lower = item.lower()
                    for dataset_name in AVAILABLE_DATASETS.keys():
                        if dataset_name in item_lower and dataset_name not in available:
                            available[dataset_name] = item
                            break
                    else:
                        # Si no se puede identificar, usar nombre genérico
                        available[f"local_{item}"] = item
    except Exception:
        pass  # Ignorar errores de acceso a directorio
    
    return available

def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """Carga el modelo y tokenizer, detectando automáticamente el tipo"""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔄 Cargando modelo desde: {model_path}")
    print(f"📱 Dispositivo: {device}")
    
    # Cargar tokenizer
    print("📝 Cargando tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_path, use_fast=False, legacy=False)
    
    # Configurar pad_token si no existe
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    # Intentar cargar como modelo HRM primero
    if HRM_MODEL_AVAILABLE:
        try:
            print("🧠 Cargando modelo HRM-Text1...")
            model = HRMText1.from_pretrained(model_path).to(device)
            
            # Compilar si es PyTorch 2.0+
            if torch.__version__.startswith("2"):
                print("⚡ Compilando modelo con torch.compile()...")
                model = torch.compile(model)
            
            print("✅ Modelo HRM-Text1 cargado exitosamente")
            return model, tokenizer
            
        except Exception as e:
            print(f"⚠️  Error cargando modelo HRM: {e}")
            print("🔄 Intentando cargar como modelo estándar...")
    
    # Fallback a modelo estándar
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        print("✅ Modelo estándar cargado exitosamente")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        sys.exit(1)

# ==============================================================================
# --- GENERACIÓN SECUENCIAL DE TOKENS ---
# ==============================================================================

def generate_tokens_streaming(
    model, 
    tokenizer, 
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: str = "cpu"
) -> Generator[str, None, Dict]:
    """
    Genera tokens de forma secuencial, yielding cada token individual
    """
    
    model.eval()
    generated_tokens = 0
    start_time = time.time()
    
    # Configurar para generación
    current_input_ids = input_ids.clone()
    current_attention_mask = attention_mask.clone()
    
    # Configurar parámetros de generación
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id or eos_token_id
    
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.split(':')[0], 
                               dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32, 
                               enabled=(device.startswith('cuda'))):
            
            for step in range(max_new_tokens):
                # Forward pass
                outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask
                )
                
                # Obtener logits del último token
                next_token_logits = outputs.logits[:, -1, :]
                
                # Aplicar temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Aplicar repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(current_input_ids[0].tolist()):
                        if next_token_logits[0, token_id] < 0:
                            next_token_logits[0, token_id] *= repetition_penalty
                        else:
                            next_token_logits[0, token_id] /= repetition_penalty
                
                # Aplicar top-k filtering
                if top_k > 0:
                    top_k_actual = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_actual)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Aplicar top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remover tokens con probabilidad cumulativa > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Muestrear el siguiente token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Decodificar token
                token_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
                
                # Yield el token generado
                yield token_text
                generated_tokens += 1
                
                # Verificar si es token de fin
                if next_token.item() == eos_token_id:
                    break
                
                # Actualizar secuencia de entrada
                current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
                current_attention_mask = torch.cat([
                    current_attention_mask, 
                    torch.ones(1, 1, device=device, dtype=current_attention_mask.dtype)
                ], dim=-1)
                
                # Truncar si es muy largo
                max_length = 2048
                if current_input_ids.size(1) > max_length:
                    current_input_ids = current_input_ids[:, -max_length:]
                    current_attention_mask = current_attention_mask[:, -max_length:]
    
    # Retornar estadísticas finales
    end_time = time.time()
    generation_time = end_time - start_time
    
    return {
        "tokens_generated": generated_tokens,
        "generation_time": generation_time,
        "tokens_per_second": generated_tokens / generation_time if generation_time > 0 else 0
    }

def chat_with_streaming(
    model, 
    tokenizer, 
    prompt: str, 
    device: str = "cpu",
    use_context: str = "",
    **generation_params
) -> tuple[str, Dict]:
    """
    Función principal de chat con streaming de tokens
    """
    
    # Preparar prompt con contexto si existe
    full_prompt = f"{use_context}\nUsuario: {prompt}\nAsistente:" if use_context else f"Usuario: {prompt}\nAsistente:"
    
    # Tokenizar
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print("🤖 Generando respuesta", end="", flush=True)
    
    # Generar tokens con streaming
    response_tokens = []
    stream_generator = generate_tokens_streaming(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        device=device,
        **generation_params
    )
    
    try:
        while True:
            token = next(stream_generator)
            response_tokens.append(token)
            
            # Mostrar token en tiempo real (opcional, comentar si es molesto)
            if generation_params.get("show_tokens", False):
                print(token, end="", flush=True)
            else:
                print(".", end="", flush=True)
                
    except StopIteration as e:
        # Obtener estadísticas finales
        stats = e.value
    
    print()  # Nueva línea después de la generación
    
    # Unir tokens en respuesta completa
    full_response = "".join(response_tokens)
    
    # Limpiar respuesta
    response = full_response.strip()
    
    # Remover prompt de la respuesta si está presente
    if response.startswith(full_prompt):
        response = response[len(full_prompt):].strip()
    
    return response, stats

# ==============================================================================
# --- INTERFAZ PRINCIPAL ---
# ==============================================================================

class InteractiveChatApp:
    """Aplicación principal de chat interactivo"""
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.history = ConversationHistory()
        self.monitor = PerformanceMonitor()
        self.generation_params = DEFAULT_GENERATION_PARAMS.copy()
        
        # Cargar modelo
        if model_path:
            self.load_model(model_path)
        else:
            self.select_model_interactive()
    
    def load_model(self, model_path: str):
        """Carga un modelo específico"""
        self.model, self.tokenizer = load_model_and_tokenizer(model_path, self.device)
        print(f"✅ Modelo cargado desde: {model_path}")
    
    def select_model_interactive(self):
        """Permite seleccionar un modelo interactivamente"""
        available_models = find_available_models()
        
        if not available_models:
            print("❌ No se encontraron modelos disponibles.")
            print("Asegúrate de haber entrenado al menos un modelo.")
            sys.exit(1)
        
        print("\n🎯 MODELOS DISPONIBLES:")
        print("=" * 50)
        for i, (dataset_name, model_path) in enumerate(available_models.items(), 1):
            print(f"{i:2}. {dataset_name:20} → {model_path}")
        
        while True:
            try:
                choice = input(f"\nSelecciona un modelo (1-{len(available_models)}): ").strip()
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(available_models):
                    selected_dataset = list(available_models.keys())[choice_idx]
                    selected_path = available_models[selected_dataset]
                    print(f"🎯 Seleccionado: {selected_dataset}")
                    self.load_model(selected_path)
                    break
                else:
                    print("❌ Selección inválida.")
            except (ValueError, KeyboardInterrupt):
                print("\n👋 Saliendo...")
                sys.exit(0)
    
    def show_help(self):
        """Muestra la ayuda del sistema"""
        help_text = """
🔧 COMANDOS DISPONIBLES:
/help          - Mostrar esta ayuda
/config        - Configurar parámetros de generación  
/stats         - Mostrar estadísticas de rendimiento
/history       - Mostrar historial de conversación
/save          - Guardar historial en archivo
/load          - Cargar historial desde archivo
/clear         - Limpiar historial de conversación
/reset         - Reiniciar estadísticas
/params        - Mostrar parámetros actuales
/exit          - Salir del chat

⚙️  PARÁMETROS DE GENERACIÓN:
- max_new_tokens: Máximo número de tokens a generar
- temperature: Creatividad (0.1-2.0, por defecto 0.7)
- top_k: Top-K filtering (por defecto 50)
- top_p: Nucleus sampling (por defecto 0.9)
- repetition_penalty: Penalización por repetición (por defecto 1.1)
- stream_tokens: Mostrar tokens en tiempo real
        """
        print(help_text)
    
    def configure_generation(self):
        """Configura parámetros de generación interactivamente"""
        print("\n⚙️  CONFIGURACIÓN DE GENERACIÓN")
        print("=" * 40)
        
        params_info = {
            "max_new_tokens": ("Máximo tokens a generar", int, 1, 1000),
            "temperature": ("Creatividad (0.1-2.0)", float, 0.1, 2.0),
            "top_k": ("Top-K filtering", int, 1, 200),
            "top_p": ("Nucleus sampling (0.1-1.0)", float, 0.1, 1.0),
            "repetition_penalty": ("Penalización repetición (1.0-2.0)", float, 1.0, 2.0),
            "stream_tokens": ("Mostrar tokens en tiempo real", bool, None, None)
        }
        
        for param, (desc, param_type, min_val, max_val) in params_info.items():
            current_val = self.generation_params[param]
            print(f"\n{desc}")
            print(f"Valor actual: {current_val}")
            
            new_val = input(f"Nuevo valor (Enter para mantener): ").strip()
            if new_val:
                try:
                    if param_type == bool:
                        new_val = new_val.lower() in ['true', 't', 'yes', 'y', '1']
                    else:
                        new_val = param_type(new_val)
                        if min_val is not None and new_val < min_val:
                            print(f"❌ Valor muy bajo (mínimo: {min_val})")
                            continue
                        if max_val is not None and new_val > max_val:
                            print(f"❌ Valor muy alto (máximo: {max_val})")
                            continue
                    
                    self.generation_params[param] = new_val
                    print(f"✅ {param} actualizado a: {new_val}")
                    
                except ValueError:
                    print(f"❌ Valor inválido para {param}")
        
        print("\n✅ Configuración actualizada")
    
    def show_stats(self):
        """Muestra estadísticas de rendimiento"""
        stats = self.monitor.get_stats()
        print(f"\n📊 ESTADÍSTICAS DE RENDIMIENTO")
        print("=" * 40)
        print(f"Generaciones: {stats['generations']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Tiempo total: {stats['total_time']:.2f}s")
        print(f"Tokens/segundo: {stats['avg_tokens_per_sec']:.2f}")
        print(f"Tokens/generación: {stats['avg_tokens_per_generation']:.1f}")
    
    def show_params(self):
        """Muestra parámetros actuales"""
        print(f"\n⚙️  PARÁMETROS ACTUALES")
        print("=" * 30)
        for param, value in self.generation_params.items():
            print(f"{param:20}: {value}")
    
    def run(self):
        """Ejecuta el bucle principal del chat"""
        print("\n" + "="*60)
        print("🤖 HRM-Text1 CHAT INTERACTIVO - VERSIÓN AVANZADA")
        print("="*60)
        print("💡 Escribe /help para ver comandos disponibles")
        print("💡 Escribe /exit para salir")
        print("💡 La generación usa streaming de tokens secuencial")
        print("="*60)
        
        while True:
            try:
                prompt = input("\n💬 Tú: ").strip()
                
                if not prompt:
                    continue
                
                # Procesar comandos
                if prompt.startswith('/'):
                    command = prompt[1:].lower()
                    
                    if command == 'help':
                        self.show_help()
                    elif command == 'config':
                        self.configure_generation()
                    elif command == 'stats':
                        self.show_stats()
                    elif command == 'params':
                        self.show_params()
                    elif command == 'history':
                        if self.history.conversations:
                            for i, conv in enumerate(self.history.conversations[-5:], 1):
                                print(f"\n{i}. Usuario: {conv['prompt']}")
                                print(f"   Bot: {conv['response'][:100]}...")
                        else:
                            print("📭 Historial vacío")
                    elif command == 'clear':
                        self.history.conversations.clear()
                        print("🗑️  Historial limpiado")
                    elif command == 'reset':
                        self.monitor.reset_stats()
                        print("🔄 Estadísticas reiniciadas")
                    elif command == 'save':
                        filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        self.history.save_to_file(filename)
                        print(f"💾 Historial guardado en: {filename}")
                    elif command == 'load':
                        filename = input("Archivo a cargar: ").strip()
                        self.history.load_from_file(filename)
                        print(f"📂 Historial cargado desde: {filename}")
                    elif command == 'exit':
                        print("👋 ¡Hasta luego!")
                        break
                    else:
                        print("❌ Comando desconocido. Usa /help para ver comandos disponibles.")
                    
                    continue
                
                # Generar respuesta
                context = self.history.get_context(include_last_n=2)
                
                response, stats = chat_with_streaming(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    device=self.device,
                    use_context=context,
                    **self.generation_params
                )
                
                # Mostrar respuesta
                print(f"🤖 Bot: {response}")
                
                # Actualizar estadísticas y historial
                self.monitor.record_generation(stats['tokens_generated'], stats['generation_time'])
                self.history.add_exchange(prompt, response, stats)
                
                # Mostrar estadísticas de la generación
                print(f"📊 {stats['tokens_generated']} tokens en {stats['generation_time']:.2f}s ({stats['tokens_per_second']:.1f} tok/s)")
                
            except KeyboardInterrupt:
                print("\n👋 Saliendo...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

# ==============================================================================
# --- FUNCIÓN PRINCIPAL ---
# ==============================================================================

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="HRM-Text1 Interactive Chat")
    parser.add_argument("--model", "-m", help="Ruta al modelo a cargar")
    parser.add_argument("--device", "-d", default="auto", help="Dispositivo (cuda/cpu/auto)")
    parser.add_argument("--dataset", help="Dataset del modelo a usar (c4, slimpajama, etc.)")
    
    args = parser.parse_args()
    
    # Determinar ruta del modelo
    model_path = None
    if args.model:
        model_path = args.model
    elif args.dataset:
        if args.dataset in AVAILABLE_DATASETS:
            # Buscar modelo en rutas conocidas
            possible_paths = [
                f"/content/drive/MyDrive/HRM/{AVAILABLE_DATASETS[args.dataset]}",
                AVAILABLE_DATASETS[args.dataset]
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                print(f"❌ Modelo para dataset '{args.dataset}' no encontrado")
                sys.exit(1)
        else:
            print(f"❌ Dataset desconocido: {args.dataset}")
            print(f"Disponibles: {', '.join(AVAILABLE_DATASETS.keys())}")
            sys.exit(1)
    
    # Iniciar aplicación
    app = InteractiveChatApp(model_path=model_path, device=args.device)
    app.run()

if __name__ == "__main__":
    main()