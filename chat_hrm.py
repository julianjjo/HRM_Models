#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Chat Interactivo para HRM-Models Micro 10M
Utiliza el modelo entrenado por hrm_training_micro_10m.py
"""

import os
import torch
from transformers import T5Tokenizer
import sys

# Evitar ejecución del código de entrenamiento al importar
import os
os.environ['HRM_IMPORT_ONLY'] = '1'

# Importar clases necesarias del script de entrenamiento
try:
    from hrm_training_micro_10m import HRMText1, HRMText1Config
except ImportError:
    print("❌ Error: No se pudo importar HRMText1 desde hrm_training_micro_10m.py")
    print("   Asegúrate de que el archivo está en el mismo directorio.")
    sys.exit(1)

def load_model_and_tokenizer(model_path):
    """Carga el modelo y tokenizer entrenado"""
    print(f"🔄 Cargando modelo desde: {model_path}")
    
    # Verificar si el directorio existe
    if not os.path.exists(model_path):
        print(f"❌ Error: El directorio del modelo no existe: {model_path}")
        return None, None
    
    try:
        # Cargar tokenizer (mismo que el training)
        tokenizer = T5Tokenizer.from_pretrained("t5-small", use_fast=False, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.padding_side = "left"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Intentar cargar como modelo guardado con save_pretrained()
        try:
            model = HRMText1.from_pretrained(model_path).to(device)
            print(f"✅ Modelo cargado con from_pretrained desde: {model_path}")
        except Exception:
            # Si falla, cargar desde checkpoint manual
            print(f"🔄 Cargando desde checkpoint manual...")
            
            # Crear configuración (debe coincidir con MODEL_PARAMS del training)
            config = HRMText1Config(
                vocab_size=len(tokenizer),
                block_size=512,
                n_embd=256,
                n_head=8,
                n_layers=6,
                d_ff=1024,
                dropout=0.1,
                halt_max_steps=4,
                ponder_loss_weight=1e-2,
                halt_bias_init=-2.2,
                gradient_checkpointing=False
            )
            
            model = HRMText1(config).to(device)
            
            # Cargar pesos del checkpoint
            checkpoint_file = os.path.join(model_path, "checkpoint.pth")
            if os.path.exists(checkpoint_file):
                checkpoint = torch.load(checkpoint_file, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Checkpoint cargado desde: {checkpoint_file}")
            else:
                print(f"⚠️ No se encontró checkpoint en: {model_path}")
                return None, None, None
        
        model.eval()
        print(f"✅ Modelo cargado exitosamente en {device}")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return None, None, None

def chat_with_model(prompt_text, model, tokenizer, device, max_new_tokens=100, temperature=0.7, top_k=50):
    """Genera respuesta del modelo para el prompt dado"""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Remover el prompt original de la respuesta
    if response.startswith(prompt_text):
        response = response[len(prompt_text):].strip()
    
    return response

def interactive_chat():
    """Función principal de chat interactivo"""
    print("🤖 HRM-Models Chat Interactivo")
    print("=" * 50)
    
    # Ruta del modelo (ajustar según sea necesario)
    model_path = os.path.expanduser("/Users/julianmican/Documents/model10MFinalTraning/HRM_Models/hrm_text1_c4_micro_10m_output")
    
    # Cargar modelo
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    if model is None:
        return
    
    print("\n💬 Chat iniciado. Comandos especiales:")
    print("   /exit - Salir del chat")
    print("   /temp <valor> - Cambiar temperatura (0.1-2.0)")
    print("   /tokens <número> - Cambiar max_new_tokens (10-500)")
    print("   /help - Mostrar ayuda")
    print("\n" + "=" * 50)
    
    # Configuración por defecto
    temperature = 0.7
    max_tokens = 100
    
    while True:
        try:
            # Obtener input del usuario
            user_input = input("\n🔥 Tú: ").strip()
            
            if not user_input:
                continue
            
            # Comandos especiales
            if user_input.lower() == "/exit":
                print("👋 ¡Hasta luego!")
                break
            elif user_input.lower() == "/help":
                print("\n📋 Comandos disponibles:")
                print("   /exit - Salir del chat")
                print("   /temp <valor> - Cambiar temperatura (0.1-2.0)")
                print("   /tokens <número> - Cambiar max_new_tokens (10-500)")
                print("   /help - Mostrar esta ayuda")
                continue
            elif user_input.startswith("/temp "):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"🌡️ Temperatura cambiada a: {temperature}")
                    else:
                        print("❌ Temperatura debe estar entre 0.1 y 2.0")
                except (IndexError, ValueError):
                    print("❌ Uso: /temp <valor> (ejemplo: /temp 0.8)")
                continue
            elif user_input.startswith("/tokens "):
                try:
                    new_tokens = int(user_input.split()[1])
                    if 10 <= new_tokens <= 500:
                        max_tokens = new_tokens
                        print(f"🔢 Max tokens cambiado a: {max_tokens}")
                    else:
                        print("❌ Max tokens debe estar entre 10 y 500")
                except (IndexError, ValueError):
                    print("❌ Uso: /tokens <número> (ejemplo: /tokens 150)")
                continue
            
            # Generar respuesta
            print("🤖 HRM-Models: ", end="", flush=True)
            response = chat_with_model(
                user_input, 
                model, 
                tokenizer, 
                device,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrumpido. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error durante la generación: {e}")

def demo_chat():
    """Función de demostración con prompts predefinidos"""
    print("🎯 Modo Demostración - HRM-Models")
    print("=" * 50)
    
    model_path = os.path.expanduser("~/Documents/HRM/hrm_text1_c4_micro_10m_output")
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    if model is None:
        return
    
    demo_prompts = [
        "The future of artificial intelligence",
        "In a world where technology",
        "The cat sat on the mat and",
        "To solve this problem, we need to",
        "Once upon a time in a distant galaxy"
    ]
    
    print("\n🎪 Generando respuestas para prompts de demostración...")
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n--- Ejemplo {i}/5 ---")
        print(f"📝 Prompt: {prompt}")
        print("🤖 Respuesta: ", end="", flush=True)
        
        response = chat_with_model(prompt, model, tokenizer, device)
        print(response)

if __name__ == "__main__":
    print("🚀 Iniciando HRM-Models Chat")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_chat()
    else:
        interactive_chat()