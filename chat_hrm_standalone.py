#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Chat Interactivo para HRM-Models Standalone
Utiliza modelos standalone sin dependencias de transformers
"""

import os
import torch
import sys
import argparse

# Evitar ejecución del código de entrenamiento al importar
os.environ['HRM_IMPORT_ONLY'] = '1'

def load_standalone_model_and_tokenizer(standalone_script_path, model_path):
    """Carga el modelo y tokenizer desde script standalone"""
    print(f"🔄 Cargando desde script standalone: {standalone_script_path}")
    print(f"🔄 Modelo desde directorio: {model_path}")
    
    # Verificar que el script standalone existe
    if not os.path.exists(standalone_script_path):
        print(f"❌ Error: Script standalone no existe: {standalone_script_path}")
        return None, None, None
    
    # Verificar que el directorio del modelo existe
    if not os.path.exists(model_path):
        print(f"❌ Error: Directorio del modelo no existe: {model_path}")
        return None, None, None
    
    try:
        # Importar dinámicamente desde el script standalone
        import importlib.util
        spec = importlib.util.spec_from_file_location("standalone_model", standalone_script_path)
        standalone_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(standalone_module)
        
        # Obtener las clases necesarias
        HRMText1 = standalone_module.HRMText1
        HRMText1Config = standalone_module.HRMText1Config
        SimpleTokenizer = standalone_module.SimpleTokenizer
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cargar tokenizer
        vocab_path = os.path.join(model_path, "vocab.json")
        if os.path.exists(vocab_path):
            tokenizer = SimpleTokenizer()
            # Cargar desde vocab.json
            import json
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
                tokenizer.word_to_id = vocab_data['word_to_id']
                tokenizer.id_to_word = {int(k): v for k, v in vocab_data['id_to_word'].items()}
                tokenizer.vocab_size = vocab_data['vocab_size']
                if 'special_tokens' in vocab_data:
                    tokenizer.special_tokens = vocab_data['special_tokens']
            print(f"✅ Tokenizer cargado desde: {vocab_path}")
        else:
            print("⚠️ Tokenizer no encontrado, creando uno básico...")
            tokenizer = SimpleTokenizer(vocab_size=32000)
            # Construir vocabulario básico
            sample_text = "The quick brown fox jumps over the lazy dog. Hello world! How are you today?"
            tokenizer.build_vocab([sample_text])
        
        # Intentar cargar modelo guardado
        try:
            # Buscar checkpoint
            checkpoint_path = os.path.join(model_path, "checkpoint.pth")
            if os.path.exists(checkpoint_path):
                print(f"🔄 Cargando desde checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Crear configuración desde checkpoint o usar valores por defecto
                if 'config' in checkpoint:
                    config_dict = checkpoint['config']
                    config = HRMText1Config(**config_dict)
                    print(f"✅ Configuración cargada desde checkpoint")
                else:
                    # Intentar inferir configuración desde el modelo
                    print("⚠️ No se encontró configuración en checkpoint, infiriendo desde modelo...")
                    # Detectar block_size desde rotary_emb shapes
                    sample_key = 'layers.0.H_module.attn.rotary_emb.cos_cached'
                    if sample_key in checkpoint['model_state_dict']:
                        shape = checkpoint['model_state_dict'][sample_key].shape
                        inferred_block_size = shape[1]  # [1, block_size, 1, head_dim]
                        print(f"🔍 Block size inferido: {inferred_block_size}")
                    else:
                        inferred_block_size = 768  # Fallback
                    
                    config = HRMText1Config(
                        vocab_size=len(tokenizer.word_to_id) if hasattr(tokenizer, 'word_to_id') else 32000,
                        block_size=inferred_block_size,
                        n_embd=384,
                        n_head=12,
                        n_layers=8,
                        d_ff=1536,
                        dropout=0.1,
                        halt_max_steps=4,
                        ponder_loss_weight=1e-2,
                        halt_bias_init=-2.2,
                        gradient_checkpointing=False
                    )
                
                model = HRMText1(config).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Modelo cargado desde checkpoint")
                
            else:
                print(f"❌ No se encontró checkpoint en: {model_path}")
                return None, None, None
        
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            return None, None, None
        
        model.eval()
        print(f"✅ Modelo standalone cargado exitosamente en {device}")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ Error al importar o cargar desde script standalone: {e}")
        return None, None, None

def chat_with_standalone_model(prompt_text, model, tokenizer, device, max_new_tokens=100, temperature=0.7, do_sample=True):
    """Genera respuesta del modelo standalone"""
    # Tokenizar el prompt
    if hasattr(tokenizer, 'encode'):
        input_ids = torch.tensor([tokenizer.encode(prompt_text)]).to(device)
    else:
        # Fallback para tokenizer básico
        tokens = tokenizer._tokenize_text(prompt_text)
        token_ids = [tokenizer.word_to_id.get(token, tokenizer.unk_token_id) for token in tokens]
        input_ids = torch.tensor([token_ids]).to(device)
    
    with torch.inference_mode():
        # Generar usando forward manual ya que el método generate del modelo tiene problemas
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            outputs = model.forward(generated_ids)
            # El modelo retorna tupla (loss, logits, past_key_values)
            if isinstance(outputs, tuple):
                logits = outputs[1][:, -1, :] / temperature
            else:
                logits = outputs.logits[:, -1, :] / temperature
            
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
            if next_token.item() == pad_token_id:
                break
                
        output_ids = generated_ids
    
    # Decodificar la respuesta
    if hasattr(tokenizer, 'decode'):
        response = tokenizer.decode(output_ids[0].cpu().tolist())
    else:
        # Fallback para tokenizer básico
        tokens = []
        for token_id in output_ids[0].cpu().tolist():
            if token_id in tokenizer.id_to_word:
                tokens.append(tokenizer.id_to_word[token_id])
        response = ' '.join(tokens)
    
    # Remover el prompt original de la respuesta
    if response.startswith(prompt_text):
        response = response[len(prompt_text):].strip()
    
    return response

def interactive_chat():
    """Función principal de chat interactivo para modelos standalone"""
    print("🤖 HRM-Models Standalone Chat Interactivo")
    print("=" * 60)
    
    # Parsear argumentos
    parser = argparse.ArgumentParser(description="Chat con HRM-Models Standalone")
    parser.add_argument("--model", type=str, required=True, 
                       help="Ruta al directorio del modelo entrenado")
    parser.add_argument("--script", type=str, required=True,
                       help="Ruta al script standalone (ej: hrm_training_small_50m_standalone.py)")
    
    if len(sys.argv) == 1:
        print("📋 Uso: python chat_hrm_standalone.py --model <ruta_modelo> --script <ruta_script>")
        print("\nEjemplo:")
        print("python chat_hrm_standalone.py \\")
        print("  --model ~/models/hrm_small_50m_output \\")
        print("  --script hrm_training_small_50m_standalone.py")
        return
    
    args = parser.parse_args()
    
    # Expandir rutas
    model_path = os.path.expanduser(args.model)
    script_path = os.path.expanduser(args.script)
    
    # Cargar modelo
    model, tokenizer, device = load_standalone_model_and_tokenizer(script_path, model_path)
    if model is None:
        return
    
    print("\n💬 Chat iniciado. Comandos especiales:")
    print("   /exit - Salir del chat")
    print("   /temp <valor> - Cambiar temperatura (0.1-2.0)")
    print("   /tokens <número> - Cambiar max_new_tokens (10-500)")
    print("   /help - Mostrar ayuda")
    print("\n" + "=" * 60)
    
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
            print("🤖 HRM-Standalone: ", end="", flush=True)
            response = chat_with_standalone_model(
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
    """Función de demostración para modelos standalone"""
    print("🎯 Modo Demostración - HRM-Models Standalone")
    print("=" * 60)
    
    # Usar argumentos para demo también
    if len(sys.argv) < 3:
        print("📋 Para modo demo, usa: python chat_hrm_standalone.py --demo --model <ruta> --script <script>")
        return
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--script", type=str, required=True)
    args = parser.parse_args()
    
    model_path = os.path.expanduser(args.model)
    script_path = os.path.expanduser(args.script)
    
    model, tokenizer, device = load_standalone_model_and_tokenizer(script_path, model_path)
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
        
        response = chat_with_standalone_model(prompt, model, tokenizer, device)
        print(response)

if __name__ == "__main__":
    print("🚀 Iniciando HRM-Models Standalone Chat")
    
    if "--demo" in sys.argv:
        demo_chat()
    else:
        interactive_chat()