#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Prueba para HRM Micro 10M Model
Prueba la funcionalidad del modelo entrenado incluyendo:
- Carga del modelo y tokenizador
- Generación de texto
- Evaluación de métricas
- Análisis de la arquitectura HRM
"""

import os
import sys
import time
import math
import argparse
from typing import List, Dict, Optional
import torch
import torch.nn.functional as F

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ tqdm no disponible, usando progreso básico")

def load_model_and_tokenizer(model_path: str):
    """Cargar modelo HRM y tokenizador desde el directorio guardado"""
    print(f"🔄 Cargando modelo desde: {model_path}")
    
    # Verificar que existan los archivos necesarios
    model_file = os.path.join(model_path, "pytorch_model.bin")
    config_file = os.path.join(model_path, "config.json")
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No se encontró pytorch_model.bin en {model_path}")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"No se encontró config.json en {model_path}")
    
    # Cargar configuración
    import json
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    print(f"📋 Configuración del modelo:")
    for key, value in config_dict.items():
        print(f"   {key}: {value}")
    
    # Importar las clases del modelo
    try:
        from hrm_training_micro_10m_standalone_hf import HRMText1Config, HRMText1
        from hf_tokenizer_wrapper_simple import create_tokenizer
        print("✅ Clases del modelo importadas correctamente")
    except ImportError as e:
        print(f"❌ Error importando clases del modelo: {e}")
        print("💡 Asegúrese de que hrm_training_micro_10m_standalone_hf.py y hf_tokenizer_wrapper_simple.py estén disponibles")
        return None, None
    
    # Crear configuración del modelo
    config = HRMText1Config(
        vocab_size=config_dict['vocab_size'],
        block_size=config_dict['block_size'],
        n_embd=config_dict['n_embd'],
        n_head=config_dict['n_head'],
        n_layers=config_dict['n_layers'],
        d_ff=config_dict['d_ff'],
        dropout=config_dict['dropout'],
        use_rotary_embeddings=config_dict.get('use_rotary_embeddings', True),
        pad_token_id=config_dict.get('pad_token_id', 0)
    )
    
    # Crear modelo
    print("🧠 Creando modelo HRM...")
    model = HRMText1(config)
    
    # Cargar pesos
    print("💾 Cargando pesos del modelo...")
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Crear tokenizador
    tokenizer_name = config_dict.get('hf_tokenizer_name', 'openai-community/gpt2')
    print(f"🔧 Cargando tokenizador: {tokenizer_name}")
    
    # Intentar cargar desde el directorio del modelo primero
    try:
        tokenizer = create_tokenizer(model_path)
        print("✅ Tokenizador cargado desde modelo guardado")
    except:
        # Fallback al tokenizador original
        tokenizer = create_tokenizer(tokenizer_name)
        print(f"✅ Tokenizador cargado desde HF Hub: {tokenizer_name}")
    
    # Estadísticas del modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Estadísticas del modelo:")
    print(f"   Total parámetros: {total_params:,}")
    print(f"   Parámetros entrenables: {trainable_params:,}")
    print(f"   Tamaño vocabulario: {len(tokenizer):,}")
    
    return model, tokenizer

def test_text_generation(model, tokenizer, prompts: List[str], max_length: int = 100, 
                        temperature: float = 0.8, top_k: int = 50, device: str = 'cuda'):
    """Probar generación de texto con varios prompts"""
    print(f"\n🎯 Probando generación de texto...")
    print(f"   Configuración: max_length={max_length}, temperature={temperature}, top_k={top_k}")
    
    model.eval()
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n📝 Prompt {i+1}: '{prompt}'")
        
        try:
            # Tokenizar prompt
            print(f"   🔄 Tokenizando prompt...")
            # Usar .encode() y convertir manualmente a tensor
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # Añadir batch dimension
            print(f"   📊 Input shape: {input_ids.shape}")
            
            start_time = time.time()
            generated_tokens = []
            
            with torch.no_grad():
                current_ids = input_ids
                
                for step in range(max_length):
                    print(f"   🔄 Paso de generación {step+1}/{max_length}")
                    
                    # Forward pass con debugging
                    try:
                        outputs = model(current_ids)
                        print(f"   ✅ Forward pass exitoso, tipo output: {type(outputs)}")
                        
                        # Extraer logits con mejor manejo de errores
                        if isinstance(outputs, dict):
                            if 'logits' in outputs:
                                logits = outputs['logits']
                            else:
                                print(f"   🔍 Keys disponibles en output dict: {list(outputs.keys())}")
                                # Intentar encontrar logits en otros campos posibles
                                logits = None
                                for key in outputs.keys():
                                    if 'logit' in key.lower() and torch.is_tensor(outputs[key]):
                                        logits = outputs[key]
                                        print(f"   ✅ Encontrados logits en key: {key}")
                                        break
                                if logits is None:
                                    raise ValueError("No se encontraron logits en el output del modelo")
                        else:
                            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                        
                        print(f"   📊 Logits shape: {logits.shape}")
                        next_token_logits = logits[0, -1, :] / temperature
                        
                        # Verificar que los logits sean válidos
                        if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                            print(f"   ⚠️ Logits inválidos detectados (NaN/Inf)")
                            # Usar distribución uniforme como fallback
                            next_token_logits = torch.ones_like(next_token_logits)
                        
                        # Top-k sampling
                        if top_k > 0:
                            top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                            next_token_probs = F.softmax(top_k_logits, dim=-1)
                            
                            # Verificar probabilidades
                            if torch.isnan(next_token_probs).any():
                                print(f"   ⚠️ Probabilidades NaN, usando distribución uniforme")
                                next_token_probs = torch.ones_like(next_token_probs) / next_token_probs.size(-1)
                            
                            next_token_idx = torch.multinomial(next_token_probs, 1)
                            next_token = top_k_indices[next_token_idx]
                        else:
                            next_token_probs = F.softmax(next_token_logits, dim=-1)
                            if torch.isnan(next_token_probs).any():
                                print(f"   ⚠️ Probabilidades NaN, usando distribución uniforme")
                                next_token_probs = torch.ones_like(next_token_probs) / next_token_probs.size(-1)
                            next_token = torch.multinomial(next_token_probs, 1)
                        
                        print(f"   🎯 Token generado: {next_token.item()}")
                        
                        generated_tokens.append(next_token.item())
                        current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                        
                        # Parar en token especial
                        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                            if next_token.item() == tokenizer.eos_token_id:
                                print(f"   🛑 EOS token detectado, parando generación")
                                break
                        
                        # Prevenir secuencias muy largas
                        if current_ids.size(1) > 256:  # Límite de seguridad
                            print(f"   ⚠️ Límite de longitud alcanzado, parando")
                            break
                            
                    except Exception as forward_error:
                        print(f"   ❌ Error en forward pass del paso {step+1}: {forward_error}")
                        import traceback
                        traceback.print_exc()
                        break
            
            generation_time = time.time() - start_time
            
            # Decodificar texto generado
            try:
                full_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
                generated_text = full_text[len(prompt):] if len(full_text) > len(prompt) else ""
            except Exception as decode_error:
                print(f"   ⚠️ Error decodificando: {decode_error}")
                generated_text = f"[Error decodificando {len(generated_tokens)} tokens]"
                full_text = prompt + generated_text
            
            tokens_per_sec = len(generated_tokens) / generation_time if generation_time > 0 else 0
            
            print(f"   ✅ Generado en {generation_time:.2f}s ({tokens_per_sec:.1f} tokens/s)")
            print(f"   📊 Tokens generados: {len(generated_tokens)}")
            print(f"   📄 Texto generado: '{generated_text[:200]}{'...' if len(generated_text) > 200 else ''}'")
            
            results.append({
                'prompt': prompt,
                'generated_text': generated_text,
                'full_text': full_text,
                'tokens_generated': len(generated_tokens),
                'generation_time': generation_time,
                'tokens_per_sec': tokens_per_sec
            })
            
        except Exception as e:
            print(f"   ❌ Error generando texto: {type(e).__name__}: {e}")
            import traceback
            print(f"   🔍 Stack trace:")
            traceback.print_exc()
            results.append({
                'prompt': prompt,
                'error': str(e),
                'error_type': type(e).__name__
            })
    
    return results

def analyze_hrm_behavior(model, tokenizer, test_texts: List[str], device: str = 'cuda'):
    """Analizar comportamiento específico de HRM (ciclos, convergencia, etc.)"""
    print(f"\n🔬 Analizando comportamiento HRM...")
    
    model.eval()
    hrm_stats = {
        'total_samples': 0,
        'avg_h_cycles': 0,
        'avg_l_steps_per_layer': [],
        'convergence_rates': [],
        'ponder_losses': [],
        'q_values_analysis': []
    }
    
    with torch.no_grad():
        for i, text in enumerate(test_texts[:5]):  # Analizar solo primeras 5 muestras
            print(f"   🔄 Analizando muestra {i+1}/5...")
            
            try:
                # Tokenizar
                print(f"   📝 Tokenizando texto: '{text[:50]}...'")
                # Usar .encode() y convertir manualmente a tensor
                tokens = tokenizer.encode(text[:200], truncation=True, max_length=128)
                input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # Añadir batch dimension
                print(f"   📊 Input IDs shape: {input_ids.shape}")
                
                # Forward pass con análisis HRM
                print(f"   🔄 Ejecutando forward pass...")
                outputs = model(input_ids)
                print(f"   ✅ Forward pass completado, tipo: {type(outputs)}")
                
                # Verificar estructura del output
                if isinstance(outputs, dict):
                    print(f"   🔍 Keys en output: {list(outputs.keys())}")
                    
                    # Buscar información HRM en diferentes posibles ubicaciones
                    hrm_info = None
                    if 'hrm_info' in outputs:
                        hrm_info = outputs['hrm_info']
                        print(f"   ✅ Encontrada hrm_info directamente")
                    elif 'hidden_states' in outputs:
                        hrm_info = outputs['hidden_states']
                        print(f"   ✅ Usando hidden_states como hrm_info")
                    else:
                        # Buscar cualquier campo que pueda contener info HRM
                        for key, value in outputs.items():
                            if isinstance(value, list) and len(value) > 0:
                                if isinstance(value[0], dict):
                                    hrm_info = value
                                    print(f"   ✅ Encontrada info HRM en key: {key}")
                                    break
                    
                    if hrm_info is not None:
                        print(f"   📊 HRM info encontrada, tipo: {type(hrm_info)}, longitud: {len(hrm_info) if isinstance(hrm_info, list) else 'N/A'}")
                        
                        # Analizar información HRM por capa
                        if isinstance(hrm_info, list):
                            for layer_idx, layer_info in enumerate(hrm_info):
                                print(f"     🔧 Analizando capa {layer_idx}")
                                
                                if isinstance(layer_info, dict):
                                    print(f"       🔍 Keys en layer_info: {list(layer_info.keys())}")
                                    
                                    # L-steps por capa
                                    if len(hrm_stats['avg_l_steps_per_layer']) <= layer_idx:
                                        hrm_stats['avg_l_steps_per_layer'].append([])
                                    
                                    l_steps = layer_info.get('l_steps', 0)
                                    print(f"       📊 L-steps: {l_steps}")
                                    hrm_stats['avg_l_steps_per_layer'][layer_idx].append(l_steps)
                                    
                                    # Convergencia
                                    convergence = layer_info.get('convergence_achieved', False)
                                    print(f"       ✅ Convergencia: {convergence}")
                                    if len(hrm_stats['convergence_rates']) <= layer_idx:
                                        hrm_stats['convergence_rates'].extend([[] for _ in range(layer_idx + 1 - len(hrm_stats['convergence_rates']))])
                                    hrm_stats['convergence_rates'][layer_idx].append(1 if convergence else 0)
                                    
                                    # Ponder loss
                                    ponder_loss = layer_info.get('ponder_loss', None)
                                    if ponder_loss is not None:
                                        print(f"       💭 Ponder loss: {ponder_loss}")
                                        hrm_stats['ponder_losses'].append(ponder_loss)
                                    
                                    # Q-values si están disponibles
                                    q_values = layer_info.get('q_values', None)
                                    if q_values is not None:
                                        q_count = len(q_values) if isinstance(q_values, list) else 1
                                        print(f"       🎯 Q-values encontrados: {q_count}")
                                        hrm_stats['q_values_analysis'].append(q_count)
                                else:
                                    print(f"       ⚠️ Layer info no es dict: {type(layer_info)}")
                        else:
                            print(f"   ⚠️ HRM info no es lista: {type(hrm_info)}")
                    else:
                        print(f"   ⚠️ No se encontró información HRM en el output")
                else:
                    print(f"   ⚠️ Output no es diccionario: {type(outputs)}")
                    # Verificar si tiene atributo hidden_states
                    if hasattr(outputs, 'hidden_states'):
                        print(f"   🔍 Encontrado hidden_states attribute")
                        hrm_info = outputs.hidden_states
                        # Repetir análisis...
                
                hrm_stats['total_samples'] += 1
                print(f"   ✅ Muestra {i+1} analizada correctamente")
                
            except Exception as e:
                print(f"   ❌ Error analizando muestra {i+1}: {type(e).__name__}: {e}")
                import traceback
                print(f"   🔍 Stack trace:")
                traceback.print_exc()
                continue
    
    # Calcular estadísticas promedio
    if hrm_stats['total_samples'] > 0:
        print(f"   📊 Muestras analizadas: {hrm_stats['total_samples']}")
        
        # L-steps promedio por capa
        for layer_idx, steps_list in enumerate(hrm_stats['avg_l_steps_per_layer']):
            if steps_list:
                avg_steps = sum(steps_list) / len(steps_list)
                print(f"   🔄 Capa {layer_idx}: Promedio L-steps = {avg_steps:.1f}")
        
        # Convergencia promedio
        for layer_idx, conv_list in enumerate(hrm_stats['convergence_rates']):
            if conv_list:
                conv_rate = sum(conv_list) / len(conv_list) * 100
                print(f"   ✅ Capa {layer_idx}: Convergencia = {conv_rate:.1f}%")
        
        # Ponder loss promedio
        if hrm_stats['ponder_losses']:
            avg_ponder = sum(hrm_stats['ponder_losses']) / len(hrm_stats['ponder_losses'])
            print(f"   💭 Ponder loss promedio: {avg_ponder:.4f}")
    
    return hrm_stats

def evaluate_perplexity(model, tokenizer, test_texts: List[str], device: str = 'cuda', max_samples: int = 100):
    """Evaluar perplejidad del modelo en textos de prueba"""
    print(f"\n📈 Evaluando perplejidad en {min(len(test_texts), max_samples)} muestras...")
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    valid_samples = 0
    
    if TQDM_AVAILABLE:
        progress = tqdm(test_texts[:max_samples], desc="Evaluando")
    else:
        progress = test_texts[:max_samples]
    
    with torch.no_grad():
        for text in progress:
            try:
                # Tokenizar texto
                tokens = tokenizer.encode(text, max_length=128, truncation=True)
                if len(tokens) < 10:  # Muy corto
                    continue
                
                # Crear input y labels
                input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
                labels = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).to(device)
                
                # Forward pass
                outputs = model(input_ids, labels=labels)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item() * len(tokens)
                    total_tokens += len(tokens)
                    valid_samples += 1
                
                if TQDM_AVAILABLE and isinstance(progress, tqdm):
                    current_ppl = math.exp(total_loss / max(total_tokens, 1)) if total_tokens > 0 else float('inf')
                    progress.set_postfix({'perplexity': f'{current_ppl:.2f}', 'valid': valid_samples})
            
            except Exception as e:
                continue  # Saltar muestras problemáticas
    
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        print(f"   📊 Resultados:")
        print(f"   - Muestras válidas: {valid_samples}/{min(len(test_texts), max_samples)}")
        print(f"   - Tokens evaluados: {total_tokens:,}")
        print(f"   - Loss promedio: {avg_loss:.4f}")
        print(f"   - Perplejidad: {perplexity:.2f}")
        
        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'valid_samples': valid_samples,
            'total_tokens': total_tokens
        }
    else:
        print("   ❌ No se pudieron evaluar muestras válidas")
        return None

def benchmark_inference_speed(model, tokenizer, device: str = 'cuda', sequence_lengths: List[int] = [32, 64, 128]):
    """Benchmark de velocidad de inferencia"""
    print(f"\n⚡ Benchmark de velocidad de inferencia...")
    
    model.eval()
    results = {}
    
    for seq_len in sequence_lengths:
        print(f"   📏 Probando secuencias de longitud {seq_len}...")
        
        # Crear input de prueba
        input_ids = torch.randint(1, len(tokenizer), (1, seq_len)).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(10):
                start_time = time.time()
                outputs = model(input_ids)
                torch.cuda.synchronize() if device == 'cuda' else None
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        tokens_per_sec = seq_len / avg_time
        
        results[seq_len] = {
            'avg_time': avg_time,
            'tokens_per_sec': tokens_per_sec,
            'times': times
        }
        
        print(f"      ⏱️ Tiempo promedio: {avg_time*1000:.1f}ms")
        print(f"      🚀 Tokens/seg: {tokens_per_sec:.1f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Probar modelo HRM Micro 10M")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Ruta al modelo guardado (ej: ./hrm-micro-10m-hf/final_model)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Dispositivo a usar")
    parser.add_argument("--test_generation", action="store_true", default=True,
                       help="Probar generación de texto")
    parser.add_argument("--test_perplexity", action="store_true", default=False,
                       help="Evaluar perplejidad")
    parser.add_argument("--test_hrm", action="store_true", default=True,
                       help="Analizar comportamiento HRM")
    parser.add_argument("--benchmark", action="store_true", default=False,
                       help="Benchmark de velocidad")
    parser.add_argument("--max_length", type=int, default=50,
                       help="Longitud máxima para generación")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Temperatura para generación")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k para generación")
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"🖥️ Usando dispositivo: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Cargar modelo y tokenizador
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        if model is None or tokenizer is None:
            print("❌ No se pudo cargar el modelo")
            return
        
        model = model.to(device)
        print(f"✅ Modelo cargado correctamente en {device}")
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # Prompts de prueba
    test_prompts = [
        "The future of artificial intelligence",
        "Once upon a time",
        "In a world where technology",
        "The quick brown fox",
        "Python is a programming language"
    ]
    
    # Textos para análisis
    test_texts = [
        "The rapid advancement of artificial intelligence has transformed how we interact with technology in our daily lives.",
        "Machine learning algorithms are becoming increasingly sophisticated, enabling computers to perform complex tasks.",
        "Natural language processing has made significant strides in recent years, with models capable of understanding context.",
        "Deep learning networks require vast amounts of data to train effectively and produce accurate results.",
        "The future of AI holds great promise for solving complex problems across various domains and industries."
    ]
    
    print(f"\n🎯 Iniciando pruebas del modelo HRM...")
    print("=" * 60)
    
    # Test 1: Generación de texto
    if args.test_generation:
        generation_results = test_text_generation(
            model, tokenizer, test_prompts,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        
        # Mostrar estadísticas de generación
        successful_generations = [r for r in generation_results if 'error' not in r]
        if successful_generations:
            avg_tokens_per_sec = sum(r['tokens_per_sec'] for r in successful_generations) / len(successful_generations)
            print(f"\n📊 Estadísticas de generación:")
            print(f"   - Generaciones exitosas: {len(successful_generations)}/{len(generation_results)}")
            print(f"   - Velocidad promedio: {avg_tokens_per_sec:.1f} tokens/s")
    
    # Test 2: Análisis HRM
    if args.test_hrm:
        hrm_stats = analyze_hrm_behavior(model, tokenizer, test_texts, device)
    
    # Test 3: Perplejidad
    if args.test_perplexity:
        perplexity_results = evaluate_perplexity(model, tokenizer, test_texts, device)
    
    # Test 4: Benchmark
    if args.benchmark:
        benchmark_results = benchmark_inference_speed(model, tokenizer, device)
        
        print(f"\n📊 Resumen benchmark:")
        for seq_len, results in benchmark_results.items():
            print(f"   Longitud {seq_len}: {results['tokens_per_sec']:.1f} tokens/s")
    
    print(f"\n✅ Pruebas completadas!")

if __name__ == "__main__":
    main()