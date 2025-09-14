#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM-Models Distributed Training Launcher
Script para lanzar entrenamiento distribuido de modelos HRM en múltiples GPUs

🚀 CARACTERÍSTICAS:
- Auto-detecta GPUs disponibles
- Configura automáticamente batch size óptimo por GPU
- Maneja configuración de entorno para torch.distributed
- Lanza entrenamiento con torchrun optimizado
- Soporte para modelos 50M y 100M
"""

import os
import sys
import subprocess
import argparse
import json
import torch
from typing import List, Dict, Tuple

def detect_gpu_configuration() -> Tuple[int, List[Dict]]:
    """Detectar configuración de GPUs disponibles"""
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible")
        return 0, []
    
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        gpu_info.append({
            'id': i,
            'name': props.name,
            'memory_gb': props.total_memory / 1024**3,
            'compute_capability': f"{props.major}.{props.minor}"
        })
    
    return gpu_count, gpu_info

def recommend_batch_size(model_size: str, gpu_count: int, gpu_memory_gb: float) -> int:
    """Recomendar batch size óptimo por GPU basado en modelo y memoria"""
    recommendations = {
        '50m': {
            'min_memory': 8,
            'batch_sizes': {
                1: 8,   # 1 GPU
                2: 6,   # 2 GPUs
                4: 4,   # 4 GPUs
                8: 3,   # 8 GPUs
            }
        },
        '100m': {
            'min_memory': 12,
            'batch_sizes': {
                1: 4,   # 1 GPU
                2: 3,   # 2 GPUs
                4: 2,   # 4 GPUs
                8: 1,   # 8 GPUs
            }
        }
    }
    
    if model_size not in recommendations:
        raise ValueError(f"Modelo {model_size} no soportado")
    
    model_config = recommendations[model_size]
    
    # Verificar memoria mínima
    if gpu_memory_gb < model_config['min_memory']:
        print(f"⚠️ Advertencia: Memoria GPU ({gpu_memory_gb:.1f}GB) menor que recomendada ({model_config['min_memory']}GB) para {model_size}")
    
    # Encontrar batch size recomendado
    if gpu_count in model_config['batch_sizes']:
        return model_config['batch_sizes'][gpu_count]
    else:
        # Extrapolar para configuraciones no estándar
        sorted_configs = sorted(model_config['batch_sizes'].items())
        if gpu_count > sorted_configs[-1][0]:
            return max(1, sorted_configs[-1][1] // 2)  # Reducir para más GPUs
        else:
            return sorted_configs[0][1]  # Usar configuración de 1 GPU

def get_optimal_training_config(model_size: str, gpu_count: int, gpu_memory_gb: float) -> Dict:
    """Obtener configuración de entrenamiento óptima"""
    base_configs = {
        '50m': {
            'learning_rate': 3e-5,
            'warmup_steps': 2000,
            'save_steps': 500,
            'eval_steps': 100,
            'train_samples': 100000,
            'val_samples': 10000,
            'max_text_length': 2000,
            'min_text_length': 50,
        },
        '100m': {
            'learning_rate': 1e-5,
            'warmup_steps': 4000,
            'save_steps': 1000,
            'eval_steps': 200,
            'train_samples': 200000,
            'val_samples': 20000,
            'max_text_length': 2048,
            'min_text_length': 100,
        }
    }
    
    config = base_configs[model_size].copy()
    config['batch_size'] = recommend_batch_size(model_size, gpu_count, gpu_memory_gb)
    config['effective_batch_size'] = config['batch_size'] * gpu_count
    
    # Ajustar learning rate basado en batch size efectivo
    if config['effective_batch_size'] > 16:
        config['learning_rate'] *= (config['effective_batch_size'] / 16) ** 0.5
    
    return config

def create_launch_command(
    model_size: str,
    gpu_count: int,
    config: Dict,
    output_dir: str,
    tokenizer: str = "openai-community/gpt2",
    additional_args: List[str] = None
) -> List[str]:
    """Crear comando de lanzamiento para torchrun"""
    
    script_map = {
        '50m': 'hrm_training_small_50m_distributed.py',
        '100m': 'hrm_training_medium_100m_distributed.py'
    }
    
    if model_size not in script_map:
        raise ValueError(f"Modelo {model_size} no soportado")
    
    script_name = script_map[model_size]
    
    # Base command con torchrun
    cmd = [
        'torchrun',
        f'--nproc_per_node={gpu_count}',
        '--master_port=29500',  # Puerto fijo para evitar conflictos
        script_name,
        f'--tokenizer={tokenizer}',
        f'--output_dir={output_dir}',
        f'--batch_size={config["batch_size"]}',
        f'--learning_rate={config["learning_rate"]}',
        f'--train_samples={config["train_samples"]}',
        f'--val_samples={config["val_samples"]}',
        f'--save_steps={config["save_steps"]}',
        f'--eval_steps={config["eval_steps"]}',
        f'--max_text_length={config["max_text_length"]}',
        f'--min_text_length={config["min_text_length"]}',
        '--fast_mode',
        '--no_streaming',
    ]
    
    # Agregar argumentos adicionales si se proporcionan
    if additional_args:
        cmd.extend(additional_args)
    
    return cmd

def setup_environment():
    """Configurar variables de entorno para entrenamiento distribuido"""
    env_vars = {
        'CUDA_VISIBLE_DEVICES': ','.join(str(i) for i in range(torch.cuda.device_count())),
        'NCCL_DEBUG': 'INFO',  # Para debugging si es necesario
        'TOKENIZERS_PARALLELISM': 'false',  # Evitar warnings de tokenizers
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("🔧 Variables de entorno configuradas:")
    for key, value in env_vars.items():
        print(f"   {key}={value}")

def validate_prerequisites() -> bool:
    """Validar que todas las dependencias estén disponibles"""
    try:
        import transformers
        import datasets
        import tqdm
        print("✅ Dependencias verificadas:")
        print(f"   - transformers: {transformers.__version__}")
        print(f"   - datasets: {datasets.__version__}")
        print(f"   - tqdm disponible")
        return True
    except ImportError as e:
        print(f"❌ Dependencia faltante: {e}")
        print("💡 Instale las dependencias:")
        print("   pip install transformers datasets tqdm")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Lanzador de entrenamiento distribuido HRM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

# Entrenamiento automático 50M con 2 GPUs
python launch_distributed_training.py --model 50m --gpus 2

# Entrenamiento 100M con 4 GPUs y configuración personalizada
python launch_distributed_training.py --model 100m --gpus 4 --train_samples 500000

# Entrenamiento con tokenizador personalizado
python launch_distributed_training.py --model 50m --gpus 2 --tokenizer DeepESP/gpt2-spanish

# Entrenamiento con configuración avanzada
python launch_distributed_training.py --model 100m --gpus 8 --batch_size 1 --learning_rate 5e-6
        """
    )
    
    parser.add_argument("--model", choices=['50m', '100m'], required=True,
                       help="Tamaño del modelo a entrenar")
    parser.add_argument("--gpus", type=int,
                       help="Número de GPUs a usar (auto-detect si no se especifica)")
    parser.add_argument("--output_dir", type=str,
                       help="Directorio de salida (auto-generado si no se especifica)")
    parser.add_argument("--tokenizer", type=str, default="openai-community/gpt2",
                       help="Tokenizador HuggingFace a usar")
    
    # Configuración de entrenamiento
    parser.add_argument("--batch_size", type=int,
                       help="Batch size por GPU (auto-calculado si no se especifica)")
    parser.add_argument("--learning_rate", type=float,
                       help="Learning rate (auto-calculado si no se especifica)")
    parser.add_argument("--train_samples", type=int,
                       help="Número de samples de entrenamiento")
    parser.add_argument("--val_samples", type=int,
                       help="Número de samples de validación")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Número de épocas")
    
    # Opciones avanzadas
    parser.add_argument("--dry_run", action="store_true",
                       help="Solo mostrar configuración sin ejecutar entrenamiento")
    parser.add_argument("--verbose", action="store_true",
                       help="Mostrar información detallada")
    
    args = parser.parse_args()
    
    print("🚀 HRM Distributed Training Launcher")
    print("=" * 50)
    
    # Validar prerequisites
    if not validate_prerequisites():
        sys.exit(1)
    
    # Detectar configuración de GPU
    gpu_count, gpu_info = detect_gpu_configuration()
    
    if gpu_count == 0:
        print("❌ No se detectaron GPUs CUDA")
        sys.exit(1)
    
    print(f"\n🔍 GPUs detectadas: {gpu_count}")
    for gpu in gpu_info:
        print(f"   GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    
    # Determinar número de GPUs a usar
    target_gpus = args.gpus if args.gpus else gpu_count
    if target_gpus > gpu_count:
        print(f"⚠️ Solicitadas {target_gpus} GPUs pero solo {gpu_count} disponibles")
        target_gpus = gpu_count
    
    # Obtener memoria promedio de GPU
    avg_gpu_memory = sum(gpu['memory_gb'] for gpu in gpu_info[:target_gpus]) / target_gpus
    
    print(f"\n📊 Configuración seleccionada:")
    print(f"   Modelo: HRM {args.model.upper()}")
    print(f"   GPUs a usar: {target_gpus}")
    print(f"   Memoria promedio por GPU: {avg_gpu_memory:.1f}GB")
    
    # Obtener configuración óptima
    config = get_optimal_training_config(args.model, target_gpus, avg_gpu_memory)
    
    # Override con argumentos del usuario si se proporcionan
    if args.batch_size:
        config['batch_size'] = args.batch_size
        config['effective_batch_size'] = args.batch_size * target_gpus
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.train_samples:
        config['train_samples'] = args.train_samples
    if args.val_samples:
        config['val_samples'] = args.val_samples
    
    # Generar directorio de salida si no se especifica
    if not args.output_dir:
        args.output_dir = f"./hrm-{args.model}-distributed-{target_gpus}gpu"
    
    print(f"\n⚙️ Configuración de entrenamiento:")
    print(f"   Batch size por GPU: {config['batch_size']}")
    print(f"   Batch size efectivo: {config['effective_batch_size']}")
    print(f"   Learning rate: {config['learning_rate']:.2e}")
    print(f"   Training samples: {config['train_samples']:,}")
    print(f"   Validation samples: {config['val_samples']:,}")
    print(f"   Save steps: {config['save_steps']}")
    print(f"   Eval steps: {config['eval_steps']}")
    print(f"   Directorio salida: {args.output_dir}")
    
    # Verificar que el script existe
    script_map = {
        '50m': 'hrm_training_small_50m_distributed.py',
        '100m': 'hrm_training_medium_100m_distributed.py'
    }
    script_path = script_map[args.model]
    
    if not os.path.exists(script_path):
        print(f"❌ Script de entrenamiento no encontrado: {script_path}")
        print("💡 Asegúrese de que el script esté en el directorio actual")
        sys.exit(1)
    
    # Crear comando de lanzamiento
    additional_args = [f'--epochs={args.epochs}']
    cmd = create_launch_command(
        args.model, target_gpus, config, args.output_dir, 
        args.tokenizer, additional_args
    )
    
    print(f"\n🚀 Comando de lanzamiento:")
    if args.verbose:
        print(" \\\n  ".join(cmd))
    else:
        print(f"   torchrun --nproc_per_node={target_gpus} {script_path} [args...]")
    
    if args.dry_run:
        print("\n🏃‍♂️ Dry run - no se ejecutará el entrenamiento")
        print("\n📋 Para ejecutar manualmente:")
        print(" \\\n  ".join(cmd))
        return
    
    # Configurar entorno
    setup_environment()
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Guardar configuración
    config_file = os.path.join(args.output_dir, 'training_config.json')
    full_config = {
        'model_size': args.model,
        'gpu_count': target_gpus,
        'gpu_info': gpu_info[:target_gpus],
        'training_config': config,
        'command': cmd,
        'tokenizer': args.tokenizer,
    }
    
    with open(config_file, 'w') as f:
        json.dump(full_config, f, indent=2)
    
    print(f"💾 Configuración guardada en: {config_file}")
    
    # Ejecutar entrenamiento
    print(f"\n🎉 ¡Iniciando entrenamiento distribuido!")
    print("=" * 50)
    
    try:
        # Cambiar al directorio del script para imports relativos
        original_dir = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(script_path))
        if script_dir:
            os.chdir(script_dir)
        
        result = subprocess.run(cmd, check=False)
        
        # Restaurar directorio original
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print(f"\n✅ ¡Entrenamiento completado exitosamente!")
            print(f"📁 Modelos guardados en: {args.output_dir}")
        else:
            print(f"\n❌ Entrenamiento falló con código: {result.returncode}")
            sys.exit(result.returncode)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Entrenamiento interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error ejecutando entrenamiento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()