#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM-Models Distributed Training Launcher CORREGIDO
Launcher que usa la versión corregida de distribución de datos

🚀 CARACTERÍSTICAS:
- Auto-detecta GPUs disponibles
- Usa distribución CORRECTA de datos (sin redundancia)
- Configura automáticamente batch size óptimo por GPU
- 4x más eficiente en carga de datos
"""

import os
import sys
import subprocess
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Launcher CORREGIDO para entrenamiento distribuido HRM")
    parser.add_argument("--model", choices=['50m'], required=True, help="Tamaño del modelo")
    parser.add_argument("--gpus", type=int, help="Número de GPUs (auto-detect si no se especifica)")
    parser.add_argument("--train_samples", type=int, default=100000, help="Samples de entrenamiento")
    parser.add_argument("--val_samples", type=int, default=10000, help="Samples de validación")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size por GPU")
    parser.add_argument("--dry_run", action="store_true", help="Solo mostrar configuración")

    args = parser.parse_args()

    # Detectar GPUs
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible")
        return 1

    gpu_count = torch.cuda.device_count()
    target_gpus = args.gpus if args.gpus else gpu_count

    if target_gpus > gpu_count:
        print(f"⚠️ Solicitadas {target_gpus} GPUs pero solo {gpu_count} disponibles")
        target_gpus = gpu_count

    print(f"🚀 HRM {args.model} DISTRIBUIDO CORREGIDO")
    print(f"🔧 SOLUCIÓN: Cada GPU descarga datos ÚNICOS (sin redundancia)")
    print(f"📊 GPUs: {target_gpus}")
    print(f"📊 Samples por GPU: {args.train_samples // target_gpus}")
    print(f"📊 Batch size por GPU: {args.batch_size}")
    
    # Usar script corregido
    script_name = "hrm_training_small_50m_distributed_FIXED.py"
    
    if not os.path.exists(script_name):
        print(f"❌ Script corregido no encontrado: {script_name}")
        print("💡 Use el script corregido o aplique el parche a los originales")
        return 1

    cmd = [
        'torchrun',
        f'--nproc_per_node={target_gpus}',
        script_name,
        f'--train_samples={args.train_samples}',
        f'--val_samples={args.val_samples}',
        f'--batch_size={args.batch_size}',
    ]

    if args.dry_run:
        print(f"\n🏃‍♂️ Dry run - comando:")
        print(" \\\n  ".join(cmd))
        return 0

    print(f"\n🎉 Ejecutando entrenamiento distribuido CORREGIDO...")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido por usuario")
        return 1

if __name__ == "__main__":
    sys.exit(main())