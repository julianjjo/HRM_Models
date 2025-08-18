#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de lanzamiento distribuido para HRM-Text1 Multi-GPU Training

Uso:
    # Para 8 GPUs en un solo nodo:
    python launch_distributed.py --script hrm_training_small_100m.py --gpus 8

    # Para múltiples nodos:
    # Nodo 0 (principal):
    python launch_distributed.py --script hrm_training_small_100m.py --gpus 8 --nodes 2 --node-rank 0 --master-addr IP_NODO_0
    # Nodo 1:
    python launch_distributed.py --script hrm_training_small_100m.py --gpus 8 --nodes 2 --node-rank 1 --master-addr IP_NODO_0

    # Con configuración personalizada:
    python launch_distributed.py --script hrm_training_large_1b.py --gpus 8 --master-port 29500
"""

import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Lanzar entrenamiento distribuido HRM-Text1')
    
    # Argumentos básicos
    parser.add_argument('--script', required=True, help='Script de entrenamiento a ejecutar')
    parser.add_argument('--gpus', type=int, default=8, help='Número de GPUs por nodo (default: 8)')
    
    # Argumentos para múltiples nodos
    parser.add_argument('--nodes', type=int, default=1, help='Número total de nodos (default: 1)')
    parser.add_argument('--node-rank', type=int, default=0, help='Rank del nodo actual (default: 0)')
    parser.add_argument('--master-addr', default='localhost', help='Dirección IP del nodo principal (default: localhost)')
    parser.add_argument('--master-port', type=int, default=29500, help='Puerto principal (default: 29500)')
    
    # Argumentos adicionales
    parser.add_argument('--backend', default='nccl', choices=['nccl', 'gloo'], help='Backend de comunicación (default: nccl)')
    parser.add_argument('--verbose', action='store_true', help='Mostrar output verbose')
    
    args = parser.parse_args()
    
    # Verificar que el script existe
    if not os.path.exists(args.script):
        print(f"❌ Error: Script {args.script} no encontrado")
        sys.exit(1)
    
    # Verificar disponibilidad de GPUs
    try:
        import torch
        available_gpus = torch.cuda.device_count()
        if available_gpus < args.gpus:
            print(f"❌ Error: Se requieren {args.gpus} GPUs pero solo hay {available_gpus} disponibles")
            sys.exit(1)
    except ImportError:
        print("⚠️  Advertencia: No se pudo verificar disponibilidad de GPUs (PyTorch no encontrado)")
    
    # Configurar variables de entorno
    world_size = args.nodes * args.gpus
    
    print(f"🚀 Configuración de lanzamiento distribuido:")
    print(f"   📦 Script: {args.script}")
    print(f"   🎯 GPUs por nodo: {args.gpus}")
    print(f"   🌐 Nodos totales: {args.nodes}")
    print(f"   🔢 World size: {world_size}")
    print(f"   📍 Nodo actual: {args.node_rank}")
    print(f"   🌍 Master address: {args.master_addr}")
    print(f"   🚪 Master port: {args.master_port}")
    print(f"   🔗 Backend: {args.backend}")
    print()
    
    # Preparar comando torchrun
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={args.gpus}",
        f"--nnodes={args.nodes}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        args.script
    ]
    
    # Configurar variables de entorno adicionales
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': ','.join(map(str, range(args.gpus))),
        'OMP_NUM_THREADS': '1',  # Importante para evitar conflictos de threading
        'NCCL_DEBUG': 'INFO' if args.verbose else 'WARN',
    })
    
    # Configuraciones específicas para H200
    if args.gpus == 8:
        env.update({
            'NCCL_IB_DISABLE': '0',  # Habilitar InfiniBand si está disponible
            'NCCL_P2P_DISABLE': '0',  # Habilitar P2P entre GPUs
            'NCCL_SHM_DISABLE': '0',  # Habilitar shared memory
        })
        print("🔧 Configuraciones optimizadas para 8x H200 aplicadas")
    
    print("🎬 Ejecutando comando:")
    print(" ".join(cmd))
    print()
    
    try:
        # Ejecutar comando
        process = subprocess.run(cmd, env=env, check=True)
        print("\n✅ Entrenamiento completado exitosamente")
        return process.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️  Entrenamiento interrumpido por el usuario")
        return 130

if __name__ == "__main__":
    sys.exit(main())