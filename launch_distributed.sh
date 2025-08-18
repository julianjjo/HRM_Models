#!/bin/bash
# Script para lanzar entrenamiento distribuido en múltiples GPUs

# Configurar número de GPUs (cambiar según tu sistema)
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=2  # Fallback si no se puede detectar
fi

echo "🚀 Lanzando entrenamiento distribuido con $NUM_GPUS GPUs..."

# Método 1: torchrun (recomendado para PyTorch >= 1.10)
if command -v torchrun &> /dev/null; then
    echo "Usando torchrun..."
    torchrun --nproc_per_node=$NUM_GPUS hrm_training_nano_25m.py
else
    # Método 2: torch.distributed.launch (PyTorch < 1.10)
    echo "Usando torch.distributed.launch..."
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS hrm_training_nano_25m.py
fi