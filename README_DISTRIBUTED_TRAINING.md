# HRM Models - Distributed Training Guide

Este documento explica cómo usar el sistema de entrenamiento distribuido para los modelos HRM 50M y 100M en múltiples GPUs.

## 🚀 Características Principales

- **Entrenamiento Distribuido**: Usa `torch.distributed` con DistributedDataParallel (DDP)
- **Auto-configuración**: Detecta automáticamente GPUs y configura batch sizes óptimos
- **Balanceado de Carga**: Distribuye datos y cómputo equitativamente entre GPUs
- **Sincronización de Gradientes**: Optimizada para modelos HRM con adaptive computation
- **Memoria Eficiente**: Gradient checkpointing para modelo 100M
- **Monitoreo Avanzado**: Progress bars, métricas HRM, y uso de memoria GPU

## 📋 Requisitos del Sistema

### Hardware Mínimo
- **Para HRM 50M**: 2+ GPUs con 8GB+ VRAM cada una
- **Para HRM 100M**: 2+ GPUs con 12GB+ VRAM cada una

### Hardware Recomendado
- **Para HRM 50M**: 4-8 GPUs con 12GB+ VRAM cada una
- **Para HRM 100M**: 4-8 GPUs con 16GB+ VRAM cada una

### Software
```bash
# Dependencias principales
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tokenizers tqdm

# Dependencias opcionales para mejor rendimiento
pip install flash-attn  # Para Flash Attention (opcional)
```

## 🏗️ Estructura de Archivos

```
HRM_Models/
├── hrm_training_small_50m_distributed.py      # Script para modelo 50M
├── hrm_training_medium_100m_distributed.py    # Script para modelo 100M  
├── launch_distributed_training.py             # Launcher automático
├── hf_tokenizer_wrapper_simple.py            # Wrapper tokenizador
└── README_DISTRIBUTED_TRAINING.md            # Esta guía
```

## 🚀 Uso Rápido

### Método 1: Launcher Automático (Recomendado)

El launcher detecta automáticamente tu configuración de GPUs y optimiza los parámetros:

```bash
# Entrenamiento 50M con auto-configuración
python launch_distributed_training.py --model 50m

# Entrenamiento 100M con 4 GPUs específicas
python launch_distributed_training.py --model 100m --gpus 4

# Con configuración personalizada
python launch_distributed_training.py --model 50m --gpus 2 --train_samples 200000

# Dry run para ver la configuración sin entrenar
python launch_distributed_training.py --model 100m --dry_run
```

### Método 2: Torchrun Manual

Para control completo, puedes usar torchrun directamente:

```bash
# HRM 50M con 2 GPUs
torchrun --nproc_per_node=2 hrm_training_small_50m_distributed.py \
    --batch_size 6 --train_samples 100000

# HRM 100M con 4 GPUs
torchrun --nproc_per_node=4 hrm_training_medium_100m_distributed.py \
    --batch_size 3 --train_samples 200000 --learning_rate 1e-5
```

## ⚙️ Configuraciones Recomendadas

### HRM Small 50M (~50M parámetros)

| GPUs | Batch Size/GPU | Batch Total | VRAM/GPU | Throughput |
|------|----------------|-------------|----------|------------|
| 2    | 6-8            | 12-16       | 8-12GB   | ~2000 tok/s |
| 4    | 4-6            | 16-24       | 8-10GB   | ~4000 tok/s |
| 8    | 2-4            | 16-32       | 6-8GB    | ~6000 tok/s |

```bash
# Configuración óptima para diferentes GPU counts
# 2 GPUs
torchrun --nproc_per_node=2 hrm_training_small_50m_distributed.py --batch_size 6

# 4 GPUs  
torchrun --nproc_per_node=4 hrm_training_small_50m_distributed.py --batch_size 4

# 8 GPUs
torchrun --nproc_per_node=8 hrm_training_small_50m_distributed.py --batch_size 3
```

### HRM Medium 100M (~100M parámetros)

| GPUs | Batch Size/GPU | Batch Total | VRAM/GPU | Throughput |
|------|----------------|-------------|----------|------------|
| 2    | 3-4            | 6-8         | 12-16GB  | ~1000 tok/s |
| 4    | 2-3            | 8-12        | 10-12GB  | ~2000 tok/s |
| 8    | 1-2            | 8-16        | 8-10GB   | ~3000 tok/s |

```bash
# Configuración óptima para diferentes GPU counts
# 2 GPUs (recomendado 16GB+ VRAM)
torchrun --nproc_per_node=2 hrm_training_medium_100m_distributed.py --batch_size 3

# 4 GPUs (recomendado 12GB+ VRAM)
torchrun --nproc_per_node=4 hrm_training_medium_100m_distributed.py --batch_size 2

# 8 GPUs (mínimo 8GB VRAM)
torchrun --nproc_per_node=8 hrm_training_medium_100m_distributed.py --batch_size 1
```

## 📊 Parámetros de Entrenamiento

### Parámetros Comunes

```bash
# Dataset
--dataset_name allenai/c4          # Dataset base
--dataset_config en               # Idioma
--train_samples 100000            # Samples entrenamiento
--val_samples 10000               # Samples validación

# Tokenizador
--tokenizer openai-community/gpt2 # Tokenizador base
--max_text_length 2000            # Longitud máxima texto
--min_text_length 50              # Longitud mínima texto

# Entrenamiento
--epochs 3                        # Número de épocas
--learning_rate 3e-5              # Learning rate
--batch_size 4                    # Batch size por GPU
--save_steps 500                  # Frecuencia de guardado
--eval_steps 100                  # Frecuencia de evaluación

# Optimización
--fast_mode                       # Descarga completa dataset
--no_streaming                    # No usar streaming
```

### Parámetros Específicos HRM 50M

```bash
# Configuración optimizada para 50M
--learning_rate 3e-5
--train_samples 100000
--val_samples 10000
--max_text_length 2000
--save_steps 500
--eval_steps 100
```

### Parámetros Específicos HRM 100M

```bash
# Configuración optimizada para 100M
--learning_rate 1e-5              # LR más conservador
--train_samples 200000            # Más datos
--val_samples 20000
--max_text_length 2048            # Context más largo
--min_text_length 100             # Textos mínimos más largos
--save_steps 1000                 # Checkpoints menos frecuentes
--eval_steps 200
```

## 🔧 Arquitectura y Optimizaciones

### Distributed Data Parallel (DDP)
- Usa `torch.nn.parallel.DistributedDataParallel` en lugar de `DataParallel`
- Sincronización de gradientes optimizada con `find_unused_parameters=True` para HRM
- Comunicación NCCL para GPUs, Gloo para CPU

### División de Datos
- Cada GPU procesa un subset diferente del dataset
- `DistributedSampler` maneja la distribución automáticamente
- Sincronización entre épocas con `sampler.set_epoch(epoch)`

### Memoria y Rendimiento
- **Gradient Checkpointing**: Activado para modelo 100M
- **Mixed Precision**: AMP con `torch.cuda.amp` para mejor rendimiento
- **Pin Memory**: Para transferencias CPU→GPU más rápidas
- **Gradient Clipping**: Más agresivo para modelos grandes

### Características HRM Específicas
- **Adaptive Computation**: Q-learning para halting automático
- **Deep Supervision**: Pérdidas intermedias de capas HRM
- **Ponder Loss**: Regularización computacional
- **Hierarchical Cycles**: H_cycles y L_cycles configurables

## 📈 Monitoreo y Métricas

### Métricas de Entrenamiento
- **Loss Components**: Main loss, Deep supervision, Ponder loss, Q-learning loss
- **HRM Metrics**: H-updates, L-steps promedio, convergencia rate
- **GPU Metrics**: Memoria usada/reservada por GPU
- **Training Speed**: Samples/segundo total (suma de todas las GPUs)

### Progress Tracking
- Barra de progreso con `tqdm` (solo proceso principal)
- Logging detallado cada 50 steps
- Evaluación periódica con early stopping
- Checkpoints automáticos y best model saving

## 🐛 Troubleshooting

### Problemas Comunes

#### 1. Out of Memory (OOM)
```bash
# Solución: Reducir batch size
--batch_size 2  # En lugar de 4

# O usar gradient checkpointing (100M tiene por defecto)
# 50M: Editar config.gradient_checkpointing=True
```

#### 2. NCCL Timeout/Hanging
```bash
# Solución: Variables de entorno
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1  # Si hay problemas con InfiniBand
```

#### 3. Port Already in Use
```bash
# Solución: Cambiar puerto master
torchrun --nproc_per_node=4 --master_port=29501 script.py
```

#### 4. NaN/Inf en Loss
- Los scripts tienen protección contra NaN incorporada
- Se salta optimización si se detectan NaN
- Loss components se estabilizan automáticamente

### Debugging

```bash
# Modo verbose para más información
python launch_distributed_training.py --model 50m --verbose

# Ver configuración sin entrenar
python launch_distributed_training.py --model 50m --dry_run

# Variables de entorno útiles
export CUDA_LAUNCH_BLOCKING=1  # Para debugging
export NCCL_DEBUG=INFO         # Para debugging NCCL
export TOKENIZERS_PARALLELISM=false  # Evitar warnings
```

## 📊 Benchmarks de Rendimiento

### HRM 50M (4x RTX 4090)
- **Throughput**: ~4,000 tokens/segundo
- **Memory/GPU**: ~8GB VRAM
- **Batch Size Total**: 16 (4 per GPU)
- **Training Time**: ~6 horas para 100K samples

### HRM 100M (4x A100 40GB)
- **Throughput**: ~2,000 tokens/segundo  
- **Memory/GPU**: ~12GB VRAM
- **Batch Size Total**: 8 (2 per GPU)
- **Training Time**: ~12 horas para 200K samples

## 🎯 Mejores Prácticas

### 1. Configuración de Hardware
- Usar GPUs del mismo tipo para mejor balancing
- Asegurar conectividad PCIe/NVLink para comunicación rápida
- Monitorear temperaturas durante entrenamientos largos

### 2. Configuración de Datos
- Usar `--no_streaming` para datasets grandes (mejor rendimiento)
- Ajustar `--train_samples` basado en memoria disponible
- Usar textos más largos para modelo 100M (`--max_text_length 2048`)

### 3. Optimización de Entrenamiento
- Empezar con configuraciones recomendadas y ajustar gradualmente
- Usar launcher automático para configuración inicial óptima
- Monitorear métricas HRM (H-updates, L-steps) para validar funcionamiento

### 4. Checkpoints y Resuming
- Los checkpoints se guardan solo en el proceso principal (rank 0)
- Para resumir entrenamiento, cargar modelo desde checkpoint guardado
- Best model se guarda automáticamente basado en validation loss

## 📞 Soporte y Contribuciones

Para reportar problemas o sugerir mejoras:
1. Verificar que el problema no esté en la sección de Troubleshooting
2. Incluir información de sistema: GPU models, VRAM, driver versions
3. Incluir logs completos y configuración usada
4. Mencionar si el problema es específico a distributed training

---

🚀 **¡Happy Training!** 🚀