# Comandos Optimizados para Servidor - 10M Samples

## 🚀 Comando Principal para 10M samples

```bash
python hrm_training_micro_10m_standalone_hf.py \
  --train_samples 10000000 \
  --val_samples 50000 \
  --epochs 1 \
  --batch_size 16 \
  --learning_rate 0.000025 \
  --num_workers 8 \
  --prefetch_factor 4 \
  --val_check_interval 5000 \
  --save_steps 10000 \
  --max_grad_norm 0.5 \
  --output_dir ./hrm-micro-10m-server \
  --device auto
```

## ⚡ Configuraciones por tipo de servidor

### **GPU Server (CUDA)**
```bash
python hrm_training_micro_10m_standalone_hf.py \
  --train_samples 10000000 \
  --val_samples 50000 \
  --epochs 1 \
  --batch_size 32 \
  --learning_rate 0.00004 \
  --num_workers 12 \
  --prefetch_factor 8 \
  --val_check_interval 2500 \
  --save_steps 5000 \
  --max_grad_norm 0.5 \
  --device cuda \
  --output_dir ./hrm-micro-10m-gpu
```

### **CPU Server (High cores)**  
```bash
python hrm_training_micro_10m_standalone_hf.py \
  --train_samples 10000000 \
  --val_samples 50000 \
  --epochs 1 \
  --batch_size 8 \
  --learning_rate 0.000015 \
  --cpu_intensive \
  --batch_size_multiplier 2 \
  --num_workers 16 \
  --prefetch_factor 2 \
  --val_check_interval 10000 \
  --save_steps 25000 \
  --max_grad_norm 0.5 \
  --device cpu \
  --output_dir ./hrm-micro-10m-cpu
```

### **Memory Optimized**
```bash
python hrm_training_micro_10m_standalone_hf.py \
  --train_samples 10000000 \
  --val_samples 25000 \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 0.000015 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --val_check_interval 5000 \
  --save_steps 15000 \
  --max_grad_norm 0.5 \
  --no_streaming \
  --output_dir ./hrm-micro-10m-mem
```

## 🎯 Parámetros clave explicados

### **Early Stopping (automático para 10M+)**
- **Patience: 5** - Detiene después de 5 evaluaciones sin mejora
- **Min Delta: 0.001** - Mejora mínima requerida
- **Automático** - Se activa solo para datasets >= 1M samples

### **Learning Rates recomendados (ULTRA-CONSERVADORES)**
- **GPU potente**: 0.000025-0.00004 (20x más bajo que el original)
- **CPU/GPU débil**: 0.000015-0.000025 (33x más bajo que el original)
- **Memoria limitada**: 0.000015

**🚨 CRÍTICO**: Incluso los LR "corregidos" anteriores seguían causando convergencia lineal demasiado rápida. Estos valores ultra-bajos son necesarios para evitar memorización.

### **Batch Sizes por hardware**
- **GPU 24GB+**: batch_size 32-64
- **GPU 8-16GB**: batch_size 16-24  
- **CPU potente**: batch_size 8-16
- **Memoria limitada**: batch_size 4-8

### **Workers y Prefetch**
- **GPU**: num_workers=12, prefetch_factor=8
- **CPU**: num_workers=16, prefetch_factor=2
- **Limitado**: num_workers=4, prefetch_factor=2

## 📊 Monitoreo recomendado

### **Métricas objetivo (CON LEARNING RATE ULTRA-BAJO)**
- **Val Loss objetivo**: 2.8 - 4.5 (realista con LR ultra-conservador)
- **Perplexity objetivo**: 16.0 - 90.0 (sin memorización, evitar < 12.0)
- **Steps estimados**: ~1.25M steps para 10M samples
- **Convergencia**: MUY lenta con mesetas y fluctuaciones naturales

### **Señales de éxito**
```
✅ Val Loss final entre 3.0-4.0 (NO más bajo)
✅ Convergencia con mesetas y fluctuaciones
✅ Perplexity 18.0-40.0 (sin memorización)
✅ Generación de texto coherente sin repetición
✅ Progreso lento pero estable (sin caídas lineales)
```

### **Señales de problema**
```
❌ Val Loss < 2.0 (overfitting severo)
❌ Perplexity < 12.0 (memorización) 
❌ Caída lineal constante sin fluctuaciones
❌ Convergencia demasiado rápida (como tu caso actual)
❌ Early stopping antes de 200k steps
❌ Generación repetitiva o incoherente
```

## 🚨 Comandos de emergencia

### **Reanudar desde checkpoint**
```bash
python hrm_training_micro_10m_standalone_hf.py \
  --resume_from_checkpoint ./hrm-micro-10m-server/checkpoint-50000 \
  --train_samples 10000000 \
  --val_samples 50000
```

### **Reducir recursos si falla**
```bash
# Reducir batch size y workers si hay OOM
python hrm_training_micro_10m_standalone_hf.py \
  --train_samples 10000000 \
  --val_samples 50000 \
  --batch_size 2 \
  --num_workers 2 \
  --prefetch_factor 1
```

## 🎛️ Configuraciones avanzadas

### **Dataset streaming (recomendado)**
- Por defecto está activado
- Usa `--no_streaming` solo si tienes 500GB+ RAM disponible

### **Tokenizer optimizations**
- Se ajustan automáticamente según CPU cores
- Para servidores 32+ cores, el paralelismo será máximo

### **Gradient accumulation** 
- Se calcula automáticamente si batch_size * batch_size_multiplier < 16
- Para simular batch sizes más grandes en hardware limitado

## 📈 Timeline esperado

**10M samples, 1 época:**
- **GPU V100/A100**: 8-12 horas
- **GPU RTX 4090**: 12-18 horas  
- **CPU 32-cores**: 2-4 días
- **CPU 16-cores**: 4-7 días

**Checkpoints recomendados cada:**
- **GPU**: 5,000-10,000 steps
- **CPU**: 15,000-25,000 steps