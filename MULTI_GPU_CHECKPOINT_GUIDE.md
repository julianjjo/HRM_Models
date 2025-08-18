# 🚀 Guía de Checkpoints Multi-GPU para Modelos HRM Grandes

## 📋 Overview

Esta guía explica cómo manejar checkpoints eficientemente con múltiples GPUs y modelos grandes, especialmente para arquitecturas HRM con computación dinámica.

## 🔧 Estrategias de Checkpoint

### 1. **Distributed Data Parallel (DDP)**

#### Configuración Básica
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Inicialización distribuida
def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

# Wrapar modelo en DDP
model = DDP(model, device_ids=[local_rank])
```

#### Guardar Checkpoints (Solo Rank 0)
```python
def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_val_loss, checkpoint_path):
    if dist.get_rank() == 0:  # Solo el proceso maestro guarda
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.module.state_dict(),  # .module para DDP
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'config': model.module.config.__dict__,
            'torch_version': torch.__version__,
            'transformers_version': transformers.__version__
        }
        
        # Guardar con nombre temporal para evitar corrupción
        temp_path = f"{checkpoint_path}.tmp"
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, checkpoint_path)  # Operación atómica
        print(f"✅ Checkpoint guardado: {checkpoint_path}")
```

#### Cargar Checkpoints
```python
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Cargar estado del modelo
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['global_step'], checkpoint['best_val_loss']
    return 0, 0, float('inf')
```

### 2. **Manejo de Memoria para Modelos Grandes**

#### Model Sharding (Experimental)
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Para modelos muy grandes (10B+)
model = FSDP(
    model,
    auto_wrap_policy=lambda module, recurse, nonwrapped_params: isinstance(module, HRMLayer),
    mixed_precision=mp_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    cpu_offload=CPUOffload(offload_params=True)
)
```

#### DeepSpeed Integration
```python
import deepspeed

# Configuración DeepSpeed para modelos grandes
ds_config = {
    "train_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS * world_size,
    "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,  # Stage 2 para modelos 1B-10B
        "cpu_offload": False,
        "contiguous_gradients": True,
        "overlap_comm": True
    }
}

model_engine, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters()
)
```

## 🧠 Problema Específico: HRM + Gradient Checkpointing

### El Desafío
```python
# ❌ PROBLEMA: Computación dinámica causa inconsistencias
def forward(self, x):
    for step in range(max_steps):
        if convergence_check():  # Número variable de iteraciones
            break
        x = self.process(x)
    return x
```

### Solución 1: Checkpointing Selectivo
```python
# ✅ SOLUCIÓN: Checkpoint solo capas estáticas
class HRMWithSelectiveCheckpointing(nn.Module):
    def forward(self, x):
        # Checkpoint capas estáticas (embeddings, attention)
        x = torch.utils.checkpoint.checkpoint(self.embedding_layer, x)
        
        # NO checkpoint para computación dinámica HRM
        for layer in self.hrm_layers:
            x = layer(x)  # Sin checkpoint
            
        # Checkpoint capa final
        x = torch.utils.checkpoint.checkpoint(self.output_layer, x)
        return x
```

### Solución 2: Segmentación por Bloques
```python
# ✅ SOLUCIÓN: Checkpoint bloques de tamaño fijo
def checkpoint_block(self, x, block_layers):
    def block_forward(x_in):
        x_out = x_in
        for layer in block_layers:
            x_out = layer(x_out)
        return x_out
    
    return torch.utils.checkpoint.checkpoint(block_forward, x)
```

## 📊 Configuraciones por Tamaño de Modelo

### Nano Model (25M)
```python
GRADIENT_CHECKPOINTING = False    # No necesario
BATCH_SIZE = 1                   # Por GPU (mínimo)
NUM_GPUS = 1                     # Suficiente
MEMORY_USAGE = ~1-3GB            # Por GPU
```

### Medium Model (350M) 
```python
GRADIENT_CHECKPOINTING = False    # Temporalmente deshabilitado
BATCH_SIZE = 4                   # Por GPU  
NUM_GPUS = 2-4                   # Recomendado
MEMORY_USAGE = ~8GB              # Por GPU
```

### Large Model (1B)
```python
GRADIENT_CHECKPOINTING = False    # Necesita implementación especial
BATCH_SIZE = 1-2                 # Por GPU
NUM_GPUS = 4-8                   # Recomendado
MEMORY_USAGE = ~16GB             # Por GPU
```

## 🛠️ Mejores Prácticas

### 1. **Checkpoint Naming Strategy**
```python
def get_checkpoint_name(epoch, global_step, val_loss):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"checkpoint_epoch{epoch:03d}_step{global_step:06d}_loss{val_loss:.4f}_{timestamp}.pt"
```

### 2. **Cleanup Old Checkpoints**
```python
def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    for checkpoint in checkpoints[keep_last_n:]:
        os.remove(checkpoint)
        print(f"🗑️  Removed old checkpoint: {checkpoint}")
```

### 3. **Validation Before Save**
```python
def validate_checkpoint(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
        
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Missing key: {key}")
                
        print("✅ Checkpoint validation passed")
        return True
    except Exception as e:
        print(f"❌ Checkpoint validation failed: {e}")
        return False
```

## 🚀 Implementación Futura: HRM-Compatible Checkpointing

### Custom Checkpointing para HRM
```python
class HRMCheckpoint:
    @staticmethod
    def checkpoint_hrm_layer(layer, *inputs, **kwargs):
        """
        Custom checkpointing que maneja computación dinámica HRM
        """
        def run_function(*inputs):
            # Ejecutar con número fijo de pasos para consistencia
            return layer(*inputs, force_consistent_computation=True, **kwargs)
        
        return torch.utils.checkpoint.checkpoint(run_function, *inputs)
```

## 📈 Monitoreo de Memoria

### Memory Profiling
```python
def monitor_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3     # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
```

## 🎯 Estado Actual

### Modelos Actuales
- ✅ **Nano (25M)**: Gradient checkpointing deshabilitado - funcional, ultra-optimizado
- ✅ **Medium (350M)**: Gradient checkpointing deshabilitado - pendiente prueba
- ✅ **Large (1B)**: Gradient checkpointing deshabilitado - pendiente prueba

### Próximos Pasos
1. Implementar checkpointing selectivo para modelos grandes
2. Integrar DeepSpeed para modelos 1B+
3. Optimizar estrategia de sharding
4. Crear sistema de checkpoint robusto multi-GPU

---

**Nota**: El gradient checkpointing está temporalmente deshabilitado en todos los modelos debido a incompatibilidad con la computación dinámica HRM. Se requiere implementación especializada para reactivarlo sin errores.