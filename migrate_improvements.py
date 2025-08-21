#!/usr/bin/env python3
"""
Script para migrar mejoras del modelo micro a todos los modelos más grandes
"""

import os
import re

# Funciones mejoradas del modelo micro
IMPROVED_FUNCTIONS = '''
def save_complete_model_for_inference(model, tokenizer, output_dir):
    """
    Guarda el modelo completo en formato compatible con hrm_llm_inference.py
    Crea config.json, pytorch_model.bin y archivos del tokenizer
    """
    try:
        # Obtener el modelo sin wrapper DDP/DataParallel
        model_to_save = model
        if hasattr(model, '_orig_mod'):
            model_to_save = model._orig_mod  # DDP
        elif hasattr(model, 'module'):
            model_to_save = model.module     # DataParallel
        
        print(f"\\n💾 Guardando modelo completo para inferencia en: {output_dir}")
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Guardar config.json
        config_dict = model_to_save.config.to_dict() if hasattr(model_to_save.config, 'to_dict') else vars(model_to_save.config)
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ config.json guardado")
        
        # 2. Guardar pytorch_model.bin
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), model_path)
        print(f"✅ pytorch_model.bin guardado")
        
        # 3. Guardar tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
            print(f"✅ Tokenizer guardado")
        
        # 4. Guardar generation_config.json
        generation_config = {
            'max_length': model_to_save.config.block_size,
            'do_sample': True,
            'temperature': 0.8,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'pad_token_id': getattr(tokenizer, 'pad_token_id', None) if tokenizer else None,
            'eos_token_id': getattr(tokenizer, 'eos_token_id', None) if tokenizer else None,
        }
        gen_config_path = os.path.join(output_dir, "generation_config.json")
        with open(gen_config_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(generation_config, f, indent=2, ensure_ascii=False)
        print(f"✅ generation_config.json guardado")
        
        print(f"✅ Modelo completo guardado exitosamente para hrm_llm_inference.py")
        return True
        
    except Exception as e:
        print(f"❌ Error al guardar modelo completo: {e}")
        return False

def save_checkpoint_distributed(model, optimizer, scheduler, scaler, epoch, global_step, 
                               best_val_loss, patience_counter, num_training_steps, 
                               checkpoint_path, is_distributed=False, rank=0):
    """
    Guarda checkpoint de manera compatible con entrenamiento distribuido y single GPU
    Solo el proceso de rank 0 guarda el checkpoint para evitar conflictos
    """
    # Solo el proceso principal (rank 0) debe guardar checkpoints
    if is_distributed and 'RANK' in os.environ and rank != 0:
        # Los demás procesos esperan a que rank 0 termine
        if hasattr(dist, 'is_initialized') and dist.is_initialized():
            dist.barrier()
        return
    
    try:
        # Obtener el modelo sin wrapper DDP/DataParallel
        model_to_save = model
        if hasattr(model, '_orig_mod'):
            model_to_save = model._orig_mod  # DDP
        elif hasattr(model, 'module'):
            model_to_save = model.module     # DataParallel
        
        print(f"\\n💾 Guardando checkpoint en paso {global_step}...")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Crear checkpoint temporal primero para atomicidad
        temp_path = checkpoint_path + '.tmp'
        
        checkpoint_data = {
            'epoch': epoch,
            'step': global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'num_training_steps': num_training_steps,
            'distributed_training': is_distributed,
            'timestamp': time.time(),
            # Datos adicionales para inferencia
            'model_config': model_to_save.config.to_dict() if hasattr(model_to_save.config, 'to_dict') else vars(model_to_save.config),
            'tokenizer_info': {
                'tokenizer_class': 'T5Tokenizer',
                'pretrained_model_name': T5_TOKENIZER_REPO,
                'vocab_size': model_to_save.config.vocab_size,
                'pad_token_id': getattr(tokenizer, 'pad_token_id', None) if 'tokenizer' in globals() else None,
                'eos_token_id': getattr(tokenizer, 'eos_token_id', None) if 'tokenizer' in globals() else None,
                'bos_token_id': getattr(tokenizer, 'bos_token_id', None) if 'tokenizer' in globals() else None,
                'unk_token_id': getattr(tokenizer, 'unk_token_id', None) if 'tokenizer' in globals() else None,
            },
            'generation_config': {
                'max_length': model_to_save.config.block_size,
                'do_sample': True,
                'temperature': 0.8,
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.1,
                'pad_token_id': getattr(tokenizer, 'pad_token_id', None) if 'tokenizer' in globals() else None,
                'eos_token_id': getattr(tokenizer, 'eos_token_id', None) if 'tokenizer' in globals() else None,
            },
            'training_metadata': {
                'dataset_name': ACTIVE_DATASET if 'ACTIVE_DATASET' in globals() else None,
                'block_size': model_to_save.config.block_size,
                'learning_rate': LEARNING_RATE_MAX if 'LEARNING_RATE_MAX' in globals() else None,
                'batch_size': BATCH_SIZE if 'BATCH_SIZE' in globals() else None,
                'grad_accumulation_steps': GRAD_ACCUM_STEPS if 'GRAD_ACCUM_STEPS' in globals() else None,
                'seed': SEED if 'SEED' in globals() else None,
            }
        }
        
        # Guardar en archivo temporal
        torch.save(checkpoint_data, temp_path)
        
        # Mover archivo temporal al final (operación atómica)
        os.rename(temp_path, checkpoint_path)
        
        print(f"✅ Checkpoint guardado exitosamente en {checkpoint_path}")
        
        # Sincronizar con otros procesos si es distribuido
        if is_distributed and 'RANK' in os.environ and hasattr(dist, 'is_initialized') and dist.is_initialized():
            dist.barrier()
            
    except Exception as e:
        print(f"❌ Error al guardar checkpoint: {e}")
        
        # Limpiar archivo temporal si existe
        temp_path = checkpoint_path + '.tmp'
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        
        # Sincronizar con otros procesos incluso si hay error
        if is_distributed and 'RANK' in os.environ and hasattr(dist, 'is_initialized') and dist.is_initialized():
            dist.barrier()
        
        raise e

def load_checkpoint_distributed(checkpoint_path, model, optimizer, scheduler, scaler, 
                               device, is_distributed=False, rank=0):
    """
    Carga checkpoint de manera compatible con entrenamiento distribuido y single GPU
    """
    if not os.path.exists(checkpoint_path):
        print("--- No se encontró checkpoint. Empezando entrenamiento desde cero. ---")
        return False, 0, 0, float('inf'), 0
    
    try:
        print(f"--- Reanudando entrenamiento desde el checkpoint: {checkpoint_path} ---")
        
        # Sincronizar antes de cargar si es distribuido
        if is_distributed and 'RANK' in os.environ and hasattr(dist, 'is_initialized') and dist.is_initialized():
            dist.barrier()
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Obtener el modelo sin wrapper para cargar estado
        model_to_load = model
        if hasattr(model, '_orig_mod'):
            model_to_load = model._orig_mod  # DDP
        elif hasattr(model, 'module'):
            model_to_load = model.module     # DataParallel
        
        # Cargar estados
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Extraer información del checkpoint
        start_epoch = checkpoint['epoch']
        start_step = checkpoint.get('step', 0)
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint.get('patience_counter', 0)
        
        print(f"✅ Checkpoint cargado exitosamente")
        print(f"   📊 Época: {start_epoch + 1}, Paso: {start_step}")
        print(f"   🏆 Mejor pérdida de validación: {best_val_loss:.4f}")
        
        return True, start_epoch, start_step, best_val_loss, patience_counter
        
    except Exception as e:
        print(f"❌ Error al cargar checkpoint: {e}")
        print("--- Empezando entrenamiento desde cero. ---")
        return False, 0, 0, float('inf'), 0

'''

# Archivos de modelos a migrar
MODEL_FILES = [
    "hrm_training_nano_25m.py",
    "hrm_training_medium_350m.py", 
    "hrm_training_large_1b.py"
]

def add_functions_before_setup_distributed(file_path):
    """Agrega las funciones mejoradas antes de setup_distributed"""
    print(f"Migrando funciones a {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar la línea de setup_distributed
    setup_distributed_pattern = r'(# Configuración distribuida\ndef setup_distributed\(\):)'
    
    if re.search(setup_distributed_pattern, content):
        # Insertar las funciones antes de setup_distributed
        new_content = re.sub(
            setup_distributed_pattern,
            IMPROVED_FUNCTIONS + '\n\n# Configuración distribuida\ndef setup_distributed():',
            content
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ Funciones agregadas a {file_path}")
        return True
    else:
        print(f"❌ No se encontró setup_distributed en {file_path}")
        return False

if __name__ == "__main__":
    print("🚀 Iniciando migración de mejoras...")
    
    for model_file in MODEL_FILES:
        file_path = f"/Users/julianmican/Documents/HRM/HRM-Text/{model_file}"
        if os.path.exists(file_path):
            add_functions_before_setup_distributed(file_path)
        else:
            print(f"❌ Archivo no encontrado: {file_path}")
    
    print("✅ Migración completada!")