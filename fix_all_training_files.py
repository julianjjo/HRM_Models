#!/usr/bin/env python3
"""
Script para aplicar correcciones de distributed training y learning rate a todos los archivos de entrenamiento
"""

import os
import re
import glob

def fix_learning_rate(content):
    """Corregir learning rates que sean muy altos"""
    # Buscar LEARNING_RATE_MAX = 2e-4 y cambiar a 1e-5
    content = re.sub(
        r'LEARNING_RATE_MAX = 2e-4([^0-9])',
        r'LEARNING_RATE_MAX = 1e-5  # Reducido urgentemente para evitar explosión de gradientes\1',
        content
    )
    
    # Buscar LEARNING_RATE_MIN = 2e-6 y cambiar a 1e-7
    content = re.sub(
        r'LEARNING_RATE_MIN = 2e-6([^0-9])',
        r'LEARNING_RATE_MIN = 1e-7  # Mínimo reducido proporcionalmente\1',
        content
    )
    
    # Buscar NEW_LEARNING_RATE = 2e-4 y cambiar a 1e-5
    content = re.sub(
        r'NEW_LEARNING_RATE = 2e-4([^0-9])',
        r'NEW_LEARNING_RATE = 1e-5   # LR reducido urgentemente para evitar explosión de gradientes\1',
        content
    )
    
    return content

def add_sharding(content):
    """Agregar sharding para distributed training"""
    sharding_code = '''
    # ### DISTRIBUTED TRAINING SHARDING ###
    # Aplicar sharding para distributed training con IterableDataset
    if is_distributed and world_size > 1:
        print(f"🔀 Aplicando sharding para distributed training (rank {rank}/{world_size})")
        
        # Shard tanto train como validation para distributed training
        for split_name in tokenized_splits.keys():
            if is_iterable_dataset(tokenized_splits[split_name]):
                print(f"   📊 Sharding {split_name}: GPU {rank} procesará 1/{world_size} de los datos")
                tokenized_splits[split_name] = tokenized_splits[split_name].shard(
                    num_shards=world_size, 
                    index=rank
                )
            else:
                print(f"   📊 {split_name} no es IterableDataset, usar DistributedSampler en DataLoader")
'''
    
    # Buscar patrón donde agregar el sharding (después de tokenización)
    patterns = [
        (r'(print\(f"✅ {split_name} tokenizado como Dataset regular"\)\n)', r'\1' + sharding_code),
        (r'(print\(f"✅ {split_name} tokenizado como IterableDataset"\)\n)', r'\1' + sharding_code),
        (r'(\.with_format\("torch"\)\n)', r'\1' + sharding_code)
    ]
    
    # Solo agregar si no existe ya
    if "DISTRIBUTED TRAINING SHARDING" not in content:
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content, count=1)
                break
    
    return content

def fix_file(filepath):
    """Aplicar todas las correcciones a un archivo"""
    print(f"🔧 Procesando: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Aplicar correcciones
        content = fix_learning_rate(content)
        content = add_sharding(content)
        
        # Solo escribir si hay cambios
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Actualizado: {filepath}")
            return True
        else:
            print(f"⚪ Sin cambios: {filepath}")
            return False
            
    except Exception as e:
        print(f"❌ Error procesando {filepath}: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 Iniciando correcciones de distributed training y learning rate...")
    
    # Buscar todos los archivos de entrenamiento
    patterns = [
        "hrm_training_*.py",
        "kaggle_*Training.py"
    ]
    
    files_fixed = 0
    total_files = 0
    
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            total_files += 1
            if fix_file(filepath):
                files_fixed += 1
    
    print(f"\n📊 Resumen:")
    print(f"   Archivos procesados: {total_files}")
    print(f"   Archivos modificados: {files_fixed}")
    print(f"   Archivos sin cambios: {total_files - files_fixed}")
    print("\n✅ ¡Correcciones completadas!")

if __name__ == "__main__":
    main()