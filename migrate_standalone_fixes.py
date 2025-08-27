#!/usr/bin/env python3
"""
Script para migrar todas las mejoras del modelo 50M a los demás modelos standalone
"""

import os
import re
import shutil
from typing import Dict, List

# Lista de archivos a actualizar
STANDALONE_FILES = [
    'hrm_training_nano_25m_standalone.py',
    'hrm_training_micro_10m_standalone.py', 
    'hrm_training_medium_100m_standalone.py',
    'hrm_training_medium_350m_standalone.py',
    'hrm_training_large_1b_standalone.py'
]

# Directorio de trabajo
WORK_DIR = '/Users/julianmican/Documents/Personal/HRM_Models'

def create_backup(filepath: str):
    """Crear backup del archivo original"""
    backup_path = filepath + '.backup'
    shutil.copy2(filepath, backup_path)
    print(f"📝 Backup creado: {backup_path}")

def read_file(filepath: str) -> str:
    """Leer archivo"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(filepath: str, content: str):
    """Escribir archivo"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def apply_huggingface_hub_import(content: str) -> str:
    """Añadir import de HfApi con try/except"""
    # Buscar el bloque de imports después de torch
    pattern = r'(from torch\.optim import AdamW\n)'
    
    huggingface_import = """
# Hugging Face Hub imports
try:
    from huggingface_hub import HfApi
    HF_API_AVAILABLE = True
except ImportError:
    HF_API_AVAILABLE = False
    print("⚠️ WARNING: huggingface_hub no está disponible. No se podrá subir al Hub.")
"""
    
    if 'HF_API_AVAILABLE' not in content:
        content = re.sub(pattern, r'\1' + huggingface_import, content)
        print("  ✅ Import de HfApi añadido")
    
    return content

def apply_total_train_samples_fix(content: str) -> str:
    """Añadir default values para TOTAL_TRAIN_SAMPLES"""
    pattern = r'TOTAL_TRAIN_SAMPLES = DATASET_INFO\["train_samples"\]'
    replacement = 'TOTAL_TRAIN_SAMPLES = DATASET_INFO["train_samples"] or 100000  # Default para datasets locales'
    
    if ' or 100000' not in content:
        content = re.sub(pattern, replacement, content)
        print("  ✅ Fix TOTAL_TRAIN_SAMPLES añadido")
    
    pattern = r'TOTAL_VAL_SAMPLES = DATASET_INFO\["val_samples"\]'
    replacement = 'TOTAL_VAL_SAMPLES = DATASET_INFO["val_samples"] or 10000  # Default para datasets locales'
    
    if ' or 10000' not in content:
        content = re.sub(pattern, replacement, content)
        print("  ✅ Fix TOTAL_VAL_SAMPLES añadido")
    
    return content

def apply_hf_api_check(content: str) -> str:
    """Añadir verificación de HF_API_AVAILABLE"""
    pattern = r'if HF_TOKEN:'
    replacement = 'if HF_TOKEN and HF_API_AVAILABLE:'
    
    if 'HF_TOKEN and HF_API_AVAILABLE' not in content:
        content = re.sub(pattern, replacement, content)
        print("  ✅ Verificación HF_API_AVAILABLE añadida")
    
    # También añadir el mensaje de error mejorado
    pattern = r'else:\s*print\("\\n⚠️  No se encontró HF_TOKEN\. El modelo solo se guardó localmente\."\)'
    replacement = '''else:
        if not HF_TOKEN:
            print("\\n⚠️  No se encontró HF_TOKEN. El modelo solo se guardó localmente.")
            print("Para subir a Hugging Face Hub, configura la variable de entorno HF_TOKEN.")
        elif not HF_API_AVAILABLE:
            print("\\n⚠️ huggingface_hub no disponible. El modelo no se subirá al Hub.")
            print("Para subir al Hub, instala: pip install huggingface_hub")'''
    
    if 'elif not HF_API_AVAILABLE' not in content:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        print("  ✅ Mensajes de error mejorados añadidos")
    
    return content

def apply_state_dict_messages(content: str) -> str:
    """Mejorar mensajes de missing keys"""
    pattern = r'if missing_keys:\s*print\(f"⚠️ Claves faltantes: \{missing_keys\}"\)'
    replacement = 'if missing_keys:\\n                        print(f"📝 Claves faltantes (normal para carga parcial): {len(missing_keys)} claves")'
    
    if 'normal para carga parcial' not in content:
        content = re.sub(pattern, replacement, content)
        print("  ✅ Mensajes de state dict mejorados")
    
    pattern = r'if unexpected_keys:\s*print\(f"⚠️ Claves inesperadas: \{unexpected_keys\}"\)'
    replacement = 'if unexpected_keys:\\n                        print(f"📝 Claves inesperadas: {len(unexpected_keys)} claves")'
    
    if '📝 Claves inesperadas:' not in content:
        content = re.sub(pattern, replacement, content)
    
    return content

def apply_checkpoint_dataset_support(content: str) -> str:
    """Añadir soporte para checkpoint_dataset"""
    if '"checkpoint_dataset"' not in content:
        # Buscar el final de DATASETS_CONFIG - patrón más general
        pattern = r'(\s+"[^"]+": \{[^}]*"type": "kaggle"[^}]*\}\s*)(}\s*\n\n# Añadir las mezclas)'
        
        checkpoint_config = ''',
    "checkpoint_dataset": {
        "name": "local_checkpoint",
        "config": None,
        "train_samples": None,  # Se detectará automáticamente
        "val_samples": None,  # Se creará del 10% del entrenamiento
        "repo_suffix": "Checkpoint",
        "description": "Dataset local desde checkpoint_dataset/ (JSONL)",
        "type": "local",  # Identificador para datasets locales
        "path": "./checkpoint_dataset"  # Ruta local a los archivos JSONL
    }'''
        
        replacement = r'\1' + checkpoint_config + r'\2'
        count = len(re.findall(pattern, content, re.DOTALL))
        if count > 0:
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            print("  ✅ Soporte para checkpoint_dataset añadido")
        else:
            print("  ⚠️ Patrón para checkpoint_dataset no encontrado")
    
    return content

def apply_dataset_loading_fix(content: str) -> str:
    """Aplicar fix para carga de datasets locales con soporte multi-GPU"""
    if 'elif DATASET_INFO.get("type") == "local":' not in content:
        # Buscar donde añadir el bloque local
        pattern = r'(elif DATASET_INFO\.get\("name"\) == "human_conversations":.*?streaming=True\)\s*\n)'
        
        local_block = '''
    elif DATASET_INFO.get("type") == "local":
        # Lógica especial para datasets locales JSONL
        local_path = DATASET_INFO.get("path", "./checkpoint_dataset")
        print(f"📁 Cargando dataset local desde: {local_path}")
        
        try:
            import glob
            import json
            
            # Buscar archivos JSONL en el directorio
            jsonl_files = glob.glob(os.path.join(local_path, "*.jsonl"))
            
            if not jsonl_files:
                raise FileNotFoundError(f"No se encontraron archivos .jsonl en {local_path}")
            
            print(f"📄 Encontrados {len(jsonl_files)} archivos JSONL")
            jsonl_files.sort()  # Asegurar orden consistente
            
            # Cargar usando datasets de Hugging Face
            from datasets import load_dataset as hf_load_dataset
            
            # Estimar tamaño del dataset
            sample_count = 0
            for file_path in jsonl_files[:3]:  # Procesar solo algunos archivos para estimar
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            sample_count += 1
                            if sample_count > 1000:  # Estimar con muestra
                                break
                if sample_count > 1000:
                    break
            
            # Estimar total y crear splits
            total_files = len(jsonl_files)
            estimated_total = sample_count * total_files // 3 if sample_count > 0 else 10000
            val_size = max(100, estimated_total // 10)  # 10% para validación, mínimo 100
            
            print(f"📈 Estimación: ~{estimated_total:,} muestras, {val_size:,} para validación")
            
            # Detectar entorno multi-GPU para evitar conflictos de acceso a archivos
            if is_distributed and world_size > 1:
                print(f"🔧 Multi-GPU detectado (rank {rank}/{world_size}). Usando streaming para evitar conflictos...")
                # Crear dos streams separados para evitar conflictos con skip/take
                train_dataset = hf_load_dataset('json', data_files={'train': jsonl_files}, streaming=True)['train']
                val_dataset = hf_load_dataset('json', data_files={'train': jsonl_files}, streaming=True)['train']
                
                raw_datasets = {
                    'train': train_dataset.skip(val_size),
                    'validation': val_dataset.take(val_size)
                }
                print(f"✅ Splits creados con streaming para multi-GPU (rank {rank})")
            else:
                # Usar dataset sin streaming para single GPU/CPU (mejor control)
                print("🔧 Single GPU/CPU. Cargando dataset sin streaming para mejor control de splits...")
                raw_data = hf_load_dataset('json', data_files={'train': jsonl_files}, streaming=False)['train']
                
                # Dividir manualmente
                split_point = len(raw_data) - val_size
                raw_datasets = {
                    'train': raw_data.select(range(split_point)),
                    'validation': raw_data.select(range(split_point, len(raw_data)))
                }
                
                print(f"✅ Splits creados: train={len(raw_datasets['train'])}, validation={len(raw_datasets['validation'])}")
            
            print(f"✅ Dataset local cargado exitosamente desde {local_path}")
            
        except Exception as e:
            print(f"❌ Error cargando dataset local: {e}")
            print("🔄 Cambiando a dataset C4 como respaldo...")
            raw_datasets = load_dataset("allenai/c4", "en", streaming=True)
'''
        
        content = re.sub(pattern, r'\1' + local_block, content, flags=re.DOTALL)
        print("  ✅ Fix de carga de dataset local con soporte multi-GPU añadido")
    else:
        # Si ya existe el bloque local, pero necesitamos actualizar para multi-GPU
        if 'if is_distributed and world_size > 1:' not in content:
            # Buscar y reemplazar la sección de carga de dataset local
            pattern = r'# Usar dataset sin streaming para datasets locales \(mejor control\).*?print\(f"✅ Splits creados: train=\{len\(raw_datasets\[\'train\'\]\)\}, validation=\{len\(raw_datasets\[\'validation\'\]\)\}"\)'
            
            replacement = '''# Detectar entorno multi-GPU para evitar conflictos de acceso a archivos
            if is_distributed and world_size > 1:
                print(f"🔧 Multi-GPU detectado (rank {rank}/{world_size}). Usando streaming para evitar conflictos...")
                # Crear dos streams separados para evitar conflictos con skip/take
                train_dataset = hf_load_dataset('json', data_files={'train': jsonl_files}, streaming=True)['train']
                val_dataset = hf_load_dataset('json', data_files={'train': jsonl_files}, streaming=True)['train']
                
                raw_datasets = {
                    'train': train_dataset.skip(val_size),
                    'validation': val_dataset.take(val_size)
                }
                print(f"✅ Splits creados con streaming para multi-GPU (rank {rank})")
            else:
                # Usar dataset sin streaming para single GPU/CPU (mejor control)
                print("🔧 Single GPU/CPU. Cargando dataset sin streaming para mejor control de splits...")
                raw_data = hf_load_dataset('json', data_files={'train': jsonl_files}, streaming=False)['train']
                
                # Dividir manualmente
                split_point = len(raw_data) - val_size
                raw_datasets = {
                    'train': raw_data.select(range(split_point)),
                    'validation': raw_data.select(range(split_point, len(raw_data)))
                }
                
                print(f"✅ Splits creados: train={len(raw_datasets['train'])}, validation={len(raw_datasets['validation'])}")'''
            
            count = len(re.findall(pattern, content, re.DOTALL))
            if count > 0:
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                print(f"  ✅ Fix multi-GPU añadido a dataset local existente")
    
    return content

def apply_shuffle_fix(content: str) -> str:
    """Corregir calls a shuffle() eliminando buffer_size"""
    pattern = r'\.shuffle\(seed=SEED, buffer_size=10_000\)'
    replacement = '.shuffle(seed=SEED)'
    
    count = len(re.findall(pattern, content))
    if count > 0:
        content = re.sub(pattern, replacement, content)
        print(f"  ✅ Corregidos {count} calls a shuffle()")
    
    return content

def apply_debug_dataloader(content: str) -> str:
    """Añadir debug para DataLoader"""
    pattern = r'(val_loader = DataLoader\(tokenized_splits\["validation"\], \*\*val_kwargs\)\n)'
    
    debug_block = '''    
    # Debug: Verificar que el dataset tiene datos
    print("🔍 Verificando que el dataset tiene datos...")
    try:
        sample_iter = iter(train_loader)
        first_batch = next(sample_iter)
        print(f"✅ Primera muestra obtenida. Batch shape: {first_batch['input_ids'].shape}")
    except StopIteration:
        print("❌ ERROR: El train_loader está vacío!")
    except Exception as e:
        print(f"❌ ERROR obteniendo muestra: {e}")

'''
    
    if '🔍 Verificando que el dataset tiene datos' not in content:
        content = re.sub(pattern, r'\1' + debug_block, content)
        print("  ✅ Debug de DataLoader añadido")
    
    return content

def apply_causal_lm_output_tuple_fix(content: str) -> str:
    """Cambiar CausalLMOutput a tupla para compatibilidad con DataParallel"""
    # Buscar el bloque completo de CausalLMOutput
    pattern = r'([ \t]*)class CausalLMOutput:.*?return CausalLMOutput\(loss=loss, logits=logits, past_key_values=None\)'
    
    replacement = r'''\1# Para compatibilidad con DataParallel, retornar tupla simple
\1# en lugar de objeto custom que causa problemas en el gather()
\1return (loss, logits, None)  # (loss, logits, past_key_values)'''
    
    count = len(re.findall(pattern, content, re.DOTALL))
    if count > 0:
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        print("  ✅ CausalLMOutput cambiado a tupla para DataParallel")
    
    return content

def apply_outputs_tuple_access(content: str) -> str:
    """Cambiar acceso a outputs.loss por outputs[0] para tuplas"""
    # Cambiar outputs.loss por outputs[0] 
    pattern1 = r'loss = outputs\.loss'
    replacement1 = r'loss = outputs[0]  # outputs es ahora una tupla (loss, logits, past_key_values)'
    
    count1 = len(re.findall(pattern1, content))
    if count1 > 0:
        content = re.sub(pattern1, replacement1, content)
        print(f"  ✅ Actualizados {count1} accesos a outputs.loss")
    
    # También buscar outputs.loss / GRAD_ACCUM_STEPS
    pattern2 = r'loss = outputs\.loss / GRAD_ACCUM_STEPS'
    replacement2 = r'loss = outputs[0] / GRAD_ACCUM_STEPS  # outputs es ahora una tupla (loss, logits, past_key_values)'
    
    count2 = len(re.findall(pattern2, content))
    if count2 > 0:
        content = re.sub(pattern2, replacement2, content)
        print(f"  ✅ Actualizados {count2} accesos a outputs.loss con GRAD_ACCUM_STEPS")
    
    return content

def apply_jsonl_fallback(content: str) -> str:
    """Añadir fallback manual para cargar JSONL sin datasets"""
    if 'Implementando cargador manual de JSONL' in content:
        print("  ⚠️ Fallback JSONL ya existe")
        return content
    
    # Buscar el bloque de error de datasets
    pattern = r'(        except Exception as e:\s+print\(f"❌ Error cargando dataset local: \{e\}"\)\s+print\("🔄 Cambiando a dataset C4 como respaldo\.\.\."\)\s+raw_datasets = load_dataset\("allenai/c4", "en", streaming=True\))'
    
    fallback_code = '''        except Exception as e:
            print(f"❌ Error cargando dataset local con HuggingFace datasets: {e}")
            print("🔄 Intentando fallback manual para JSONL...")
            
            # Fallback manual: cargar JSONL sin librería datasets
            try:
                print("📁 Implementando cargador manual de JSONL...")
                all_data = []
                
                # Leer todos los archivos JSONL
                for file_idx, file_path in enumerate(jsonl_files):
                    print(f"📖 Leyendo archivo {file_idx+1}/{len(jsonl_files)}: {os.path.basename(file_path)}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    # Normalizar campo de texto
                                    text = data.get('text', data.get('content', data.get('message', str(data))))
                                    if text and len(text.strip()) > 10:  # Filtrar textos muy cortos
                                        all_data.append({'text': text.strip()})
                                        if len(all_data) % 10000 == 0:
                                            print(f"   📊 Procesados {len(all_data):,} registros...")
                                except json.JSONDecodeError:
                                    continue  # Saltar líneas con JSON inválido
                    
                    # Limitar para evitar memoria excesiva en desarrollo
                    if len(all_data) >= 100000:
                        print(f"⚠️ Limitando dataset a 100,000 muestras para evitar uso excesivo de memoria")
                        break
                
                print(f"✅ Cargados {len(all_data):,} registros desde archivos JSONL")
                
                if len(all_data) < 100:
                    raise ValueError(f"Dataset muy pequeño: solo {len(all_data)} muestras válidas")
                
                # Crear splits train/validation
                val_size = max(100, len(all_data) // 10)  # 10% para validación, mínimo 100
                train_size = len(all_data) - val_size
                
                # Shuffle para mezclar los datos
                import random
                random.shuffle(all_data)
                
                # Crear objetos simulando datasets de Hugging Face
                class SimpleDataset:
                    def __init__(self, data):
                        self.data = data
                    def __len__(self):
                        return len(self.data)
                    def __iter__(self):
                        return iter(self.data)
                    def __getitem__(self, idx):
                        return self.data[idx]
                    def map(self, func, *args, **kwargs):
                        # Simulación simple de .map()
                        return SimpleDataset(self.data)  # El tokenizado se hace después
                
                raw_datasets = {
                    'train': SimpleDataset(all_data[:train_size]),
                    'validation': SimpleDataset(all_data[train_size:train_size + val_size])
                }
                
                print(f"✅ Dataset manual creado: train={len(raw_datasets['train'])} val={len(raw_datasets['validation'])}")
                
            except Exception as fallback_error:
                print(f"❌ Fallback manual también falló: {fallback_error}")
                print("🔄 Cambiando a dataset C4 como último respaldo...")
                raw_datasets = load_dataset("allenai/c4", "en", streaming=True)'''
    
    count = len(re.findall(pattern, content, re.DOTALL))
    if count > 0:
        content = re.sub(pattern, fallback_code, content, flags=re.DOTALL)
        print("  ✅ Fallback manual JSONL añadido")
    else:
        print("  ⚠️ Patrón para fallback JSONL no encontrado")
    
    return content

def apply_all_fixes(filepath: str):
    """Aplicar todos los fixes a un archivo"""
    print(f"\n🔧 Procesando: {os.path.basename(filepath)}")
    
    # Crear backup
    create_backup(filepath)
    
    # Leer contenido
    content = read_file(filepath)
    
    # Aplicar todos los fixes
    content = apply_huggingface_hub_import(content)
    content = apply_total_train_samples_fix(content)
    content = apply_hf_api_check(content)
    content = apply_state_dict_messages(content)
    content = apply_checkpoint_dataset_support(content)
    content = apply_dataset_loading_fix(content)
    content = apply_shuffle_fix(content)
    content = apply_debug_dataloader(content)
    content = apply_causal_lm_output_tuple_fix(content)
    content = apply_outputs_tuple_access(content)
    content = apply_jsonl_fallback(content)
    
    # Escribir archivo actualizado
    write_file(filepath, content)
    print(f"✅ {os.path.basename(filepath)} actualizado")

def main():
    """Función principal"""
    print("🚀 Iniciando migración de mejoras a todos los modelos standalone...")
    
    # Cambiar al directorio de trabajo
    os.chdir(WORK_DIR)
    
    # Procesar cada archivo
    for filename in STANDALONE_FILES:
        filepath = os.path.join(WORK_DIR, filename)
        if os.path.exists(filepath):
            apply_all_fixes(filepath)
        else:
            print(f"⚠️ Archivo no encontrado: {filename}")
    
    print(f"\n✅ Migración completada para {len(STANDALONE_FILES)} archivos")
    print("📝 Se han creado backups con extensión .backup")
    print("🧪 Recomendado: Probar cada modelo después de la migración")

if __name__ == "__main__":
    main()