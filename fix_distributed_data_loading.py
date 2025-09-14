# -*- coding: utf-8 -*-
"""
Fix para corregir la distribución de datos en entrenamiento distribuido HRM
Este script corrige el problema donde todos los procesos cargan los mismos datos
"""

def load_dataset_hf_distributed(tokenizer, split: str = "train", num_samples: int = 1000,
                   dataset_name: str = "allenai/c4", dataset_config: str = "en",
                   text_column: str = "text", min_text_length: int = 50,
                   max_text_length: int = 2000, use_streaming: bool = True, 
                   fast_mode: bool = False, rank: int = 0, world_size: int = 1):
    """Cargar dataset distribuido CORRECTAMENTE - cada rank carga solo su porción"""
    try:
        from datasets import load_dataset
        from tqdm import tqdm

        # CORRECCIÓN: Calcular samples por rank ANTES de cargar
        samples_per_rank = num_samples // world_size
        # Cada rank debe procesar un rango diferente del dataset
        start_sample = rank * samples_per_rank
        end_sample = start_sample + samples_per_rank
        
        # El último rank toma cualquier sample restante
        if rank == world_size - 1:
            end_sample = num_samples

        if rank == 0:  # Solo el proceso principal imprime información
            print(f"📥 Cargando dataset '{dataset_name}' DISTRIBUIDO:")
            print(f"   Total samples: {num_samples}")
            print(f"   Samples por rank: {samples_per_rank}")
            print(f"   Rank {rank}: samples {start_sample}-{end_sample}")

        # OPTIMIZACIÓN: Para datasets grandes, usar modo no-streaming es más rápido
        if fast_mode and num_samples > 50000:
            if rank == 0:
                print("⚡ Fast mode: Usando dataset no-streaming para mejor rendimiento...")
            use_streaming = False

        # Cargar dataset
        if use_streaming:
            dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        else:
            if rank == 0:
                print("📦 Descargando dataset completo (más rápido para lotes grandes)...")
            dataset = load_dataset(dataset_name, dataset_config, split=split)

        texts = []
        processed_count = 0
        current_sample = 0

        # CORRECCIÓN: Solo el rank 0 muestra progreso global, otros ranks son silenciosos
        if rank == 0 and hasattr(tqdm, '__name__'):
            progress_desc = f"Procesando {dataset_name} (Rank {rank})"
            progress = tqdm(enumerate(dataset), desc=progress_desc, total=end_sample)
        else:
            progress = enumerate(dataset)

        for i, item in progress:
            # CORRECCIÓN: Saltar samples hasta llegar a nuestro rango
            if current_sample < start_sample:
                current_sample += 1
                continue
                
            # CORRECCIÓN: Parar cuando lleguemos al final de nuestro rango
            if current_sample >= end_sample:
                break

            text = item.get(text_column, '')
            if isinstance(text, list):
                text = ' '.join(text)  # Para datasets con texto en listas

            if text and len(text.strip()) >= min_text_length:
                # Procesar y limpiar texto
                text = text.strip()[:max_text_length]
                texts.append(text)
                processed_count += 1

                if rank == 0 and hasattr(progress, 'set_postfix'):
                    progress.set_postfix({
                        'válidos': processed_count,
                        'sample': current_sample,
                        'ratio': f'{processed_count/(current_sample-start_sample+1)*100:.1f}%'
                    })
            
            current_sample += 1

        if rank == 0 and hasattr(progress, 'close'):
            progress.close()

        # Solo el rank 0 imprime estadísticas finales
        if rank == 0:
            print(f"✅ Rank {rank} procesó {processed_count} textos válidos de samples {start_sample}-{end_sample}")
        
        # IMPORTANTE: Cada rank ahora tiene datos ÚNICOS, no necesitamos división adicional
        return texts

    except Exception as e:
        if rank == 0:
            print(f"❌ Error cargando dataset HF en rank {rank}: {e}")
        raise RuntimeError(f"Falló carga de dataset {dataset_name} en rank {rank}: {e}") from e

class DistributedTextDatasetFixed:
    """Dataset distribuido CORREGIDO - no divide datos, cada rank ya tiene su porción única"""

    def __init__(self, tokenizer, texts, block_size: int = 256, split_type: str = "train",
                 device=None, max_length: int = 1024, min_text_length: int = 10, 
                 rank: int = 0, world_size: int = 1):
        self.tokenizer = tokenizer
        self.texts = texts  # YA contiene solo los textos únicos para este rank
        self.block_size = block_size
        self.split_type = split_type
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.max_length = max_length
        self.min_text_length = min_text_length
        self.rank = rank
        self.world_size = world_size

        # CORRECCIÓN: NO dividir texts aquí, ya están divididos correctamente
        self.local_texts = texts  # Usar directamente los texts únicos que llegaron

        print(f"📚 Dataset {split_type} Rank {rank}: {len(self.local_texts)} textos ÚNICOS, block_size={block_size}")

    def __iter__(self):
        # Tokenización on-the-fly para cada proceso con SUS textos únicos
        for text in self.local_texts:
            if not text or len(text.strip()) < self.min_text_length:
                continue

            try:
                # Tokenizar con HF
                tokens = self.tokenizer.encode(text, add_special_tokens=True,
                                             max_length=self.max_length, truncation=True)

                # Crear chunks
                for i in range(0, len(tokens) - self.block_size + 1, self.block_size):
                    chunk = tokens[i:i + self.block_size]
                    if len(chunk) == self.block_size:
                        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                        labels = torch.tensor(chunk[1:], dtype=torch.long)
                        yield {
                            'input_ids': input_ids,
                            'labels': labels
                        }
            except Exception as e:
                if self.rank == 0:  # Solo el proceso principal imprime errores
                    print(f"⚠️ Error tokenizando texto en rank {self.rank}: {e}")
                continue

# Función helper para validar distribución
def validate_data_distribution(datasets_by_rank, rank, world_size):
    """Validar que cada rank tiene datos diferentes"""
    import hashlib
    
    # Crear hash de los primeros textos para verificar unicidad
    if len(datasets_by_rank[rank]) > 0:
        first_texts = datasets_by_rank[rank][:min(10, len(datasets_by_rank[rank]))]
        text_hash = hashlib.md5(''.join(first_texts).encode()).hexdigest()
        
        print(f"🔍 Rank {rank} validation:")
        print(f"   Textos únicos: {len(datasets_by_rank[rank])}")
        print(f"   Hash primeros textos: {text_hash[:8]}")
        
        # En una implementación real, podrías comparar hashes entre ranks
        # para asegurar que son diferentes
        return text_hash
    return None

# Ejemplo de uso corregido
def example_fixed_usage():
    """Ejemplo de cómo usar la carga distribuida corregida"""
    import torch.distributed as dist
    import os
    
    # Setup distributed (esto sería en el script principal)
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        world_size = 1
    
    # CARGAR DATOS DISTRIBUIDOS CORRECTAMENTE
    train_texts = load_dataset_hf_distributed(
        tokenizer=None,  # Pasarías tu tokenizer aquí
        split="train", 
        num_samples=100000,  # Total samples
        dataset_name="allenai/c4",
        rank=rank, 
        world_size=world_size
    )
    
    # CREAR DATASET SIN DIVISIÓN ADICIONAL
    train_dataset = DistributedTextDatasetFixed(
        tokenizer=None,  # Tu tokenizer
        texts=train_texts,  # Cada rank ya tiene textos únicos
        block_size=512,
        rank=rank,
        world_size=world_size
    )
    
    print(f"✅ Rank {rank}: Dataset configurado con {len(train_texts)} textos únicos")

if __name__ == "__main__":
    print("🔧 Fix para distribución de datos en HRM distributed training")
    print("Este archivo contiene las funciones corregidas.")
    print("Debes aplicar estos cambios a los scripts principales.")