#!/usr/bin/env python3
"""
Standalone Dataset Loader para reemplazar HuggingFace load_dataset
"""

import json
import gzip
import requests
import os
import random
from typing import Iterator, Dict, Any, List
import torch
from torch.utils.data import IterableDataset

class DolmaDatasetLoader:
    """Cargador standalone para dataset Dolma"""
    
    def __init__(self, cache_dir="./dolma_cache", max_files=5):
        self.cache_dir = cache_dir
        self.max_files = max_files
        os.makedirs(cache_dir, exist_ok=True)
        
        # URLs de los archivos Dolma (inglés)
        self.dolma_urls = [
            "https://olmo-data.org/dolma-v1_5r1/cc_en_head/cc_en_head-0001.json.gz",
            "https://olmo-data.org/dolma-v1_5r1/cc_en_head/cc_en_head-0002.json.gz", 
            "https://olmo-data.org/dolma-v1_5r1/cc_en_head/cc_en_head-0003.json.gz",
            "https://olmo-data.org/dolma-v1_5r1/cc_en_head/cc_en_head-0004.json.gz",
            "https://olmo-data.org/dolma-v1_5r1/cc_en_head/cc_en_head-0005.json.gz",
        ]
    
    def download_file(self, url: str, force_download: bool = False) -> str:
        """Descargar archivo si no existe"""
        filename = os.path.basename(url)
        local_path = os.path.join(self.cache_dir, filename)
        
        if os.path.exists(local_path) and not force_download:
            return local_path
            
        print(f"🔄 Descargando: {filename}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (1024*1024*100) == 0:  # Cada 100MB
                        print(f"   📦 Descargado: {downloaded // (1024*1024)} MB")
            
            print(f"✅ Descarga completa: {filename}")
            return local_path
            
        except Exception as e:
            print(f"❌ Error descargando {url}: {e}")
            raise
    
    def load_texts_from_file(self, file_path: str, max_samples: int = None) -> Iterator[str]:
        """Cargar textos de un archivo Dolma"""
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                count = 0
                for line in f:
                    if max_samples and count >= max_samples:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        if 'text' in data and data['text'].strip():
                            yield data['text'].strip()
                            count += 1
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"❌ Error leyendo {file_path}: {e}")
    
    def load_dataset(self, split: str = "train", streaming: bool = True, max_samples_per_file: int = 10000):
        """Cargar dataset completo"""
        print(f"🚀 Cargando dataset Dolma (split: {split}, streaming: {streaming})")
        
        # Descargar archivos necesarios
        files_to_use = self.dolma_urls[:self.max_files]
        local_files = []
        
        for url in files_to_use:
            try:
                local_path = self.download_file(url)
                local_files.append(local_path)
            except Exception as e:
                print(f"⚠️ Saltando archivo por error: {e}")
                continue
        
        if not local_files:
            raise RuntimeError("No se pudieron descargar archivos del dataset")
        
        print(f"✅ Archivos disponibles: {len(local_files)}")
        
        if streaming:
            return DolmaStreamingDataset(local_files, max_samples_per_file, split)
        else:
            # Para modo no-streaming, cargar todo en memoria
            all_texts = []
            for file_path in local_files:
                for text in self.load_texts_from_file(file_path, max_samples_per_file):
                    all_texts.append(text)
            
            return DolmaStaticDataset(all_texts, split)

class DolmaStreamingDataset(IterableDataset):
    """Dataset streaming para Dolma"""
    
    def __init__(self, file_paths: List[str], max_samples_per_file: int, split: str = "train"):
        self.file_paths = file_paths
        self.max_samples_per_file = max_samples_per_file
        self.split = split
        
    def __iter__(self):
        for file_path in self.file_paths:
            try:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    count = 0
                    for line in f:
                        if count >= self.max_samples_per_file:
                            break
                            
                        try:
                            data = json.loads(line.strip())
                            if 'text' in data and data['text'].strip():
                                yield {"text": data['text'].strip()}
                                count += 1
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                print(f"⚠️ Error procesando {file_path}: {e}")
                continue

class DolmaStaticDataset:
    """Dataset estático para Dolma"""
    
    def __init__(self, texts: List[str], split: str = "train"):
        self.texts = texts
        self.split = split
        
        # Dividir en train/validation si es necesario
        if split == "validation":
            # Tomar últimos 10% como validación
            split_idx = int(len(texts) * 0.9)
            self.texts = texts[split_idx:]
        elif split == "train":
            # Tomar primeros 90% como training
            split_idx = int(len(texts) * 0.9)
            self.texts = texts[:split_idx]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {"text": self.texts[idx]}
    
    def __iter__(self):
        for text in self.texts:
            yield {"text": text}

# Función standalone para reemplazar load_dataset de HuggingFace
def load_dataset(dataset_name: str, config: str = None, streaming: bool = True, split: str = "train"):
    """
    Función standalone para cargar datasets sin HuggingFace
    
    Args:
        dataset_name: Nombre del dataset (ej: "allenai/dolma")
        config: Configuración (no usado por ahora)
        streaming: Si usar modo streaming
        split: Split a cargar ("train", "validation")
    """
    
    if dataset_name in ["allenai/dolma", "allenai/c4", "dolma"]:
        # Usar nuestro loader de Dolma
        loader = DolmaDatasetLoader(max_files=3)  # Empezar con 3 archivos
        return loader.load_dataset(split=split, streaming=streaming)
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' no soportado en modo standalone")

# Para compatibilidad con el script principal
def create_simple_datasets():
    """Crear datasets simples para train y validation"""
    
    class SimpleDatasetDict:
        def __init__(self):
            self._datasets = {}
        
        def __getitem__(self, split):
            if split not in self._datasets:
                # Cargar dataset solo cuando se necesite
                self._datasets[split] = load_dataset("allenai/dolma", streaming=True, split=split)
            return self._datasets[split]
        
        def keys(self):
            return ["train", "validation"]
    
    return SimpleDatasetDict()