#!/usr/bin/env python3
"""
Script para descargar y analizar un archivo del dataset Dolma
"""

import requests
import gzip
import json
import os
from urllib.parse import urlparse

def download_dolma_file(url, output_path):
    """Descargar un archivo de Dolma"""
    print(f"🔄 Descargando: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Descarga completa: {output_path}")
        return True
    
    except requests.RequestException as e:
        print(f"❌ Error descargando {url}: {e}")
        return False

def analyze_dolma_structure(file_path, max_samples=10):
    """Analizar estructura del archivo Dolma"""
    print(f"\n📊 Analizando estructura de: {file_path}")
    
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            sample_count = 0
            total_chars = 0
            
            for line_num, line in enumerate(f, 1):
                if line_num > max_samples:
                    break
                
                try:
                    data = json.loads(line.strip())
                    sample_count += 1
                    
                    print(f"\n--- Muestra {sample_count} ---")
                    print(f"Claves disponibles: {list(data.keys())}")
                    
                    # Analizar campos de texto
                    if 'text' in data:
                        text = data['text']
                        total_chars += len(text)
                        print(f"Texto (primeros 200 chars): {text[:200]}...")
                        print(f"Longitud del texto: {len(text)} caracteres")
                    
                    # Otros metadatos
                    for key, value in data.items():
                        if key != 'text':
                            if isinstance(value, str) and len(value) > 50:
                                print(f"{key}: {value[:50]}...")
                            else:
                                print(f"{key}: {value}")
                
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error JSON en línea {line_num}: {e}")
                    continue
            
            avg_chars = total_chars / sample_count if sample_count > 0 else 0
            print(f"\n📈 Estadísticas:")
            print(f"   Muestras analizadas: {sample_count}")
            print(f"   Promedio caracteres por muestra: {avg_chars:.0f}")
            print(f"   Total caracteres: {total_chars}")
            
    except Exception as e:
        print(f"❌ Error analizando archivo: {e}")

def main():
    """Función principal"""
    # URL del primer archivo inglés de la lista
    dolma_url = "https://olmo-data.org/dolma-v1_5r1/cc_en_head/cc_en_head-0001.json.gz"
    
    # Crear directorio para datos
    data_dir = "dolma_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Nombre del archivo local
    filename = os.path.basename(urlparse(dolma_url).path)
    local_path = os.path.join(data_dir, filename)
    
    print(f"🎯 Dataset Dolma - Analizador de Estructura")
    print(f"URL: {dolma_url}")
    print(f"Archivo local: {local_path}")
    
    # Descargar solo si no existe
    if not os.path.exists(local_path):
        if not download_dolma_file(dolma_url, local_path):
            print("❌ Fallo la descarga. Terminando.")
            return
    else:
        print(f"✅ Archivo ya existe: {local_path}")
    
    # Analizar estructura
    analyze_dolma_structure(local_path, max_samples=5)
    
    # Información del archivo
    file_size = os.path.getsize(local_path)
    print(f"\n📁 Información del archivo:")
    print(f"   Tamaño: {file_size / 1024 / 1024:.1f} MB")
    print(f"   Ruta: {local_path}")

if __name__ == "__main__":
    main()