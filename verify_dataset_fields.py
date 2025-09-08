#!/usr/bin/env python3
"""
Script para verificar los campos del dataset alpindale/light-novels
sin streaming para ver la estructura completa
"""

from datasets import load_dataset
import json

def verify_dataset_fields():
    print("🔍 Verificando campos del dataset alpindale/light-novels")
    print("=" * 60)
    
    try:
        # Cargar dataset SIN streaming para poder inspeccionar la estructura
        print("📥 Cargando dataset (no streaming)...")
        dataset = load_dataset("alpindale/light-novels", streaming=False)
        
        print(f"✅ Dataset cargado exitosamente")
        print(f"📊 Splits disponibles: {list(dataset.keys())}")
        
        # Verificar cada split
        for split_name, split_data in dataset.items():
            print(f"\n--- SPLIT: {split_name.upper()} ---")
            print(f"📈 Número de ejemplos: {len(split_data)}")
            
            if len(split_data) > 0:
                # Obtener el primer ejemplo
                first_example = split_data[0]
                print(f"🔑 Campos disponibles: {list(first_example.keys())}")
                
                # Mostrar información detallada de cada campo
                for field_name, field_value in first_example.items():
                    field_type = type(field_value).__name__
                    
                    if isinstance(field_value, str):
                        field_length = len(field_value)
                        field_preview = field_value[:100] + "..." if len(field_value) > 100 else field_value
                        print(f"   • {field_name}: {field_type} (longitud: {field_length})")
                        print(f"     Vista previa: '{field_preview}'")
                    else:
                        print(f"   • {field_name}: {field_type} = {field_value}")
                
                # Buscar el campo de texto principal
                print(f"\n🎯 Análisis de campos de texto:")
                text_fields = []
                for field_name, field_value in first_example.items():
                    if isinstance(field_value, str) and len(field_value) > 50:
                        text_fields.append({
                            'name': field_name,
                            'length': len(field_value),
                            'preview': field_value[:200]
                        })
                
                if text_fields:
                    print(f"   📝 Campos de texto encontrados: {len(text_fields)}")
                    for field in text_fields:
                        print(f"   • {field['name']}: {field['length']} caracteres")
                        print(f"     Preview: '{field['preview']}...'")
                        print()
                else:
                    print(f"   ⚠️  No se encontraron campos de texto largos")
                    
        # Mostrar algunas muestras adicionales
        if 'train' in dataset and len(dataset['train']) > 1:
            print(f"\n--- MUESTRAS ADICIONALES DEL TRAIN ---")
            for i in range(min(3, len(dataset['train']))):
                example = dataset['train'][i]
                print(f"\nMuestra {i+1}:")
                for field_name, field_value in example.items():
                    if isinstance(field_value, str):
                        preview = field_value[:150] + "..." if len(field_value) > 150 else field_value
                        print(f"   {field_name}: '{preview}'")
                    else:
                        print(f"   {field_name}: {field_value}")
                        
    except Exception as e:
        print(f"❌ Error cargando el dataset: {e}")
        print("\n💡 Intentando con streaming=True para obtener información básica...")
        
        try:
            # Fallback con streaming
            dataset_stream = load_dataset("alpindale/light-novels", streaming=True)
            print(f"✅ Dataset cargado con streaming")
            print(f"📊 Splits disponibles: {list(dataset_stream.keys())}")
            
            # Obtener una muestra del train
            train_iter = iter(dataset_stream['train'])
            first_example = next(train_iter)
            
            print(f"\n--- ESTRUCTURA DEL DATASET (STREAMING) ---")
            print(f"🔑 Campos disponibles: {list(first_example.keys())}")
            
            for field_name, field_value in first_example.items():
                field_type = type(field_value).__name__
                if isinstance(field_value, str):
                    field_length = len(field_value)
                    field_preview = field_value[:100] + "..." if len(field_value) > 100 else field_value
                    print(f"   • {field_name}: {field_type} (longitud: {field_length})")
                    print(f"     Vista previa: '{field_preview}'")
                else:
                    print(f"   • {field_name}: {field_type} = {field_value}")
                    
        except Exception as stream_error:
            print(f"❌ Error también con streaming: {stream_error}")

if __name__ == "__main__":
    verify_dataset_fields()