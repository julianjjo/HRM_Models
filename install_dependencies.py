#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de instalación automática de dependencias para HRM-Text1
"""

import subprocess
import sys
import importlib

def install_package(package_name, import_name=None):
    """Instala un paquete si no está disponible"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} ya está instalado")
        return True
    except ImportError:
        print(f"📦 Instalando {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} instalado exitosamente")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando {package_name}: {e}")
            return False

def main():
    """Función principal"""
    print("🚀 INSTALADOR DE DEPENDENCIAS HRM-Text1")
    print("=" * 50)
    
    # Lista de dependencias requeridas
    dependencies = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("huggingface-hub", "huggingface_hub"),
        ("tqdm", "tqdm"),
        ("langdetect", "langdetect"),  # Opcional para filtrado por idioma
        ("numpy", "numpy"),
    ]
    
    # Dependencias opcionales
    optional_dependencies = [
        ("accelerate", "accelerate"),  # Para entrenamiento distribuido
        ("bitsandbytes", "bitsandbytes"),  # Para cuantización
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    print("📋 Instalando dependencias principales...")
    for package_name, import_name in dependencies:
        if install_package(package_name, import_name):
            success_count += 1
        print()
    
    print("📋 Instalando dependencias opcionales...")
    for package_name, import_name in optional_dependencies:
        print(f"¿Instalar {package_name}? (y/n):", end=" ")
        response = input().strip().lower()
        if response in ['y', 'yes', 's', 'si']:
            install_package(package_name, import_name)
        else:
            print(f"⏩ Saltando {package_name}")
        print()
    
    print("=" * 50)
    print(f"📊 RESUMEN: {success_count}/{total_count} dependencias principales instaladas")
    
    if success_count == total_count:
        print("🎉 ¡Todas las dependencias principales están listas!")
        print("✅ Puedes ejecutar el script de entrenamiento o chat interactivo")
    else:
        print("⚠️  Algunas dependencias no se pudieron instalar")
        print("💡 Puedes intentar instalarlas manualmente con pip")
    
    print("\n🔧 COMANDOS ÚTILES:")
    print("   python hrm_llm_training_c4_b.py    # Entrenar modelo")
    print("   python interactive_chat.py         # Chat interactivo")
    print("   python interactive_chat.py --help  # Ver opciones de chat")

if __name__ == "__main__":
    main()