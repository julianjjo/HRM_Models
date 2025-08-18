# 📁 Configuración de Rutas Personalizadas - HRM-Text1

Guía completa para configurar dónde se guardan los checkpoints y modelos entrenados.

## 🎯 Métodos de Configuración

### **Método 1: Editar Script Directamente** ⭐ RECOMENDADO
```python
# En hrm_training_small_100m.py, línea ~393
CUSTOM_BASE_PATH = "/tu/ruta/personalizada"
```

### **Método 2: Variable de Entorno** 🌍 FLEXIBLE
```bash
# En terminal (Linux/Mac)
export HRM_OUTPUT_BASE="/tu/ruta/personalizada"
python hrm_training_small_100m.py

# En Windows
set HRM_OUTPUT_BASE="D:\HRM_Models"
python hrm_training_small_100m.py
```

### **Método 3: Por Defecto Automático** 🤖 AUTOMÁTICO
El script detecta automáticamente la mejor ruta según tu sistema:
- Google Colab: `/content/drive/MyDrive/HRM`
- Unix/Mac: `~/Documents/HRM`
- Windows: `./HRM_Models`

## 📋 Ejemplos por Sistema

### **Google Colab:**
```python
# Guardar en Drive personalizado
CUSTOM_BASE_PATH = "/content/drive/MyDrive/MisModelos"

# Guardar en carpeta específica del proyecto
CUSTOM_BASE_PATH = "/content/drive/MyDrive/Investigacion/ModelosHRM"
```

### **Linux/Ubuntu:**
```python
# En directorio home
CUSTOM_BASE_PATH = "/home/usuario/modelos_hrm"

# En SSD externo
CUSTOM_BASE_PATH = "/mnt/ssd/hrm_models"

# En servidor compartido
CUSTOM_BASE_PATH = "/shared/models/hrm"
```

### **macOS:**
```python
# En Documents
CUSTOM_BASE_PATH = "/Users/usuario/Documents/HRM_Models"

# En Desktop
CUSTOM_BASE_PATH = "/Users/usuario/Desktop/Modelos"

# En disco externo
CUSTOM_BASE_PATH = "/Volumes/ExternalDrive/HRM"
```

### **Windows:**
```python
# En C:
CUSTOM_BASE_PATH = "C:/HRM_Models"

# En otro disco
CUSTOM_BASE_PATH = "D:/MachineLearning/HRM"

# En red
CUSTOM_BASE_PATH = "//servidor/shared/modelos"
```

## 📂 Estructura de Archivos Generada

Con cualquier ruta base configurada, se creará esta estructura:
```
[RUTA_BASE]/
├── hrm_text1_c4_output-large/          # Para dataset C4
│   ├── config.json                     # Configuración del modelo
│   ├── pytorch_model.bin               # Modelo entrenado
│   ├── tokenizer.json                  # Tokenizer
│   ├── best_model.bin                  # Mejor checkpoint
│   └── checkpoint.pth                  # Checkpoint actual
├── hrm_text1_mixed_output-large/       # Para dataset Mixed
└── hrm_text1_slimpajama_output-large/  # Para SlimPajama
```

## ⚙️ Configuración Avanzada

### **Diferentes Rutas por Dataset:**
```python
# Editar función determine_output_base() para lógica personalizada
def determine_output_base():
    if ACTIVE_DATASET == "slimpajama":
        return "/ssd/large_models"  # Datasets grandes en SSD
    elif ACTIVE_DATASET in ["c4", "mixed"]:
        return "/hdd/standard_models"  # Otros en HDD
    else:
        return "/home/user/experimental"  # Experimentales
```

### **Nombres Personalizados:**
```python
# Personalizar nombre del directorio final
OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"mi_modelo_{ACTIVE_DATASET}_{datetime.now().strftime('%Y%m%d')}")
```

## 🛡️ Validaciones Automáticas

El script automáticamente:

✅ **Verifica existencia** del directorio padre  
✅ **Crea directorios** si no existen  
✅ **Comprueba permisos** de escritura  
✅ **Estima espacio libre** disponible  
✅ **Muestra advertencias** si hay problemas  

### **Verificaciones de Espacio:**
- ⚠️ **< 2 GB**: Advertencia de poco espacio
- 💡 **2-10 GB**: Recomendación para entrenamientos largos
- ✅ **> 10 GB**: Espacio óptimo

## 🚨 Resolución de Problemas

### **Error: "Sin permisos de escritura"**
```bash
# Linux/Mac - Cambiar permisos
chmod 755 /tu/ruta/base
sudo chown $USER:$USER /tu/ruta/base

# Windows - Ejecutar como administrador
```

### **Error: "Directorio no existe"**
```python
# El script creará automáticamente, pero si falla:
import os
os.makedirs("/tu/ruta/base", exist_ok=True)
```

### **Error: "Poco espacio disponible"**
```bash
# Verificar espacio
df -h /tu/ruta/base          # Linux/Mac
dir /tu/ruta/base            # Windows

# Usar ruta con más espacio
CUSTOM_BASE_PATH = "/otra/ruta/con/espacio"
```

## 💡 Mejores Prácticas

### **📊 Para Datasets Grandes (SlimPajama):**
- Usar SSD para mejor rendimiento
- Asegurar al menos 50 GB libres
- Evitar rutas de red para checkpoint frecuentes

### **🔄 Para Experimentos:**
- Usar rutas con fecha/hora
- Crear subdirectorios por experimento
- Mantener logs junto con checkpoints

### **☁️ Para Google Colab:**
- Siempre usar Google Drive
- Crear carpetas organizadas por proyecto
- Respaldar modelos importantes

### **🖥️ Para Servidores:**
- Usar rutas absolutas
- Verificar permisos de grupo
- Considerar cuotas de disco

## 🔧 Scripts Útiles

### **Verificar Configuración Actual:**
```python
# Añadir al final del script
print(f"📋 CONFIGURACIÓN ACTUAL:")
print(f"   Ruta base: {OUTPUT_BASE}")
print(f"   Dataset: {ACTIVE_DATASET}")
print(f"   Directorio final: {OUTPUT_DIR}")
print(f"   Espacio libre: {shutil.disk_usage(OUTPUT_BASE).free / (1024**3):.1f} GB")
```

### **Limpiar Modelos Antiguos:**
```bash
# Encontrar modelos por fecha
find /tu/ruta/base -name "hrm_text1_*" -type d -mtime +30

# Limpiar checkpoints temporales
find /tu/ruta/base -name "checkpoint.pth" -size +1G
```

¡Con esta configuración tendrás control total sobre dónde se guardan tus modelos! 🎯