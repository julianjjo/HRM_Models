# 🤖 HRM-Text1 Chat Interactivo

Dos versiones optimizadas del chat interactivo para diferentes entornos.

## 📋 Archivos Disponibles

### 1. `interactive_chat.py` - Chat Completo
**Para:** Ejecutar desde línea de comandos o scripts Python

**Características:**
- ✅ Detección automática de modelos
- ✅ Generación streaming de tokens
- ✅ Sistema completo de comandos
- ✅ Gestión de historial
- ✅ Estadísticas de rendimiento
- ✅ Configuración avanzada

### 2. `jupyter_chat.py` - Chat Simplificado  
**Para:** Jupyter Notebooks, Google Colab, IPython

**Características:**
- ✅ Optimizado para notebooks
- ✅ Importación sin errores de `__file__`
- ✅ API simple y limpia
- ✅ Detección automática de entorno
- ✅ Funciones de conveniencia

## 🚀 Uso Rápido

### En Línea de Comandos:
```bash
# Chat completo con selección de modelo
python interactive_chat.py

# Usar modelo específico
python interactive_chat.py --dataset mixed

# Ver opciones
python interactive_chat.py --help
```

### En Jupyter/Colab:
```python
# Importar y usar
from jupyter_chat import quick_chat, demo

# Crear chat
chat = quick_chat()

# Chatear
chat.generate("Hola, ¿cómo estás?")

# Configurar parámetros
chat.config(temperature=0.8, max_new_tokens=200)

# Ver historial
chat.show_history()

# Demo automática
demo()
```

## ⚙️ Configuración de Parámetros

### Parámetros Disponibles:
- **`max_new_tokens`**: Longitud máxima (50-500)
- **`temperature`**: Creatividad (0.1-2.0)
- **`top_k`**: Filtrado top-k (1-200)
- **`top_p`**: Nucleus sampling (0.1-1.0)
- **`repetition_penalty`**: Anti-repetición (1.0-2.0)

### Ejemplos:
```python
# Más creativo
chat.config(temperature=1.2, top_p=0.95)

# Más conservador
chat.config(temperature=0.3, top_k=20)

# Respuestas más largas
chat.config(max_new_tokens=300)
```

## 🔧 Comandos del Chat Completo

Cuando uses `interactive_chat.py`:

- `/help` - Mostrar ayuda
- `/config` - Configurar parámetros
- `/stats` - Ver estadísticas
- `/history` - Ver historial
- `/save` - Guardar historial
- `/clear` - Limpiar historial
- `/exit` - Salir

## 🎯 Detección Automática de Modelos

Ambas versiones buscan modelos en:
- `/content/drive/MyDrive/HRM/` (Google Colab)
- `./` (Directorio actual)
- `~/HRM/` (Directorio home)
- Cualquier carpeta `hrm_text1_*_output-large`

## 📊 Datasets Soportados

- `c4` - Common Crawl multilingüe
- `mixed` - Combinación balanceada
- `slimpajama` - SlimPajama completo
- `slimpajama_es` - SlimPajama español
- `slimpajama_en` - SlimPajama inglés
- `high_quality` - Mezcla de alta calidad
- Y muchos más...

## 🛠️ Resolución de Problemas

### Error: "Modelo HRM no disponible"
```python
# Verificar que el archivo de entrenamiento esté presente
import os
print(os.path.exists('hrm_llm_training_c4_b.py'))

# Si no está, el chat usará modelos estándar de transformers
```

### Error: "No se encontraron modelos"
```bash
# Verificar directorio actual
ls -la hrm_text1_*

# O especificar ruta manualmente
python interactive_chat.py --model /ruta/a/tu/modelo
```

### En Google Colab:
```python
# Montar Drive primero
from google.colab import drive
drive.mount('/content/drive')

# Luego usar jupyter_chat
from jupyter_chat import quick_chat
chat = quick_chat()
```

## 💡 Tips de Uso

1. **Primera vez**: Usa `demo()` para probar
2. **Desarrollo**: Usa `jupyter_chat.py` para experimentos rápidos
3. **Producción**: Usa `interactive_chat.py` para funcionalidad completa
4. **Rendimiento**: Ajusta `max_new_tokens` según tus necesidades
5. **Creatividad**: Experimenta con `temperature` y `top_p`

## 🎮 Ejemplos de Uso

### Conversación Simple:
```python
chat = quick_chat()
chat.generate("Explica qué es la inteligencia artificial")
```

### Ajustar Creatividad:
```python
chat.config(temperature=1.5)  # Más creativo
chat.generate("Escribe un poema sobre robots")
```

### Análisis Técnico:
```python
chat.config(temperature=0.2)  # Más preciso
chat.generate("¿Cómo funciona un transformer en deep learning?")
```

¡Disfruta chateando con tu modelo HRM-Text1! 🚀