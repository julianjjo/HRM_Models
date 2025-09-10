# Chat HRM para HuggingFace

Interfaz de chat simple para modelos HRM entrenados con HuggingFace.

## Características

✅ **Carga automática** - Carga modelos HRM entrenados desde directorios HF  
✅ **Tokenizador integrado** - Usa tokenizadores de HuggingFace directamente  
✅ **Generación avanzada** - Top-k, top-p, temperatura configurable  
✅ **Dos modos** - Prompt único o chat interactivo  
✅ **Multi-dispositivo** - CPU, CUDA, MPS automático  

## Uso Rápido

### Prompt único
```bash
python chat_hrm_hf.py --prompt "Hello, how are you?" --max_length 50
```

### Chat interactivo
```bash
python chat_hrm_hf.py
```

### Parámetros principales
```bash
python chat_hrm_hf.py \
  --model_path ./hrm-micro-10m-hf/final_model \
  --prompt "What is AI?" \
  --max_length 100 \
  --temperature 0.8 \
  --top_k 50 \
  --top_p 0.9 \
  --device auto
```

## Modelos soportados

El chat funciona con cualquier modelo HRM entrenado que tenga:
- `pytorch_model.bin` - Pesos del modelo
- `config.json` - Configuración del modelo  
- Tokenizador HF (vocab.json, tokenizer.json, etc.)

## Comandos del chat interactivo

- `/quit` o `/exit` - Salir del chat
- `/clear` - Limpiar historial de conversación
- Ctrl+C - Salir del chat

## Ejemplos

```bash
# Usar mejor modelo disponible
python chat_hrm_hf.py --model_path ./hrm-small-50m-hf/best_model

# Generación más creativa
python chat_hrm_hf.py --prompt "Write a story" --temperature 1.0 --top_p 0.8

# Generación más determinística  
python chat_hrm_hf.py --prompt "Explain Python" --temperature 0.5 --top_k 20

# Forzar uso de CPU
python chat_hrm_hf.py --device cpu
```

## Notas

- El modelo carga automáticamente el tokenizador desde la misma carpeta
- Se recomienda usar modelos con al menos 1000+ pasos de entrenamiento para mejores resultados
- La calidad de las respuestas depende del dataset de entrenamiento y número de épocas
- Para conversaciones largas, el contexto se trunca automáticamente al tamaño del bloque

## Estructura de archivos esperada

```
modelo_entrenado/
├── pytorch_model.bin     # Pesos del modelo (requerido)
├── config.json          # Configuración (opcional, usa defaults si no existe)
├── vocab.json           # Vocabulario del tokenizador
├── tokenizer.json       # Tokenizador rápido
├── merges.txt           # Merges de BPE
└── ...                  # Otros archivos de HF
```