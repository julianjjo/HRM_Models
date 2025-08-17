# -*- coding: utf-8 -*-
"""
HRM-Text1 Training Script - ADAPTADO PARA MÚLTIPLES DATASETS
VERSIÓN FINAL: Corregido el error de operación in-place, optimizado con streaming y generación.
### CHECKPOINT ###: Añadida lógica para guardar y reanudar el entrenamiento desde checkpoints.
### VERSIÓN AMPLIADA ###: Aumentado el tamaño del modelo para mayor capacidad.
### FIX DATALOADER ###: Solucionado el error de multiprocessing en DataLoader.
### MULTI-DATASET ###: Soporte para múltiples datasets y mezclas.

INSTRUCCIONES DE USO:

1. CONFIGURACIÓN DE DATASETS (línea ~245):
   Datasets individuales:
   - "c4": Common Crawl multilingüe
   - "openwebtext": Texto web en inglés
   - "pile": Dataset diverso de EleutherAI
   - "spanish": Texto en español (OSCAR)
   - "slimpajama": SlimPajama completo (627B tokens)
   - "slimpajama_es": SlimPajama filtrado en español
   - "slimpajama_en": SlimPajama filtrado en inglés
   
   Mezclas predefinidas:
   - "mixed": Combinación balanceada (incluye SlimPajama)
   - "mixed_es": Combinación enfocada en español
   
   Mezclas personalizadas (líneas 160-184):
   - "high_quality": Enfocada en calidad (SlimPajama + Pile + OpenWebText)
   - "multilingual_balanced": Multilingüe balanceado
   - "experimental_full": Experimental con todos los datasets

2. CONFIGURACIÓN DE PORCENTAJES:
   - DATASET_SUBSET_PERCENT (línea 156): Porcentaje del dataset total (1-100)
   - CUSTOM_MIX_RATIOS (líneas 160-184): Define tus propias mezclas
   
   Ejemplo de mezcla personalizada:
   CUSTOM_MIX_RATIOS = {
       "mi_mezcla": {
           "slimpajama_en": 0.6,  # 60%
           "spanish": 0.3,        # 30%
           "pile": 0.1           # 10%
       }
   }
   Luego usar: ACTIVE_DATASET = "mi_mezcla"

3. CONFIGURACIÓN DE RUTAS PERSONALIZADAS (líneas 390-430):
   
   Método 1 - Editar script directamente:
   CUSTOM_BASE_PATH = "/tu/ruta/personalizada"
   
   Método 2 - Variable de entorno:
   export HRM_OUTPUT_BASE="/tu/ruta/personalizada"
   python hrm_llm_training_c4_b.py
   
   Ejemplos de rutas:
   - Linux/Mac: "/home/usuario/modelos_hrm"
   - Windows: "D:/HRM_Models" 
   - Colab: "/content/drive/MyDrive/MisModelos"
   - Relativa: "./mis_modelos"

4. DETECCIÓN DE IDIOMA:
   - Requiere: pip install langdetect
   - Se aplica automáticamente a datasets con "language_filter"

5. MODIFICACIÓN DE LEARNING RATE AL CARGAR CHECKPOINT:
   - MODIFY_LR_ON_LOAD = True (línea ~507)
   - NEW_LEARNING_RATE = valor_deseado (línea ~508)
"""

import os, random, contextlib, multiprocessing as mp, atexit
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset
from transformers import T5Tokenizer, PreTrainedModel, PretrainedConfig, GenerationMixin
from tqdm.auto import tqdm

from huggingface_hub import HfFolder, HfApi

# Para detección de idioma
try:
    import langdetect
    LANGUAGE_DETECTION_AVAILABLE = True
    print("✅ Detección de idioma disponible con langdetect")
except ImportError:
    LANGUAGE_DETECTION_AVAILABLE = False
    print("⚠️  langdetect no disponible. Filtrado por idioma deshabilitado.")
    print("💡 Para habilitar automáticamente, ejecuta: pip install langdetect")
    
    # Intentar instalación automática si estamos en un entorno compatible
    try:
        import subprocess
        import sys
        response = input("¿Deseas instalar langdetect automáticamente? (y/n): ").strip().lower()
        if response in ['y', 'yes', 's', 'si']:
            print("🔄 Instalando langdetect...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "langdetect"])
            print("✅ langdetect instalado. Reiniciando detección...")
            try:
                import langdetect
                LANGUAGE_DETECTION_AVAILABLE = True
                print("✅ Detección de idioma ahora disponible")
            except ImportError:
                print("❌ Error al importar langdetect después de la instalación")
        else:
            print("⏩ Continuando sin detección de idioma")
    except Exception:
        pass  # Silenciar errores en entornos no interactivos

# Optimización específica para NVIDIA Ampere+
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    print("GPU NVIDIA compatible con TF32 detectada. Activando la precisión de matmul 'high'.")
    torch.set_float32_matmul_precision('high')

# ==============================================================================
# --- DEFINICIÓN DEL MODELO (Sin cambios en esta sección) ---
# ==============================================================================

class HRMText1Config(PretrainedConfig):
    model_type = "hrm_text1"
    def __init__(self, vocab_size=32128, block_size=512, n_embd=512, n_head=8, d_ff=2048, dropout=0.1, halt_max_steps=8, ponder_loss_weight=1e-2, halt_bias_init=-2.2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size, self.block_size, self.n_embd, self.n_head, self.d_ff, self.dropout, self.halt_max_steps, self.ponder_loss_weight, self.halt_bias_init = vocab_size, block_size, n_embd, n_head, d_ff, dropout, halt_max_steps, ponder_loss_weight, halt_bias_init

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-6):
        super().__init__(); self.eps, self.weight = eps, nn.Parameter(torch.ones(n_embd))
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))

class SwiGLUMuchPelu(nn.Module):
    def __init__(self, n_embd, d_ff, dropout=0.1):
        super().__init__(); self.w1, self.w2, self.w3, self.dropout = nn.Linear(n_embd, d_ff, bias=False), nn.Linear(n_embd, d_ff, bias=False), nn.Linear(d_ff, n_embd, bias=False), nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class HRMBlock(nn.Module):
    def __init__(self, n_embd, n_head, d_ff, dropout=0.1):
        super().__init__(); self.norm1, self.attn, self.norm2, self.mlp, self.dropout = RMSNorm(n_embd), nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True), RMSNorm(n_embd), SwiGLUMuchPelu(n_embd, d_ff, dropout), nn.Dropout(dropout)
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x_norm = self.norm1(x); attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False); x = x + self.dropout(attn_out); x = x + self.dropout(self.mlp(self.norm2(x))); return x

class HRMInner(nn.Module):
    def __init__(self, config):
        super().__init__(); self.H_module, self.L_module = HRMBlock(config.n_embd, config.n_head, config.d_ff, config.dropout), HRMBlock(config.n_embd, config.n_head, config.d_ff, config.dropout)
    def forward(self, z_H, z_L, attn_mask=None, key_padding_mask=None):
        z_L_input = z_L + z_H; z_L_new = self.L_module(z_L_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask); z_H_input = z_H + z_L_new; z_H_new = self.H_module(z_H_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask); return z_H_new, z_L_new

class HRMText1(PreTrainedModel, GenerationMixin):
    config_class = HRMText1Config
    main_input_name = "input_ids"

    def __init__(self, config: HRMText1Config):
        super().__init__(config)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.register_buffer("pos_ids", torch.arange(config.block_size).unsqueeze(0))
        self.inner_model = HRMInner(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.halt_head = nn.Sequential(nn.Linear(config.n_embd, 1), nn.Sigmoid())
        with torch.no_grad(): self.halt_head[0].bias.fill_(config.halt_bias_init)

    def forward(self, input_ids, labels=None, attention_mask=None, past_key_values=None, **kwargs):
        batch_size, seq_len = input_ids.shape; device = input_ids.device
        z_L = self.token_embeddings(input_ids) + self.pos_embeddings(self.pos_ids[:, :seq_len])
        z_H = torch.zeros_like(z_L)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        remainders, total_z_H, n_updates = torch.ones((batch_size, seq_len), device=device), torch.zeros_like(z_H), torch.zeros((batch_size, seq_len), device=device)
        eps = 1e-6

        for step in range(self.config.halt_max_steps):
            p_halt = self.halt_head(z_H).squeeze(-1).clamp(eps, 1 - eps)
            is_last_step = step == (self.config.halt_max_steps - 1)
            halt_now_prob = p_halt if not is_last_step else torch.ones_like(p_halt)

            contrib = remainders * halt_now_prob
            total_z_H = total_z_H + contrib.unsqueeze(-1) * z_H

            if is_last_step: break

            remainders = remainders * (1 - p_halt)
            n_updates = n_updates + 1

            if torch.all(remainders < eps): break

            z_H, z_L = self.inner_model(z_H, z_L, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        logits, loss = self.lm_head(total_z_H), None
        if labels is not None:
            shift_logits, shift_labels = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            ponder_loss = torch.mean(n_updates)
            loss = lm_loss + self.config.ponder_loss_weight * ponder_loss

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", torch.ones_like(input_ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# ==============================================================================
# --- CONFIGURACIÓN DEL SCRIPT ---
# ==============================================================================

# --- CONFIGURACIÓN DE PORCENTAJES DE DATASETS ---
# Porcentaje del dataset completo a usar (1-100)
DATASET_SUBSET_PERCENT = 5

# CONFIGURACIÓN PERSONALIZADA DE MEZCLAS
# Puedes crear tus propias combinaciones aquí o modificar las existentes
CUSTOM_MIX_RATIOS = {
    # Ejemplo de mezcla personalizada enfocada en calidad
    "high_quality": {
        "slimpajama_en": 0.5,  # 50% SlimPajama inglés (alta calidad)
        "pile": 0.3,           # 30% The Pile (diversidad)
        "openwebtext": 0.2     # 20% OpenWebText (web content)
    },
    
    # Ejemplo de mezcla para contenido multilingüe balanceado
    "multilingual_balanced": {
        "c4": 0.4,             # 40% C4 (multilingüe)
        "slimpajama_en": 0.3,  # 30% SlimPajama inglés
        "spanish": 0.2,        # 20% Español
        "slimpajama_es": 0.1   # 10% SlimPajama español
    },
    
    # Ejemplo de mezcla experimental con todos los datasets
    "experimental_full": {
        "slimpajama": 0.25,    # 25% SlimPajama completo
        "c4": 0.25,            # 25% C4 multilingüe
        "pile": 0.2,           # 20% The Pile
        "openwebtext": 0.15,   # 15% OpenWebText
        "spanish": 0.15        # 15% Español
    }
}

# --- CONFIGURACIÓN DE DATASETS MÚLTIPLES ---
# Selecciona el dataset a usar cambiando ACTIVE_DATASET
ACTIVE_DATASET = "c4"  # Opciones: "c4", "openwebtext", "pile", "spanish", "mixed"

DATASETS_CONFIG = {
    "c4": {
        "name": "allenai/c4",
        "config": "multilingual",
        "train_samples": 364_868_892,
        "val_samples": 364_608,
        "repo_suffix": "C4",
        "description": "Common Crawl multilingüe"
    },
    "openwebtext": {
        "name": "openwebtext",
        "config": None,
        "train_samples": 8_013_769,
        "val_samples": None,  # Se usará split automático
        "repo_suffix": "OpenWebText",
        "description": "Dataset de texto web en inglés"
    },
    "pile": {
        "name": "EleutherAI/pile",
        "config": None,
        "train_samples": 210_607_728,
        "val_samples": 214_670,
        "repo_suffix": "Pile",
        "description": "Dataset diverso de EleutherAI"
    },
    "spanish": {
        "name": "oscar",
        "config": "unshuffled_deduplicated_es",
        "train_samples": 58_395_538,
        "val_samples": None,  # Se usará split automático
        "repo_suffix": "Spanish",
        "description": "Texto en español de OSCAR"
    },
    "slimpajama": {
        "name": "cerebras/SlimPajama-627B",
        "config": None,
        "train_samples": 627_000_000_000,  # 627B tokens aproximadamente
        "val_samples": None,  # Se usará split automático
        "repo_suffix": "SlimPajama",
        "description": "Dataset SlimPajama de 627B tokens (multilingüe)",
        "language_filter": None  # Usar todo el dataset
    },
    "slimpajama_es": {
        "name": "cerebras/SlimPajama-627B",
        "config": None,
        "train_samples": 50_000_000_000,  # Estimación para contenido en español
        "val_samples": None,
        "repo_suffix": "SlimPajama-ES",
        "description": "SlimPajama filtrado para contenido en español",
        "language_filter": "es"  # Filtrar solo español
    },
    "slimpajama_en": {
        "name": "cerebras/SlimPajama-627B",
        "config": None,
        "train_samples": 400_000_000_000,  # Estimación para contenido en inglés
        "val_samples": None,
        "repo_suffix": "SlimPajama-EN",
        "description": "SlimPajama filtrado para contenido en inglés",
        "language_filter": "en"  # Filtrar solo inglés
    },
    "mixed": {
        "name": "mixed",  # Identificador especial
        "config": None,
        "train_samples": 400_000_000,  # Estimación combinada
        "val_samples": 200_000,
        "repo_suffix": "Mixed",
        "description": "Combinación de múltiples datasets",
        "mix_ratios": {  # Proporción de cada dataset en la mezcla
            "c4": 0.3,
            "slimpajama_en": 0.3,
            "openwebtext": 0.2,
            "pile": 0.1,
            "spanish": 0.1
        }
    },
    "mixed_es": {
        "name": "mixed",  # Identificador especial
        "config": None,
        "train_samples": 100_000_000,  # Estimación para español
        "val_samples": 50_000,
        "repo_suffix": "Mixed-ES",
        "description": "Combinación de datasets con contenido en español",
        "mix_ratios": {  # Proporción de cada dataset en la mezcla
            "slimpajama_es": 0.6,
            "spanish": 0.4
        }
    },
    
    # Mezclas personalizadas dinámicas
    "custom": {
        "name": "mixed",
        "config": None,
        "train_samples": 300_000_000,
        "val_samples": 150_000,
        "repo_suffix": "Custom",
        "description": "Mezcla personalizada (configurable)",
        "mix_ratios": {}  # Se llenará dinámicamente
    }
}

# Añadir las mezclas personalizadas a la configuración principal
for custom_name, mix_ratios in CUSTOM_MIX_RATIOS.items():
    DATASETS_CONFIG[custom_name] = {
        "name": "mixed",
        "config": None,
        "train_samples": 300_000_000,
        "val_samples": 150_000,
        "repo_suffix": f"Custom-{custom_name.replace('_', '-').title()}",
        "description": f"Mezcla personalizada: {custom_name.replace('_', ' ').title()}",
        "mix_ratios": mix_ratios
    }

# Mostrar datasets disponibles
print("=== DATASETS DISPONIBLES ===")
for key, config in DATASETS_CONFIG.items():
    marker = " ← SELECCIONADO" if key == ACTIVE_DATASET else ""
    print(f"• {key}: {config['description']}{marker}")
print("=" * 30)

# Configuración del dataset activo
DATASET_INFO = DATASETS_CONFIG[ACTIVE_DATASET]
DATASET_NAME = DATASET_INFO["name"]
DATASET_CONFIG = DATASET_INFO["config"]

HF_REPO_ID = f"dreamwar/HRM-Text1-{DATASET_INFO['repo_suffix']}-large"
SEED = 42
NUM_EPOCHS = 2
BLOCK_SIZE = 512

# --- CAMBIOS PARA EL MODELO GRANDE ---
# Se reduce el tamaño del lote y se aumenta la acumulación de gradiente para
# manejar el aumento de uso de memoria VRAM.
BATCH_SIZE = 40
GRAD_ACCUM_STEPS = 2

LEARNING_RATE_MAX, LEARNING_RATE_MIN, WEIGHT_DECAY = 3e-4, 1e-5, 0.05
MIXED_PRECISION, EARLY_STOPPING_PATIENCE = True, 2

# --- CAMBIOS PARA EL MODELO GRANDE ---
# Se duplican las dimensiones clave del modelo para aumentar su capacidad.
MODEL_PARAMS = {
    "n_embd": 1024,             # <-- CAMBIO: Aumentado de 512 a 1024
    "n_head": 16,               # <-- CAMBIO: Aumentado de 8 a 16
    "d_ff": 4096,               # <-- CAMBIO: Aumentado de 2048 a 4096 (4 * n_embd)
    "dropout": 0.1,
    "halt_max_steps": 8,
    "ponder_loss_weight": 1e-2,
    "halt_bias_init": -2.2
}

T5_TOKENIZER_REPO = "t5-small"

# ==============================================================================
# --- CONFIGURACIÓN DE RUTAS PERSONALIZADAS ---
# ==============================================================================

# CONFIGURACIÓN DE RUTA BASE (personalizable)
# Puedes cambiar esta ruta para usar tu directorio preferido
CUSTOM_BASE_PATH = None  # Dejar None para usar la ruta por defecto

# EJEMPLOS DE RUTAS PERSONALIZADAS:
# CUSTOM_BASE_PATH = "/tu/ruta/personalizada"
# CUSTOM_BASE_PATH = "/home/usuario/modelos_hrm"
# CUSTOM_BASE_PATH = "D:/HRM_Models"  # Windows
# CUSTOM_BASE_PATH = "/content/drive/MyDrive/MisModelos"  # Colab personalizado

# Variable de entorno para ruta base (sobrescribe CUSTOM_BASE_PATH)
# Usar: export HRM_OUTPUT_BASE="/tu/ruta" antes de ejecutar el script
HRM_OUTPUT_BASE_ENV = os.environ.get('HRM_OUTPUT_BASE')

# Determinar ruta base final
def determine_output_base():
    """Determina la ruta base según la configuración"""
    # Prioridad: Variable de entorno > Ruta personalizada > Ruta por defecto
    if HRM_OUTPUT_BASE_ENV:
        return HRM_OUTPUT_BASE_ENV
    elif CUSTOM_BASE_PATH:
        return CUSTOM_BASE_PATH
    else:
        # Rutas por defecto según el entorno
        if os.path.exists("/content/drive/MyDrive"):
            return "/content/drive/MyDrive/HRM"  # Google Colab
        elif os.path.exists(os.path.expanduser("~/Documents")):
            return os.path.expanduser("~/Documents/HRM")  # Sistemas Unix/Mac
        else:
            return "./HRM_Models"  # Directorio actual como fallback

# Configurar rutas finales
OUTPUT_BASE = determine_output_base()
OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"hrm_text1_{ACTIVE_DATASET}_output-large")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.bin")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")

print(f"📁 Ruta base configurada: {OUTPUT_BASE}")
print(f"📁 Directorio de salida: {OUTPUT_DIR}")

# ==============================================================================
# --- VALIDACIÓN Y CREACIÓN DE DIRECTORIOS ---
# ==============================================================================

def validate_and_create_output_dir(output_dir, force_create=True):
    """
    Valida y crea el directorio de salida con verificaciones de seguridad
    """
    try:
        # Verificar que el directorio padre sea accesible
        parent_dir = os.path.dirname(output_dir)
        
        if not os.path.exists(parent_dir):
            if force_create:
                print(f"🔨 Creando directorio padre: {parent_dir}")
                os.makedirs(parent_dir, exist_ok=True)
            else:
                raise FileNotFoundError(f"Directorio padre no existe: {parent_dir}")
        
        # Crear directorio de salida
        if not os.path.exists(output_dir):
            print(f"🔨 Creando directorio de salida: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        else:
            print(f"✅ Directorio de salida existe: {output_dir}")
        
        # Verificar permisos de escritura
        test_file = os.path.join(output_dir, ".write_test")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"✅ Permisos de escritura verificados")
        except PermissionError:
            raise PermissionError(f"Sin permisos de escritura en: {output_dir}")
        
        # Verificar espacio disponible (estimación básica)
        try:
            import shutil
            free_space = shutil.disk_usage(output_dir).free
            free_gb = free_space / (1024**3)
            print(f"💾 Espacio libre disponible: {free_gb:.1f} GB")
            
            if free_gb < 2:
                print(f"⚠️  ADVERTENCIA: Poco espacio libre ({free_gb:.1f} GB). Se recomiendan al menos 2 GB")
            elif free_gb < 10:
                print(f"💡 Espacio moderado ({free_gb:.1f} GB). Para entrenamientos largos se recomiendan al menos 10 GB")
        except:
            print("ℹ️  No se pudo verificar el espacio disponible")
        
        return True
        
    except Exception as e:
        print(f"❌ Error configurando directorio de salida: {e}")
        print(f"💡 Sugerencias:")
        print(f"   - Verificar permisos del directorio padre")
        print(f"   - Usar una ruta diferente con CUSTOM_BASE_PATH")
        print(f"   - Verificar que tengas suficiente espacio en disco")
        return False

# Validar y crear directorios
print("\n🔍 Validando configuración de directorios...")
if not validate_and_create_output_dir(OUTPUT_DIR):
    print("❌ No se pudo configurar el directorio de salida. Abortando.")
    exit(1)

print(f"✅ Configuración de directorios completada")
print(f"📋 Archivos que se guardarán:")
print(f"   🏆 Mejor modelo: {BEST_MODEL_PATH}")
print(f"   💾 Checkpoints: {CHECKPOINT_PATH}")
print(f"   📝 Modelo final: {OUTPUT_DIR}/")

# --- CONFIGURACIÓN PARA MODIFICACIÓN DE LEARNING RATE ---
# Flag para activar/desactivar la modificación del learning rate al cargar checkpoint
# USO: Cambiar MODIFY_LR_ON_LOAD a True y ajustar NEW_LEARNING_RATE según sea necesario
# Esto permite continuar el entrenamiento con un learning rate diferente sin perder el progreso
MODIFY_LR_ON_LOAD = False  # Cambiar a True para activar la modificación
NEW_LEARNING_RATE = 1e-4   # Nuevo valor del learning rate cuando MODIFY_LR_ON_LOAD es True


# ==============================================================================
# --- FUNCIONES AUXILIARES PARA VALIDACIÓN DE CONFIGURACIÓN ---
# ==============================================================================

def validate_mix_ratios(mix_ratios, dataset_name=""):
    """
    Valida que los ratios de mezcla sumen 1.0 y que todos los datasets existan
    """
    if not mix_ratios:
        return True, "No hay ratios de mezcla definidos"
    
    # Verificar que los datasets existen
    available_datasets = set(DATASETS_CONFIG.keys()) - {"mixed", "mixed_es", "custom"} - set(CUSTOM_MIX_RATIOS.keys())
    for dataset in mix_ratios.keys():
        if dataset not in available_datasets:
            return False, f"Dataset '{dataset}' no existe. Disponibles: {sorted(available_datasets)}"
    
    # Verificar que suman aproximadamente 1.0
    total = sum(mix_ratios.values())
    if abs(total - 1.0) > 0.01:  # Tolerancia de 1%
        return False, f"Los ratios deben sumar 1.0, actualmente suman {total:.3f}"
    
    # Verificar que todos los valores son positivos
    for dataset, ratio in mix_ratios.items():
        if ratio <= 0:
            return False, f"El ratio para '{dataset}' debe ser positivo, actual: {ratio}"
    
    return True, f"Configuración válida para {dataset_name}"

def normalize_mix_ratios(mix_ratios):
    """
    Normaliza los ratios para que sumen exactamente 1.0
    """
    total = sum(mix_ratios.values())
    if total == 0:
        return mix_ratios
    
    return {dataset: ratio / total for dataset, ratio in mix_ratios.items()}

def show_mix_summary(mix_ratios, dataset_name=""):
    """
    Muestra un resumen de la configuración de mezcla
    """
    print(f"\n=== CONFIGURACIÓN DE MEZCLA: {dataset_name.upper()} ===")
    for dataset, ratio in sorted(mix_ratios.items()):
        desc = DATASETS_CONFIG.get(dataset, {}).get("description", "Desconocido")
        print(f"• {dataset:15} {ratio:>6.1%} - {desc}")
    print("=" * 50)

# ==============================================================================
# --- FUNCIONES AUXILIARES PARA FILTRADO DE IDIOMA ---
# ==============================================================================

def detect_language(text, target_lang=None, confidence_threshold=0.7):
    """
    Detecta el idioma de un texto y retorna True si coincide con el target_lang
    """
    if not LANGUAGE_DETECTION_AVAILABLE or target_lang is None:
        return True
    
    try:
        # Usar solo una muestra del texto para eficiencia
        sample_text = text[:500] if len(text) > 500 else text
        
        if len(sample_text.strip()) < 50:  # Texto muy corto
            return True
        
        detected_lang = langdetect.detect(sample_text)
        
        # Para algunos idiomas comunes, usar códigos alternativos
        lang_mapping = {
            'es': ['es', 'ca'],  # Español incluye catalán
            'en': ['en'],
            'fr': ['fr'],
            'de': ['de'],
            'it': ['it'],
            'pt': ['pt']
        }
        
        target_langs = lang_mapping.get(target_lang, [target_lang])
        return detected_lang in target_langs
        
    except Exception:
        # En caso de error, incluir el texto
        return True

def create_language_filter_function(target_lang):
    """
    Crea una función de filtro para un idioma específico
    """
    def language_filter(examples):
        if not LANGUAGE_DETECTION_AVAILABLE or target_lang is None:
            return examples
        
        filtered_examples = {key: [] for key in examples.keys()}
        
        # Detectar campo de texto
        text_field = None
        for field in ['text', 'content', 'document']:
            if field in examples:
                text_field = field
                break
        
        if text_field is None:
            return examples
        
        # Filtrar por idioma
        for i, text in enumerate(examples[text_field]):
            if isinstance(text, str) and detect_language(text, target_lang):
                for key in examples.keys():
                    filtered_examples[key].append(examples[key][i])
        
        return filtered_examples
    
    return language_filter

# ==============================================================================
# --- FUNCIONES AUXILIARES PARA DATALOADER ---
# ==============================================================================

def get_dataloader_workers():
    """Determina el número seguro de workers para DataLoader"""
    try:
        # Detectar si estamos en Google Colab
        if 'google.colab' in str(get_ipython()):
            print("Detectado entorno Google Colab. Usando num_workers=0 para evitar problemas de multiprocessing.")
            return 0
    except:
        pass

    try:
        # Detectar si estamos en Jupyter/IPython
        get_ipython()
        print("Detectado entorno Jupyter/IPython. Usando num_workers=0 para mayor estabilidad.")
        return 0
    except:
        pass

    # Para sistemas normales, usar menos workers para evitar problemas
    workers = min(2, mp.cpu_count())
    print(f"Detectado sistema normal. Usando {workers} workers para DataLoader.")
    return workers

def cleanup_dataloaders():
    """Función para limpiar DataLoaders al salir"""
    global train_loader, val_loader
    try:
        if 'train_loader' in globals():
            del train_loader
        if 'val_loader' in globals():
            del val_loader
        torch.cuda.empty_cache()
        print("DataLoaders limpiados correctamente.")
    except:
        pass

# Registrar la función de limpieza
atexit.register(cleanup_dataloaders)


# ==============================================================================
# --- INICIO DEL SCRIPT ---
# ==============================================================================

def set_seed(seed: int):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(SEED)

### CHECKPOINT ###: Crear el directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo detectado: {device}")

try:
    HF_TOKEN = os.environ['HF_TOKEN']; HfFolder.save_token(HF_TOKEN); print("Hugging Face token loaded.")
except Exception:
    print("HF_TOKEN secret not found."); HF_TOKEN = None

print("Loading tokenizer (T5 slow)...")
tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_REPO, use_fast=False, legacy=False)
if tokenizer.pad_token is None: tokenizer.add_special_tokens({"pad_token": "<pad>"})
print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

# Usar las cifras específicas del dataset seleccionado y calcular muestras
TOTAL_TRAIN_SAMPLES = DATASET_INFO["train_samples"]
TOTAL_VAL_SAMPLES = DATASET_INFO["val_samples"]

num_train_samples = int(TOTAL_TRAIN_SAMPLES * (DATASET_SUBSET_PERCENT / 100.0))

# Manejar datasets que no tienen split de validación predefinido
if TOTAL_VAL_SAMPLES is None:
    # Para datasets sin validación, usar el 1% del entrenamiento como validación
    num_val_samples = max(1000, int(num_train_samples * 0.01))
    print(f"Dataset sin split de validación. Usando {num_val_samples:,} ejemplos como validación.")
else:
    num_val_samples = int(TOTAL_VAL_SAMPLES * (DATASET_SUBSET_PERCENT / 100.0))

print(f"Loading dataset '{DATASET_NAME}' ({DATASET_INFO['description']}) in streaming mode.")

if ACTIVE_DATASET == "mixed" or ACTIVE_DATASET in CUSTOM_MIX_RATIOS or "mix_ratios" in DATASET_INFO:
    # Cargar y mezclar múltiples datasets
    print("--- CARGANDO DATASETS PARA MEZCLA ---")
    mixed_datasets = {}
    mix_ratios = DATASET_INFO["mix_ratios"]
    
    # Validar configuración de mezcla
    is_valid, message = validate_mix_ratios(mix_ratios, ACTIVE_DATASET)
    if not is_valid:
        print(f"❌ ERROR EN CONFIGURACIÓN: {message}")
        print("Usa normalize_mix_ratios() para corregir automáticamente")
        exit(1)
    else:
        print(f"✅ {message}")
    
    # Normalizar ratios para asegurar que sumen 1.0
    mix_ratios = normalize_mix_ratios(mix_ratios)
    
    # Mostrar resumen de la mezcla
    show_mix_summary(mix_ratios, ACTIVE_DATASET)
    
    for dataset_key, ratio in mix_ratios.items():
        if ratio > 0:
            ds_config = DATASETS_CONFIG[dataset_key]
            print(f"Cargando {dataset_key} ({ratio*100:.1f}%): {ds_config['description']}")
            
            if ds_config["config"]:
                ds = load_dataset(ds_config["name"], ds_config["config"], streaming=True)
            else:
                ds = load_dataset(ds_config["name"], streaming=True)
            
            # Aplicar filtro de idioma específico del dataset si existe
            ds_lang_filter = ds_config.get("language_filter")
            if ds_lang_filter and LANGUAGE_DETECTION_AVAILABLE:
                print(f"  Aplicando filtro de idioma {ds_lang_filter} a {dataset_key}")
                lang_filter_func = create_language_filter_function(ds_lang_filter)
                ds["train"] = ds["train"].filter(lang_filter_func, batched=True, batch_size=100)
                if "validation" in ds:
                    ds["validation"] = ds["validation"].filter(lang_filter_func, batched=True, batch_size=100)
            
            # Calcular muestras según la proporción
            samples_for_this_ds = int(num_train_samples * ratio)
            # Usar hash absoluto para evitar seeds negativos
            dataset_seed = SEED + abs(hash(dataset_key)) % 1000000
            mixed_datasets[dataset_key] = {
                "train": ds["train"].take(samples_for_this_ds).shuffle(seed=dataset_seed, buffer_size=5_000),
                "validation": ds.get("validation", ds["train"]).take(int(num_val_samples * ratio)) if ds.get("validation") else None
            }
    
    # Combinar los datasets
    from datasets import interleave_datasets
    
    train_datasets = [mixed_datasets[key]["train"] for key in mix_ratios.keys() if mix_ratios[key] > 0]
    train_probs = [mix_ratios[key] for key in mix_ratios.keys() if mix_ratios[key] > 0]
    
    raw_datasets = {
        "train": interleave_datasets(train_datasets, probabilities=train_probs, seed=SEED, stopping_strategy="all_exhausted")
    }
    
    # Para validación, tomar una muestra pequeña de cada dataset
    val_datasets = [mixed_datasets[key]["validation"] for key in mix_ratios.keys() 
                   if mix_ratios[key] > 0 and mixed_datasets[key]["validation"] is not None]
    if val_datasets:
        raw_datasets["validation"] = interleave_datasets(val_datasets, probabilities=train_probs, seed=SEED, stopping_strategy="all_exhausted")
    else:
        # Si no hay validación, usar una muestra del entrenamiento
        raw_datasets["validation"] = raw_datasets["train"].take(num_val_samples)
    
    print(f"Dataset mezclado creado con {len(train_datasets)} fuentes")
    
else:
    # Cargar dataset único
    if DATASET_CONFIG:
        raw_datasets = load_dataset(DATASET_NAME, DATASET_CONFIG, streaming=True)
    else:
        raw_datasets = load_dataset(DATASET_NAME, streaming=True)
    
    # Aplicar filtro de idioma si está especificado
    language_filter = DATASET_INFO.get("language_filter")
    if language_filter and LANGUAGE_DETECTION_AVAILABLE:
        print(f"--- APLICANDO FILTRO DE IDIOMA: {language_filter.upper()} ---")
        print("NOTA: Esto puede reducir significativamente la velocidad de carga inicial")
        
        # Crear función de filtro
        lang_filter_func = create_language_filter_function(language_filter)
        
        # Aplicar filtro a los datasets
        raw_datasets["train"] = raw_datasets["train"].filter(lang_filter_func, batched=True, batch_size=100)
        if "validation" in raw_datasets:
            raw_datasets["validation"] = raw_datasets["validation"].filter(lang_filter_func, batched=True, batch_size=100)
    elif language_filter and not LANGUAGE_DETECTION_AVAILABLE:
        print(f"⚠️  ADVERTENCIA: Filtro de idioma '{language_filter}' solicitado pero langdetect no está disponible")
        print("💡 Puedes instalar langdetect con: pip install langdetect")
        print("🔄 Continuando sin filtro de idioma...")


language_filter_info = ""
if DATASET_INFO.get("language_filter"):
    language_filter_info = f" (FILTRADO: {DATASET_INFO['language_filter'].upper()})"

print(f"\n!!! USANDO DATASET: {ACTIVE_DATASET.upper()} - {DATASET_INFO['description']}{language_filter_info} !!!")
print(f"!!! USANDO UN SUBCONJUNTO DEL {DATASET_SUBSET_PERCENT}% DEL DATASET !!!")
print(f"Tomando aprox. {num_train_samples:,} ejemplos de entrenamiento.")
print(f"Tomando aprox. {num_val_samples:,} ejemplos de validación.\n")

# Configurar los splits según el dataset
if ACTIVE_DATASET != "mixed":
    # Para datasets únicos, aplicar la lógica original
    if "validation" in raw_datasets:
        raw_datasets["train"] = raw_datasets["train"].take(num_train_samples).shuffle(seed=SEED, buffer_size=10_000)
        raw_datasets["validation"] = raw_datasets["validation"].take(num_val_samples)
    else:
        # Para datasets sin split de validación, dividir el entrenamiento
        print("Dividiendo dataset de entrenamiento para crear validación...")
        total_for_split = num_train_samples + num_val_samples
        train_dataset = raw_datasets["train"].take(total_for_split).shuffle(seed=SEED, buffer_size=10_000)
        
        # Crear splits manualmente
        raw_datasets["train"] = train_dataset.skip(num_val_samples).take(num_train_samples)
        raw_datasets["validation"] = train_dataset.take(num_val_samples)
# Para dataset mezclado, los splits ya están configurados

def tokenize_function(examples):
    """Función de tokenización flexible que maneja diferentes formatos de dataset"""
    texts = []
    
    # Manejar diferentes campos de texto según el dataset
    if "text" in examples:
        # Formato estándar (C4, OpenWebText, Pile)
        text_field = examples["text"]
    elif "content" in examples:
        # Algunos datasets usan 'content'
        text_field = examples["content"]
    elif "document" in examples:
        # Algunos datasets usan 'document'
        text_field = examples["document"]
    else:
        # Intentar encontrar el primer campo que parezca texto
        for key in examples.keys():
            if isinstance(examples[key][0], str) and len(examples[key][0]) > 50:
                text_field = examples[key]
                print(f"Usando campo '{key}' como texto")
                break
        else:
            raise ValueError(f"No se encontró campo de texto válido en el dataset. Campos disponibles: {list(examples.keys())}")
    
    # Procesar textos
    for text in text_field:
        if isinstance(text, str) and len(text) > 50:
            texts.append(str(text) + tokenizer.eos_token)
    
    return tokenizer(texts, truncation=True, max_length=BLOCK_SIZE, padding="max_length", add_special_tokens=False)

print("Applying tokenization function (on-the-fly)...")
tokenized_splits = {}

# Detectar columnas a eliminar dinámicamente
sample = next(iter(raw_datasets["train"]))
columns_to_remove = [col for col in sample.keys() if col not in ["input_ids", "attention_mask"]]
print(f"Columnas detectadas en el dataset: {list(sample.keys())}")
print(f"Columnas a eliminar después de tokenización: {columns_to_remove}")

for split_name in ["train", "validation"]:
    tokenized_splits[split_name] = raw_datasets[split_name].map(
        tokenize_function, 
        batched=True, 
        remove_columns=columns_to_remove
    ).with_format("torch")

# ### FIX DATALOADER ###: Usar la función segura para determinar workers
safe_num_workers = get_dataloader_workers()

print(f"Creando DataLoaders con {safe_num_workers} workers...")
train_loader = DataLoader(
    tokenized_splits["train"],
    batch_size=BATCH_SIZE,
    num_workers=safe_num_workers,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=2 if safe_num_workers > 0 else None
)
val_loader = DataLoader(
    tokenized_splits["validation"],
    batch_size=BATCH_SIZE,
    num_workers=safe_num_workers,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=2 if safe_num_workers > 0 else None
)

config = HRMText1Config(vocab_size=len(tokenizer), block_size=BLOCK_SIZE, **MODEL_PARAMS)
model = HRMText1(config).to(device)
print(f"Número de parámetros del modelo: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE_MAX, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
num_training_steps = (num_train_samples // (BATCH_SIZE * GRAD_ACCUM_STEPS)) * NUM_EPOCHS
print(f"Total de pasos de entrenamiento calculados: {num_training_steps}")

scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=LEARNING_RATE_MIN)
scaler = torch.amp.GradScaler(enabled=(MIXED_PRECISION and device.type == 'cuda'))

### CHECKPOINT ###: Sección para cargar el checkpoint si existe
start_epoch = 0
start_step = 0
best_val_loss = float('inf')
patience_counter = 0
CHECKPOINT_STEPS = 1000  # Guardar checkpoint cada 100 pasos

if os.path.exists(CHECKPOINT_PATH):
    print(f"--- Reanudando entrenamiento desde el checkpoint: {CHECKPOINT_PATH} ---")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])

    # Modificar el learning rate si el flag está activado
    if MODIFY_LR_ON_LOAD:
        print(f"--- Modificando learning rate de {optimizer.param_groups[0]['lr']:.6f} a {NEW_LEARNING_RATE:.6f} ---")
        # Modificar el learning rate en el checkpoint del optimizer antes de cargarlo
        for param_group in checkpoint['optimizer_state_dict']['param_groups']:
            param_group['lr'] = NEW_LEARNING_RATE
        print(f"Learning rate modificado en el checkpoint del optimizer")
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Verificar que el learning rate se aplicó correctamente
    if MODIFY_LR_ON_LOAD:
        actual_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate después de cargar: {actual_lr:.6f}")
        if abs(actual_lr - NEW_LEARNING_RATE) > 1e-8:
            print(f"⚠️  Advertencia: El learning rate no se aplicó correctamente")
        else:
            print(f"✅ Learning rate modificado exitosamente a: {NEW_LEARNING_RATE:.6f}")

    start_epoch = checkpoint['epoch']
    start_step = checkpoint.get('step', 0)  # .get para compatibilidad con checkpoints antiguos
    best_val_loss = checkpoint['best_val_loss']
    patience_counter = checkpoint.get('patience_counter', 0) # .get para compatibilidad con checkpoints antiguos

    # VERIFICAR SI EL DATASET CAMBIÓ Y REAJUSTAR SCHEDULER
    checkpoint_training_steps = checkpoint.get('num_training_steps', 0)
    if checkpoint_training_steps != num_training_steps:
        print(f"Dataset cambió. Reajustando scheduler: {checkpoint_training_steps} -> {num_training_steps}")
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=LEARNING_RATE_MIN)
        # Ajustar el paso actual proporcionalmente
        current_progress = start_step / checkpoint_training_steps if checkpoint_training_steps > 0 else 0
        new_step = int(current_progress * num_training_steps)
        for _ in range(new_step):
            scheduler.step()
        print(f"Scheduler reajustado. Progreso: {current_progress:.2%}, nuevo paso: {new_step}")

    print(f"Checkpoint cargado. Reanudando desde la época {start_epoch + 1}, paso {start_step}.")
    print(f"Mejor pérdida de validación hasta ahora: {best_val_loss:.4f}")
else:
    print("--- No se encontró checkpoint. Empezando entrenamiento desde cero. ---")


if torch.__version__.startswith("2"):
    print("Compilando el modelo con torch.compile()...")
    model = torch.compile(model)

def main_training():
    """Función principal de entrenamiento con manejo de limpieza"""
    global train_loader, val_loader, global_step, best_val_loss, patience_counter

    try:
        ### CHECKPOINT ###: El bucle ahora empieza desde start_epoch
        global_step = start_step
        for epoch in range(start_epoch, NUM_EPOCHS):
            model.train()
            optimizer.zero_grad()
            progress = tqdm(train_loader, desc=f"Época {epoch+1}/{NUM_EPOCHS}")
            for i, batch in enumerate(progress):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = input_ids.clone()

                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=MIXED_PRECISION):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / GRAD_ACCUM_STEPS

                if loss is not None and torch.isfinite(loss):
                    scaler.scale(loss).backward()
                    if (i + 1) % GRAD_ACCUM_STEPS == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                        global_step += 1

                        # Guardar checkpoint cada CHECKPOINT_STEPS pasos
                        if global_step % CHECKPOINT_STEPS == 0:
                            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
                            print(f"\nGuardando checkpoint en paso {global_step}...")
                            torch.save({
                                'epoch': epoch,
                                'step': global_step,
                                'model_state_dict': model_to_save.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'scaler_state_dict': scaler.state_dict(),
                                'best_val_loss': best_val_loss,
                                'patience_counter': patience_counter,
                                'num_training_steps': num_training_steps,  # Guardar para verificar cambios
                            }, CHECKPOINT_PATH)
                            print(f"Checkpoint guardado en {CHECKPOINT_PATH}")

                    progress.set_postfix({"loss": f"{loss.item()*GRAD_ACCUM_STEPS:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}", "step": global_step})

            model.eval()
            total_val_loss, val_batches = 0.0, 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validando..."):
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = input_ids.clone()

                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=MIXED_PRECISION):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                    if outputs.loss is not None and torch.isfinite(outputs.loss):
                        total_val_loss += outputs.loss.item()
                        val_batches += 1

            if val_batches > 0:
                avg_val_loss = total_val_loss / val_batches
                print(f"Época {epoch+1}: Pérdida de Validación = {avg_val_loss:.4f}")

                model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"Nueva mejor pérdida de validación. Guardando modelo en {BEST_MODEL_PATH}")
                    torch.save(model_to_save.state_dict(), BEST_MODEL_PATH)
                    patience_counter = 0
                else:
                    patience_counter += 1

                ### CHECKPOINT ###: Guardar el estado del entrenamiento al final de cada época
                print(f"Guardando checkpoint al final de época {epoch+1} en {CHECKPOINT_PATH}...")
                torch.save({
                    'epoch': epoch + 1,  # La próxima época a ejecutar
                    'step': global_step,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'num_training_steps': num_training_steps,  # Guardar para verificar cambios
                }, CHECKPOINT_PATH)

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Detención temprana por falta de mejora en la validación.")
                break

        print("Entrenamiento finalizado.")

    finally:
        # Limpieza explícita al finalizar
        try:
            if 'train_loader' in globals():
                del train_loader
            if 'val_loader' in globals():
                del val_loader
            torch.cuda.empty_cache()
            print("Limpieza post-entrenamiento completada.")
        except:
            pass

# Ejecutar el entrenamiento
main_training()

model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

if os.path.exists(BEST_MODEL_PATH):
    print(f"Cargando el mejor modelo desde '{BEST_MODEL_PATH}' para el guardado final.")
    model_to_save.load_state_dict(torch.load(BEST_MODEL_PATH))

model_to_save.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Modelo y tokenizador guardados en '{OUTPUT_DIR}'")


def chat_with_model(prompt_text, model, tokenizer, max_new_tokens=60, temperature=0.7, top_k=50):
    model.eval()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=MIXED_PRECISION):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n--- Probando la Generación del Modelo Final ---")
try:
    inference_model = HRMText1.from_pretrained(OUTPUT_DIR).to(device)
    if torch.__version__.startswith("2"): inference_model = torch.compile(inference_model)

    prompts = ["The cat sat on the", "Artificial intelligence is a field that", "To be, or not to be, that is the question:"]
    for prompt in prompts:
        response = chat_with_model(prompt, inference_model, tokenizer)
        print(f"\nPrompt: {prompt}\nRespuesta: {response}")
except Exception as e:
    print(f"El test de generación falló: {e}")

print("\n--- Script completado exitosamente ---")

# ==============================================================================
# --- EJEMPLOS DE CONFIGURACIONES PERSONALIZADAS ---
# ==============================================================================

"""
EJEMPLOS DE USO AVANZADO:

1. CONFIGURACIÓN RÁPIDA PARA PRUEBAS:
   DATASET_SUBSET_PERCENT = 1  # Solo 1% del dataset
   ACTIVE_DATASET = "openwebtext"  # Dataset pequeño y rápido

2. CONFIGURACIÓN PARA ESPAÑOL OPTIMIZADA:
   ACTIVE_DATASET = "mixed_es"  # Mezcla enfocada en español
   DATASET_SUBSET_PERCENT = 10  # 10% del dataset

3. CONFIGURACIÓN PERSONALIZADA PARA INVESTIGACIÓN:
   CUSTOM_MIX_RATIOS = {
       "research_mix": {
           "slimpajama_en": 0.5,  # 50% datos de alta calidad
           "c4": 0.3,             # 30% diversidad multilingüe
           "pile": 0.2            # 20% contenido especializado
       }
   }
   ACTIVE_DATASET = "research_mix"

4. CONFIGURACIÓN PARA MODELO MULTILINGÜE:
   ACTIVE_DATASET = "multilingual_balanced"
   DATASET_SUBSET_PERCENT = 15

5. CONFIGURACIÓN MÁXIMA CALIDAD:
   ACTIVE_DATASET = "high_quality"
   DATASET_SUBSET_PERCENT = 20

NOTAS IMPORTANTES:
- Los porcentajes más altos requieren más tiempo y memoria
- Las mezclas personalizadas deben sumar 1.0 (100%)
- El script valida automáticamente las configuraciones
- Usa "slimpajama" completo solo si tienes suficiente almacenamiento
"""