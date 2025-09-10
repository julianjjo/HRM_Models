#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alias para compatibilidad hacia atrás
Redirige a hf_tokenizer_wrapper_simple.py que es la versión limpia
"""

# Importar todo del wrapper simplificado
from hf_tokenizer_wrapper_simple import *

# Mensaje de aviso
import warnings
warnings.warn(
    "hf_tokenizer_wrapper.py es ahora un alias. "
    "Use hf_tokenizer_wrapper_simple.py directamente.",
    DeprecationWarning,
    stacklevel=2
)

print("📢 Usando wrapper HF simplificado (solo HuggingFace, sin espacios especiales)")