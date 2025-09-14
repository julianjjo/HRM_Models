#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script para validar configuración de entrenamiento distribuido HRM
Verifica GPUs, dependencias, y funcionalidad básica de distributed training
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import subprocess
import json

def test_cuda_availability():
    """Test CUDA and GPU availability"""
    print("🔍 Verificando CUDA y GPUs...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA no está disponible")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA disponible con {gpu_count} GPUs")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB, SM {props.major}.{props.minor})")
        
        # Test basic GPU operations
        try:
            device = torch.device(f'cuda:{i}')
            x = torch.randn(100, 100, device=device)
            y = torch.mm(x, x)
            print(f"      ✅ GPU {i} operacional")
        except Exception as e:
            print(f"      ❌ GPU {i} error: {e}")
            return False
    
    return True

def test_dependencies():
    """Test required dependencies"""
    print("\n🔍 Verificando dependencias...")
    
    required_packages = {
        'transformers': 'HuggingFace Transformers',
        'datasets': 'HuggingFace Datasets', 
        'tokenizers': 'HuggingFace Tokenizers',
        'tqdm': 'Progress bars'
    }
    
    missing = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {description}")
        except ImportError:
            print(f"   ❌ {description} (falta {package})")
            missing.append(package)
    
    if missing:
        print(f"\n💡 Instalar dependencias faltantes:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def test_tokenizer():
    """Test HF tokenizer functionality"""
    print("\n🔍 Verificando tokenizador...")
    
    try:
        from hf_tokenizer_wrapper_simple import create_tokenizer
        print("   ✅ Wrapper de tokenizador disponible")
        
        tokenizer = create_tokenizer("openai-community/gpt2")
        print(f"   ✅ Tokenizador GPT-2 cargado (vocab_size: {len(tokenizer)})")
        
        # Test tokenization
        test_text = "Hello, this is a test for the tokenizer."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"   ✅ Tokenización funcional ({len(tokens)} tokens)")
        
        return True
    except Exception as e:
        print(f"   ❌ Error con tokenizador: {e}")
        return False

def test_distributed_init():
    """Test distributed training initialization"""
    print("\n🔍 Verificando inicialización distribuida...")
    
    # Check if we're in a distributed environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print("   ℹ️ Detectado entorno distribuido")
        try:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            print(f"   📊 Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
            
            # Initialize process group
            dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
            
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                device = torch.device(f'cuda:{local_rank}')
            else:
                device = torch.device('cpu')
            
            print(f"   ✅ Proceso distribuido inicializado en {device}")
            
            # Test basic distributed operations
            tensor = torch.ones(2, 2, device=device) * rank
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = sum(range(world_size)) * torch.ones(2, 2, device=device)
            
            if torch.allclose(tensor, expected):
                print(f"   ✅ All-reduce funcional (suma = {tensor[0,0].item()})")
            else:
                print(f"   ❌ All-reduce falló (esperado {expected[0,0].item()}, got {tensor[0,0].item()})")
                return False
            
            dist.destroy_process_group()
            return True
            
        except Exception as e:
            print(f"   ❌ Error inicialización distribuida: {e}")
            return False
    else:
        print("   ℹ️ No hay entorno distribuido, test de single GPU...")
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            test_tensor = torch.randn(100, 100, device=device)
            result = torch.mm(test_tensor, test_tensor)
            print("   ✅ Operaciones GPU básicas funcionan")
            return True
        else:
            print("   ⚠️ No hay GPUs para test")
            return True

def test_model_creation():
    """Test basic HRM model creation"""
    print("\n🔍 Verificando creación de modelo HRM...")
    
    try:
        # Test imports from training script
        sys.path.append('.')
        
        # Try to import from distributed training script
        try:
            from hrm_training_small_50m_distributed import HRMText1Config, HRMText1, setup_distributed
            print("   ✅ Importación 50M exitosa")
            model_type = "50m"
        except ImportError:
            print("   ⚠️ No se pudo importar script 50M")
            try:
                from hrm_training_medium_100m_distributed import HRMText1Config, HRMText1, setup_distributed
                print("   ✅ Importación 100M exitosa")
                model_type = "100m"
            except ImportError:
                print("   ❌ No se pudieron importar scripts de entrenamiento")
                return False
        
        # Create a small test config
        if model_type == "50m":
            config = HRMText1Config(
                vocab_size=1000,  # Small vocab for test
                block_size=64,    # Small context for test
                n_embd=128,       # Small embedding for test
                n_head=4,         # Few heads for test
                n_layers=2,       # Few layers for test
                d_ff=256,         # Small FFN for test
            )
        else:  # 100m
            config = HRMText1Config(
                vocab_size=1000,  # Small vocab for test
                block_size=128,   # Small context for test
                n_embd=256,       # Small embedding for test
                n_head=8,         # Few heads for test
                n_layers=4,       # Few layers for test
                d_ff=512,         # Small FFN for test
            )
        
        # Create model
        model = HRMText1(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Modelo HRM test creado ({total_params:,} parámetros)")
        
        # Test forward pass
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs.logits
            
            print(f"   ✅ Forward pass exitoso (output shape: {logits.shape})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error creación modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_scripts():
    """Test that training scripts exist and are valid"""
    print("\n🔍 Verificando scripts de entrenamiento...")
    
    scripts = [
        'hrm_training_small_50m_distributed.py',
        'hrm_training_medium_100m_distributed.py',
        'launch_distributed_training.py'
    ]
    
    all_exist = True
    for script in scripts:
        if os.path.exists(script):
            print(f"   ✅ {script}")
            
            # Try to parse as Python to check syntax
            try:
                with open(script, 'r') as f:
                    compile(f.read(), script, 'exec')
                print(f"      ✅ Sintaxis válida")
            except SyntaxError as e:
                print(f"      ❌ Error de sintaxis: {e}")
                all_exist = False
        else:
            print(f"   ❌ {script} no encontrado")
            all_exist = False
    
    return all_exist

def run_distributed_test():
    """Run a small distributed test using torchrun"""
    print("\n🔍 Ejecutando test distribuido básico...")
    
    if torch.cuda.device_count() < 2:
        print("   ⚠️ Se necesitan al menos 2 GPUs para test distribuido")
        return True
    
    # Create a simple distributed test script
    test_script_content = '''
import torch
import torch.distributed as dist
import os

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    # Test tensor
    tensor = torch.ones(2, device=f"cuda:{local_rank}") * rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    expected_sum = sum(range(world_size))
    if abs(tensor[0].item() - expected_sum) < 1e-6:
        print(f"Rank {rank}: ✅ All-reduce successful ({tensor[0].item()})")
    else:
        print(f"Rank {rank}: ❌ All-reduce failed")
        exit(1)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
'''
    
    # Write temporary test script
    test_script_path = 'temp_distributed_test.py'
    with open(test_script_path, 'w') as f:
        f.write(test_script_content)
    
    try:
        # Run with 2 GPUs
        cmd = [
            'torchrun',
            '--nproc_per_node=2',
            '--master_port=29502',
            test_script_path
        ]
        
        print(f"   Ejecutando: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("   ✅ Test distribuido exitoso")
            return True
        else:
            print(f"   ❌ Test distribuido falló:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ Test distribuido timeout")
        return False
    except Exception as e:
        print(f"   ❌ Error ejecutando test: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(test_script_path):
            os.remove(test_script_path)

def generate_system_report():
    """Generate system configuration report"""
    print("\n📋 Generando reporte del sistema...")
    
    report = {
        "system": {
            "platform": sys.platform,
            "python_version": sys.version,
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "gpus": []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            report["gpus"].append({
                "id": i,
                "name": props.name,
                "memory_gb": props.total_memory / 1024**3,
                "compute_capability": f"{props.major}.{props.minor}",
                "multiprocessor_count": props.multi_processor_count,
            })
    
    # Try to get package versions
    packages = ['transformers', 'datasets', 'tokenizers', 'tqdm']
    report["packages"] = {}
    
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            report["packages"][package] = version
        except ImportError:
            report["packages"][package] = "not_installed"
    
    # Save report
    with open('system_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("   💾 Reporte guardado en: system_report.json")
    
    # Print summary
    print(f"   🐍 Python: {sys.version.split()[0]}")
    print(f"   🔥 PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   🚀 CUDA: {torch.version.cuda}")
        print(f"   📱 GPUs: {torch.cuda.device_count()}")
    else:
        print("   💻 CUDA: No disponible")

def main():
    """Main test function"""
    print("🧪 HRM Distributed Training Setup Test")
    print("=" * 50)
    
    tests = [
        ("CUDA y GPUs", test_cuda_availability),
        ("Dependencias", test_dependencies),
        ("Tokenizador", test_tokenizer),
        ("Scripts de entrenamiento", test_training_scripts),
        ("Creación de modelo", test_model_creation),
        ("Inicialización distribuida", test_distributed_init),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Error inesperado en {test_name}: {e}")
            results.append((test_name, False))
    
    # Optional distributed test if not in distributed environment
    if 'RANK' not in os.environ:
        try:
            result = run_distributed_test()
            results.append(("Test distribuido", result))
        except Exception as e:
            print(f"   ⚠️ Test distribuido saltado: {e}")
    
    # Generate system report
    generate_system_report()
    
    # Summary
    print(f"\n📊 Resumen de Tests:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\n🎯 Resultado: {passed}/{total} tests pasaron")
    
    if passed == total:
        print("\n🎉 ¡Sistema listo para entrenamiento distribuido!")
        print("\nPróximos pasos:")
        print("1. python launch_distributed_training.py --model 50m --dry_run")
        print("2. python launch_distributed_training.py --model 50m --gpus 2")
    else:
        print("\n⚠️ Algunos tests fallaron. Revisar errores arriba.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)