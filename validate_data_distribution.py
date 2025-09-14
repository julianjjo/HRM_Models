#!/usr/bin/env python3
"""
Validador de distribución de datos para HRM distributed training
Demuestra el PROBLEMA y la SOLUCIÓN
"""

def test_old_distribution():
    """Prueba la distribución INCORRECTA (como estaba antes)"""
    print("❌ DISTRIBUCIÓN INCORRECTA (método anterior):")
    print("=" * 50)
    
    total_samples = 100000
    world_size = 4
    
    print(f"Total samples solicitados por CADA rank: {total_samples}")
    print(f"World size: {world_size}")
    print()
    
    # Así estaba ANTES - cada rank carga los mismos datos y luego los divide
    for rank in range(world_size):
        print(f"Rank {rank}:")
        print(f"  1. Carga dataset completo: {total_samples} samples")
        
        # Luego DistributedTextDataset divide
        texts_per_rank = total_samples // world_size
        start_idx = rank * texts_per_rank
        end_idx = start_idx + texts_per_rank if rank < world_size - 1 else total_samples
        local_samples = end_idx - start_idx
        
        print(f"  2. DistributedTextDataset divide: samples {start_idx}-{end_idx} ({local_samples} samples)")
        print(f"  3. GPU {rank} procesa: {local_samples} samples únicos")
        print()
    
    total_redundant_loading = total_samples * world_size
    total_actual_processing = total_samples  # Solo se procesa una vez cada sample, pero se carga 4 veces
    
    print(f"📊 PROBLEMA:")
    print(f"  - Datos cargados total: {total_redundant_loading:,} samples (redundancia {world_size}x)")
    print(f"  - Datos procesados total: {total_actual_processing:,} samples")
    print(f"  - Eficiencia de carga: {(total_actual_processing/total_redundant_loading)*100:.1f}%")
    print(f"  - Desperdicio de ancho de banda: {((total_redundant_loading-total_actual_processing)/total_redundant_loading)*100:.1f}%")

def test_new_distribution():
    """Prueba la distribución CORRECTA (método corregido)"""
    print("\n✅ DISTRIBUCIÓN CORRECTA (método corregido):")
    print("=" * 50)
    
    total_samples = 100000
    world_size = 4
    
    print(f"Total samples en dataset: {total_samples}")
    print(f"World size: {world_size}")
    print()
    
    total_loaded = 0
    
    # Así está AHORA - cada rank carga solo su porción
    for rank in range(world_size):
        samples_per_rank = total_samples // world_size
        start_sample = rank * samples_per_rank
        end_sample = start_sample + samples_per_rank
        
        if rank == world_size - 1:
            end_sample = total_samples
        
        local_samples = end_sample - start_sample
        total_loaded += local_samples
        
        print(f"Rank {rank}:")
        print(f"  1. Carga SOLO su porción: samples {start_sample}-{end_sample}")
        print(f"  2. GPU {rank} procesa: {local_samples} samples ÚNICOS")
        print()
    
    print(f"📊 SOLUCIÓN:")
    print(f"  - Datos cargados total: {total_loaded:,} samples (sin redundancia)")
    print(f"  - Datos procesados total: {total_loaded:,} samples")  
    print(f"  - Eficiencia de carga: 100%")
    print(f"  - Desperdicio de ancho de banda: 0%")
    print(f"  - Aceleración de carga: {world_size}x")

def test_edge_cases():
    """Prueba casos edge como números no divisibles"""
    print("\n🧪 CASOS EDGE (números no divisibles):")
    print("=" * 50)
    
    test_cases = [
        (10000, 3),   # No divisible perfectamente
        (100001, 4),  # Número impar
        (1000, 7),    # Muchos ranks
    ]
    
    for total_samples, world_size in test_cases:
        print(f"\nCaso: {total_samples} samples, {world_size} GPUs")
        
        total_assigned = 0
        for rank in range(world_size):
            samples_per_rank = total_samples // world_size
            start_sample = rank * samples_per_rank
            end_sample = start_sample + samples_per_rank
            
            if rank == world_size - 1:
                end_sample = total_samples
            
            local_samples = end_sample - start_sample
            total_assigned += local_samples
            
            print(f"  Rank {rank}: {local_samples} samples (índices {start_sample}-{end_sample})")
        
        print(f"  ✅ Total asignado: {total_assigned} == {total_samples} (sin pérdida)")

def demo_actual_problem():
    """Demuestra el problema real con logs simulados"""
    print("\n🎯 DEMOSTRACIÓN DEL PROBLEMA REAL:")
    print("=" * 50)
    
    print("Logs simulados del método INCORRECTO:")
    print("📥 Rank 0: Descargando 100,000 samples de C4...")
    print("📥 Rank 1: Descargando 100,000 samples de C4...")  
    print("📥 Rank 2: Descargando 100,000 samples de C4...")
    print("📥 Rank 3: Descargando 100,000 samples de C4...")
    print("📚 Rank 0: 25,000 textos locales de 100,000 totales")
    print("📚 Rank 1: 25,000 textos locales de 100,000 totales") 
    print("📚 Rank 2: 25,000 textos locales de 100,000 totales")
    print("📚 Rank 3: 25,000 textos locales de 100,000 totales")
    print("⚠️  PROBLEMA: ¡Cada rank descargó los MISMOS 100K samples!")
    
    print("\nLogs simulados del método CORREGIDO:")
    print("📥 Rank 0: Descargando samples 0-25,000...")
    print("📥 Rank 1: Descargando samples 25,000-50,000...")
    print("📥 Rank 2: Descargando samples 50,000-75,000...")  
    print("📥 Rank 3: Descargando samples 75,000-100,000...")
    print("📚 Rank 0: 25,000 textos ÚNICOS")
    print("📚 Rank 1: 25,000 textos ÚNICOS")
    print("📚 Rank 2: 25,000 textos ÚNICOS") 
    print("📚 Rank 3: 25,000 textos ÚNICOS")
    print("✅ SOLUCIÓN: ¡Cada rank descarga datos diferentes!")

def main():
    print("🔍 ANÁLISIS DE DISTRIBUCIÓN DE DATOS HRM")
    print("🎯 Detectando y corrigiendo problema de redundancia")
    
    test_old_distribution()
    test_new_distribution()
    test_edge_cases()
    demo_actual_problem()
    
    print("\n" + "="*70)
    print("📝 RESUMEN:")
    print("❌ PROBLEMA: Todos los ranks cargan los mismos datos y luego los dividen")
    print("✅ SOLUCIÓN: Cada rank carga solo su porción única desde el inicio")
    print("🚀 BENEFICIOS:")
    print("   - 4x menos tráfico de red")
    print("   - 4x menos memoria temporal")  
    print("   - 4x más rápido iniciar entrenamiento")
    print("   - Escalabilidad real con más GPUs")

if __name__ == "__main__":
    main()