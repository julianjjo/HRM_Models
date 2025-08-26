#!/usr/bin/env python3
"""
Quick fix for SimpleIterableDataset subscript error in all standalone models
"""

import os
import re

# Files to fix
standalone_files = [
    "hrm_training_micro_10m_standalone.py",
    "hrm_training_nano_25m_standalone.py", 
    "hrm_training_medium_100m_standalone.py",
    "hrm_training_medium_350m_standalone.py",
    "hrm_training_large_1b_standalone.py"
]

# Patterns to fix
fixes = [
    # Fix 1: Add isinstance check before accessing dict
    {
        'old_pattern': r'if has_validation:\s+raw_datasets\["train"\]',
        'new_code': '''if has_validation:
            # Verificar si raw_datasets es un dict o SimpleIterableDataset
            if isinstance(raw_datasets, dict):
                raw_datasets["train"]'''
    },
    # Fix 2: Handle SimpleIterableDataset case
    {
        'old_pattern': r'raw_datasets\["validation"\] = raw_datasets\["validation"\]\.take\(num_val_samples\)',
        'new_code': '''raw_datasets["validation"] = raw_datasets["validation"].take(num_val_samples)
            else:
                # raw_datasets es SimpleIterableDataset, crear splits manualmente
                print("📊 Detectado SimpleIterableDataset, creando splits...")
                train_dataset = raw_datasets.shuffle(seed=SEED, buffer_size=10_000)
                raw_datasets = {
                    "train": train_dataset.take(num_train_samples),
                    "validation": train_dataset.skip(num_train_samples).take(num_val_samples)
                }'''
    }
]

def fix_file(filepath):
    print(f"🔧 Fixing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply basic fix - add isinstance check for all raw_datasets["train"] accesses
    # This is a simpler approach than the complex pattern matching above
    
    # Find the dataset handling section and apply targeted fixes
    patterns_to_fix = [
        (r'train_dataset = raw_datasets\["train"\]\.take\(total_for_split\)\.shuffle',
         r'train_dataset = (raw_datasets["train"] if isinstance(raw_datasets, dict) else raw_datasets).take(total_for_split).shuffle'),
        
        (r'train_dataset = raw_datasets\["train"\]\.shuffle',
         r'train_dataset = (raw_datasets["train"] if isinstance(raw_datasets, dict) else raw_datasets).shuffle'),
        
        (r'raw_datasets\["train"\] = raw_datasets\["train"\]\.take\(num_train_samples\)\.shuffle',
         r'''if isinstance(raw_datasets, dict):
                raw_datasets["train"] = raw_datasets["train"].take(num_train_samples).shuffle(seed=SEED, buffer_size=10_000)
                raw_datasets["validation"] = raw_datasets["validation"].take(num_val_samples)
            else:
                print("📊 Detectado SimpleIterableDataset, creando splits...")
                train_dataset = raw_datasets.shuffle(seed=SEED, buffer_size=10_000)
                raw_datasets = {
                    "train": train_dataset.take(num_train_samples),
                    "validation": train_dataset.skip(num_train_samples).take(num_val_samples)
                }''')
    ]
    
    modified = False
    for old_pattern, new_code in patterns_to_fix:
        if re.search(old_pattern, content):
            content = re.sub(old_pattern, new_code, content)
            modified = True
            print(f"   ✅ Applied fix for: {old_pattern[:50]}...")
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✅ {filepath} updated")
    else:
        print(f"   ⚠️ No patterns found to fix in {filepath}")

def main():
    print("🔧 Fixing SimpleIterableDataset errors in standalone models...")
    
    for filename in standalone_files:
        if os.path.exists(filename):
            fix_file(filename)
        else:
            print(f"❌ File not found: {filename}")
    
    print("✅ All fixes applied!")

if __name__ == "__main__":
    main()