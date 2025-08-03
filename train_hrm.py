# train_hrm.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from HRM import HRM
from tqdm import tqdm
import os

# --- 1. Configuration ---
# Model & Tokenizer
MODEL_NAME = "hrm-tinystories"
TOKENIZER_NAME = "Qwen/Qwen3-32B"
HRM_DIM = 512
HRM_DEPTH_L = 2  # Depth of the Low-level (Specialist) module
HRM_DEPTH_H = 4  # Depth of the High-level (Manager) module

# Training
BATCH_SIZE = 8
SEQ_LEN = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1  # Set to 1 for a quick demo, increase for real training
NUM_SEGMENTS = 4  # Number of "deep supervision" steps per batch
REASONING_STEPS = 8 # 'T' from the paper: internal steps for the L-module

# --- 2. Setup Device and Tokenizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
VOCAB_SIZE = tokenizer.vocab_size

# --- 3. Load and Prepare Dataset ---
print("Loading and preparing TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split='train')

# Function to tokenize and chunk the data
def tokenize_and_chunk(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # We drop the small remainder, we could add padding if we wanted to
    total_length = (total_length // SEQ_LEN) * SEQ_LEN
    
    # Split by chunks of SEQ_LEN
    result = {
        k: [t[i : i + SEQ_LEN] for i in range(0, total_length, SEQ_LEN)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# Function to flatten the chunked dataset
def flatten_dataset(examples):
    """Convert list of sequences to individual examples"""
    flattened = {key: [] for key in examples.keys()}
    
    # Get the number of sequences (should be the same for all keys)
    num_sequences = len(examples['input_ids'])
    
    # Create individual examples for each sequence
    for i in range(num_sequences):
        for key in examples.keys():
            flattened[key].append(examples[key][i])
    
    return flattened

# Tokenize the entire dataset
tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples['text']),
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=dataset.column_names
)

# Chunk the tokenized dataset
chunked_dataset = tokenized_dataset.map(
    tokenize_and_chunk,
    batched=True,
    num_proc=os.cpu_count(),
)

# Flatten the chunked dataset so each sequence is a separate example
flattened_dataset = chunked_dataset.map(
    flatten_dataset,
    batched=True,
    num_proc=os.cpu_count(),
)

# Create a DataLoader
dataloader = DataLoader(flattened_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 4. Initialize Model and Optimizer ---
print("Initializing HRM model...")
hrm = HRM(
    networks=[
        dict(dim=HRM_DIM, depth=HRM_DEPTH_L),
        dict(dim=HRM_DIM, depth=HRM_DEPTH_H)
    ],
    num_tokens=VOCAB_SIZE,
    dim=HRM_DIM,
    reasoning_steps=REASONING_STEPS
).to(device)

optimizer = torch.optim.AdamW(hrm.parameters(), lr=LEARNING_RATE)

# Helper function to detach hidden states
def detach_hiddens(hiddens):
    """Detach hidden states (handles both dict and tensor formats)"""
    if hiddens is None:
        return None
    elif isinstance(hiddens, dict):
        return {k: v.detach() if torch.is_tensor(v) else v for k, v in hiddens.items()}
    elif torch.is_tensor(hiddens):
        return hiddens.detach()
    else:
        # Handle other iterable types like list or tuple
        return [detach_hiddens(h) for h in hiddens]

# --- 5. Training Loop ---
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    hrm.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in progress_bar:
        # Prepare inputs and labels for next-token prediction
        # Convert to tensor and move to device
        inputs = torch.stack(batch['input_ids'])[:, :-1].to(device)
        labels = torch.stack(batch['labels'])[:, 1:].to(device)

        # Initialize hidden state for this batch
        hiddens = None
        segment_loss = 0

        # The "deep supervision" loop
        for _ in range(NUM_SEGMENTS):
            optimizer.zero_grad()
            
            # Pass hidden state from previous segment
            loss, new_hiddens, _ = hrm(inputs, hiddens=hiddens, labels=labels)
            
            # Backpropagate loss for this segment
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hrm.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            
            # CRITICAL: Detach hidden state to prevent backprop across segments
            # This is the "one-step gradient approximation" that makes training efficient
            hiddens = detach_hiddens(new_hiddens)
            
            segment_loss += loss.item()

        avg_segment_loss = segment_loss / NUM_SEGMENTS
        total_loss += avg_segment_loss
        progress_bar.set_postfix({"loss": f"{avg_segment_loss:.4f}"})

    print(f"Epoch {epoch+1} finished. Average Batch Loss: {total_loss / len(dataloader):.4f}")

# --- 6. Save the Model ---
print(f"Training complete. Saving model to {MODEL_NAME}.pth")
torch.save(hrm.state_dict(), f"{MODEL_NAME}.pth")
