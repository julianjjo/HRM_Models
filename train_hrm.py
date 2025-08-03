# train_hrm_wandb.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from HRM import HRM
from tqdm import tqdm
import os
import wandb

# 1. Configuration
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
REASONING_STEPS = 8  # 'T' from the paper: internal steps for the L-module

## WandB Logging
wandb.init(
    project="hrm-training-demo",
    name=MODEL_NAME,
    config={
        "model_name": MODEL_NAME,
        "tokenizer": TOKENIZER_NAME,
        "hrm_dim": HRM_DIM,
        "depth_l": HRM_DEPTH_L,
        "depth_h": HRM_DEPTH_H,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "num_segments": NUM_SEGMENTS,
        "reasoning_steps": REASONING_STEPS,
    }
)

# 2. Setup Device and Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
VOCAB_SIZE = tokenizer.vocab_size

# 3. Load and Prepare Dataset
print("Loading and preparing TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split='train')

def tokenize_and_chunk(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // SEQ_LEN) * SEQ_LEN
    result = {
        k: [t[i: i + SEQ_LEN] for i in range(0, total_length, SEQ_LEN)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def flatten_dataset(examples):
    flattened = {key: [] for key in examples.keys()}
    num_sequences = len(examples['input_ids'])
    for i in range(num_sequences):
        for key in examples.keys():
            flattened[key].append(examples[key][i])
    return flattened

tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples['text']),
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=dataset.column_names
)

chunked_dataset = tokenized_dataset.map(
    tokenize_and_chunk,
    batched=True,
    num_proc=os.cpu_count(),
)

flattened_dataset = chunked_dataset.map(
    flatten_dataset,
    batched=True,
    num_proc=os.cpu_count(),
)

dataloader = DataLoader(flattened_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4. Initialize Model and Optimizer
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

def detach_hiddens(hiddens):
    if hiddens is None:
        return None
    elif isinstance(hiddens, dict):
        return {k: v.detach() if torch.is_tensor(v) else v for k, v in hiddens.items()}
    elif torch.is_tensor(hiddens):
        return hiddens.detach()
    else:
        return [detach_hiddens(h) for h in hiddens]

# 5. Training Loop
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    hrm.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in progress_bar:
        inputs = torch.stack(batch['input_ids'])[:, :-1].to(device)
        labels = torch.stack(batch['labels'])[:, 1:].to(device)

        hiddens = None
        segment_loss = 0

        for _ in range(NUM_SEGMENTS):
            optimizer.zero_grad()
            loss, new_hiddens, _ = hrm(inputs, hiddens=hiddens, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hrm.parameters(), 1.0)
            optimizer.step()
            hiddens = detach_hiddens(new_hiddens)
            segment_loss += loss.item()

        avg_segment_loss = segment_loss / NUM_SEGMENTS
        total_loss += avg_segment_loss
        progress_bar.set_postfix({"loss": f"{avg_segment_loss:.4f}"})
        
        wandb.log({"train_loss": avg_segment_loss})

    avg_epoch_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} finished. Average Batch Loss: {avg_epoch_loss:.4f}")

    wandb.log({"epoch_avg_loss": avg_epoch_loss, "epoch": epoch + 1})


# 6. Save the Model
print(f"Training complete. Saving model to {MODEL_NAME}.pth")
torch.save(hrm.state_dict(), f"{MODEL_NAME}.pth")

wandb.finish()
