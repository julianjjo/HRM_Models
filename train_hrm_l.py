import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from HRM import HRM
from tqdm import tqdm
import os
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

# 1. Configuration
# Model & Tokenizer
MODEL_NAME = "hrm-tinystories-expanded"
TOKENIZER_NAME = "Qwen/Qwen3-32B"

# Network parameters (for both low/high-level modules)
HRM_DIM = 1024
HRM_DEPTH_L = 6  # Specialist module depth
HRM_DEPTH_H = 8  # Manager module depth
ATTENTION_DIM_HEAD_L = 64  # Low-level attention head dim
HEADS_L = 8  # Low-level attention heads
ATTENTION_DIM_HEAD_H = 128  # High-level attention head dim
HEADS_H = 16  # High-level attention heads
REASONING_STEPS = 10

# Normalization and positional embedding settings
USE_RMSNORM_L = True
USE_RMSNORM_H = True
ROTARY_POS_EMB_L = True
ROTARY_POS_EMB_H = True
PRE_NORM_L = False
PRE_NORM_H = False

# Training
BATCH_SIZE = 8
SEQ_LEN = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
NUM_SEGMENTS = 4
CHECKPOINT_INTERVAL = 1000  # Configurable interval for checkpointing (in steps)

# WandB Logging
wandb.init(
    project="hrm-training-demo",
    name=MODEL_NAME,
    config={
        "model_name": MODEL_NAME,
        "tokenizer": TOKENIZER_NAME,
        "hrm_dim": HRM_DIM,
        "depth_l": HRM_DEPTH_L,
        "depth_h": HRM_DEPTH_H,
        "attn_dim_head_l": ATTENTION_DIM_HEAD_L,
        "heads_l": HEADS_L,
        "attn_dim_head_h": ATTENTION_DIM_HEAD_H,
        "heads_h": HEADS_H,
        "use_rmsnorm_l": USE_RMSNORM_L,
        "use_rmsnorm_h": USE_RMSNORM_H,
        "rotary_pos_emb_l": ROTARY_POS_EMB_L,
        "rotary_pos_emb_h": ROTARY_POS_EMB_H,
        "pre_norm_l": PRE_NORM_L,
        "pre_norm_h": PRE_NORM_H,
        "reasoning_steps": REASONING_STEPS,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "num_segments": NUM_SEGMENTS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
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
print("Initializing expanded HRM model...")
hrm = HRM(
    networks=[
        dict(
            dim=HRM_DIM,
            depth=HRM_DEPTH_L,
            attn_dim_head=ATTENTION_DIM_HEAD_L,
            heads=HEADS_L,
            use_rmsnorm=USE_RMSNORM_L,
            rotary_pos_emb=ROTARY_POS_EMB_L,
            pre_norm=PRE_NORM_L
        ),
        dict(
            dim=HRM_DIM,
            depth=HRM_DEPTH_H,
            attn_dim_head=ATTENTION_DIM_HEAD_H,
            heads=HEADS_H,
            use_rmsnorm=USE_RMSNORM_H,
            rotary_pos_emb=ROTARY_POS_EMB_H,
            pre_norm=PRE_NORM_H
        )
    ],
    num_tokens=VOCAB_SIZE,
    dim=HRM_DIM,
    reasoning_steps=REASONING_STEPS
).to(device)

optimizer = torch.optim.AdamW(hrm.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(dataloader))

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
total_steps = 0  # Counter for total training steps
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
            total_steps += 1  # Increment step counter after each gradient update
            if total_steps % CHECKPOINT_INTERVAL == 0:
                checkpoint_path = f"{MODEL_NAME}_checkpoint_{total_steps}.pth"
                torch.save(hrm.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at step {total_steps} to {checkpoint_path}")
            scheduler.step()
            hiddens = detach_hiddens(new_hiddens)
            segment_loss += loss.item()

        avg_segment_loss = segment_loss / NUM_SEGMENTS
        total_loss += avg_segment_loss
        progress_bar.set_postfix({"loss": f"{avg_segment_loss:.4f}"})
        
        wandb.log({"train_loss": avg_segment_loss})
        wandb.log({"learning_rate": scheduler.get_last_lr()[0]})

    avg_epoch_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} finished. Average Batch Loss: {avg_epoch_loss:.4f}")
    wandb.log({"epoch_avg_loss": avg_epoch_loss, "epoch": epoch + 1})

# 6. Save the Model
print(f"Training complete. Saving model to {MODEL_NAME}.pth")
torch.save(hrm.state_dict(), f"{MODEL_NAME}.pth")

wandb.finish()