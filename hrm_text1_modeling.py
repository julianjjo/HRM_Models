import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# HRM Architecture (w/ Positional Embeddings for CausalLM)
# I shall refer to it as HRM-Text1
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))

class SwiGLUMuchPelu(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class HRMBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUMuchPelu(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class HRMInner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.H_module = HRMBlock(config["d_model"], config["n_heads"], config["d_ff"], config["dropout"])
        self.L_module = HRMBlock(config["d_model"], config["n_heads"], config["d_ff"], config["dropout"])
    def forward(self, z_H, z_L, attn_mask=None, key_padding_mask=None):
        z_L_input = z_L + z_H
        z_L_new = self.L_module(z_L_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        z_H_input = z_H + z_L_new
        z_H_new = self.H_module(z_H_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return z_H_new, z_L_new

class HRMText1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config["vocab_size"], config["d_model"])
        self.pos_embeddings = nn.Embedding(config["block_size"], config["d_model"])  # Positional embeddings
        self.register_buffer("pos_ids", torch.arange(config["block_size"]).unsqueeze(0))
        self.inner_model = HRMInner(config)
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        self.halt_head = nn.Sequential(nn.Linear(config["d_model"], 1), nn.Sigmoid())
        self.max_steps = config["halt_max_steps"]
        self.ponder_loss_weight = config["ponder_loss_weight"]

        with torch.no_grad():
            self.halt_head[0].bias.fill_(config.get("halt_bias_init", -2.0))

    def forward(self, input_ids, labels=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        z_L = self.token_embeddings(input_ids) + self.pos_embeddings(self.pos_ids[:, :seq_len])
        z_H = torch.zeros_like(z_L)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

        halting_probs = torch.zeros((batch_size, seq_len, self.max_steps), device=device)
        remainders = torch.ones((batch_size, seq_len), device=device)
        total_z_H = torch.zeros_like(z_H)
        n_updates = torch.zeros((batch_size, seq_len), device=device)

        eps = 1e-6
        for step in range(self.max_steps):
            p_halt = self.halt_head(z_H).squeeze(-1)
            p_halt = p_halt.clamp(eps, 1 - eps)
            is_last_step = (step == self.max_steps - 1)

            halt_now_prob = torch.ones_like(p_halt) if is_last_step else p_halt
            contrib = remainders * halt_now_prob

            halting_probs[:, :, step] = contrib
            total_z_H += contrib.unsqueeze(-1) * z_H

            # Update remainders; on last step, force to zero since we halt unconditionally
            remainders = remainders * (1 - p_halt) if not is_last_step else torch.zeros_like(remainders)

            # Accumulate the probability of executing the next inner_model call
            # (We don't add on the last step, as there is no call after it)
            if not is_last_step:
                n_updates += remainders

            if torch.all(remainders < eps):
                break

            z_H, z_L = self.inner_model(z_H, z_L, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        logits = self.lm_head(total_z_H)
        loss, ponder_loss, lm_loss = None, None, None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config["vocab_size"]), shift_labels.view(-1))
            ponder_loss = torch.mean(n_updates)
            loss = lm_loss + self.ponder_loss_weight * ponder_loss

        return {"loss": loss, "logits": logits, "ponder_loss": ponder_loss, "lm_loss": lm_loss}
