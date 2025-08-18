import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
from configuration_hrm_text1 import HRMText1Config

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
    
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))

class SwiGLUMuchPelu(nn.Module):
    def __init__(self, n_embd, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(n_embd, d_ff, bias=False)
        self.w2 = nn.Linear(n_embd, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class HRMBlock(nn.Module):
    def __init__(self, n_embd, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = SwiGLUMuchPelu(n_embd, d_ff, dropout)
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
        self.H_module = HRMBlock(config.n_embd, config.n_head, config.d_ff, config.dropout)
        self.L_module = HRMBlock(config.n_embd, config.n_head, config.d_ff, config.dropout)
    
    def forward(self, z_H, z_L, attn_mask=None, key_padding_mask=None):
        z_L_input = z_L + z_H
        z_L_new = self.L_module(z_L_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        z_H_input = z_H + z_L_new
        z_H_new = self.H_module(z_H_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return z_H_new, z_L_new

class HRMText1Model(PreTrainedModel):
    config_class = HRMText1Config
    main_input_name = "input_ids"
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.register_buffer("pos_ids", torch.arange(config.block_size).unsqueeze(0))
        self.inner_model = HRMInner(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.halt_head = nn.Sequential(nn.Linear(config.n_embd, 1), nn.Sigmoid())
        self.max_steps = config.halt_max_steps
        self.ponder_loss_weight = config.ponder_loss_weight

        with torch.no_grad():
            halt_bias_value = getattr(config, "halt_bias_init", -2.0)
            self.halt_head[0].bias.fill_(halt_bias_value)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
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

            remainders = remainders * (1 - p_halt) if not is_last_step else torch.zeros_like(remainders)

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
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            ponder_loss = torch.mean(n_updates)
            loss = lm_loss + self.ponder_loss_weight * ponder_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def _prepare_causal_attention_mask(self, attention_mask, input_shape, dtype):
        bsz, seq_len = input_shape
        causal_mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=attention_mask.device)
        mask_cond = torch.arange(causal_mask.size(-1), device=attention_mask.device)
        causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0)
        causal_mask = causal_mask.to(dtype)
        return causal_mask.unsqueeze(0).expand(bsz, 1, seq_len, seq_len)

class HRMText1ForCausalLM(HRMText1Model, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """Prepara inputs para la generación"""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def get_output_embeddings(self):
        """Retorna la capa de embeddings de salida"""
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """Establece la capa de embeddings de salida"""
        self.lm_head = new_embeddings
    
    def resize_token_embeddings(self, new_num_tokens=None):
        """Redimensiona los embeddings de tokens"""
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        
        # También redimensionar lm_head
        old_lm_head = self.get_output_embeddings()
        new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
        self.set_output_embeddings(new_lm_head)
        
        return self.get_input_embeddings()
    
    def get_input_embeddings(self):
        """Retorna la capa de embeddings de entrada"""
        return self.token_embeddings
    
    def set_input_embeddings(self, new_embeddings):
        """Establece la capa de embeddings de entrada"""
        self.token_embeddings = new_embeddings