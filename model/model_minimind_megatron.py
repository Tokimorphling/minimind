# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Megatron Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.model_minimind import MiniMindConfig, RMSNorm, precompute_freqs_cis, apply_rotary_pos_emb, repeat_kv

# Import Megatron-Core

from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region


# model_parallel_cuda_manual_seed(1234)

class MegatronAttention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        
        # Parallel setup
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.n_local_heads = self.num_attention_heads // world_size
        self.n_local_kv_heads = self.num_key_value_heads // world_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        # ColumnParallelLinear for QKV
        self.qkv_proj = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            (args.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            config=args
        )
        
        # RowParallelLinear for O
        self.o_proj = tensor_parallel.RowParallelLinear(
            args.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=nn.init.xavier_normal_,
            config=args,
            skip_bias_add=True
        )
        
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        
        # [bsz, seq_len, (H + 2*KVH) * D]
        qkv = self.qkv_proj(x)[0]
        
        # Split Q, K, V
        q_size = self.n_local_heads * self.head_dim
        kv_size = self.n_local_kv_heads * self.head_dim
        
        xq = qkv[:, :, :q_size]
        xk = qkv[:, :, q_size : q_size + kv_size]
        xv = qkv[:, :, q_size + kv_size :]
        
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
            if attention_mask is not None:
                extended_attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
                scores = scores + extended_attention_mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output)[0])
        return output, past_kv

class MegatronFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            
        # Gate and Up Proj (ColumnParallel)
        self.gate_up_proj = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            config=config,
        )
        
        # Down Proj (RowParallel)
        self.down_proj = tensor_parallel.RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=nn.init.xavier_normal_,
            config=config,
            skip_bias_add=True
        )
        self.dropout = nn.Dropout(config.dropout)
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate_up = self.gate_up_proj(x)[0]
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        x = self.act_fn(gate) * up
        return self.dropout(self.down_proj(x)[0])

class MiniMindMegatronBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = MegatronAttention(config)
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MegatronFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindMegatronModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # Vocab Parallel Embedding
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, init_method=nn.init.xavier_normal_, config=config
        )
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindMegatronBlock(l, config) for l in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings, 
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states, position_embeddings,
                past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents

class MiniMindMegatronForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindMegatronModel(self.config)
        # ColumnParallelLinear for Head
        self.lm_head = tensor_parallel.ColumnParallelLinear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            config=config
        )

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=False, **kwargs):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        logits_parallel, _ = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits_parallel[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Parallel Cross Entropy
            loss = tensor_parallel.vocab_parallel_cross_entropy(shift_logits, shift_labels)
            loss = loss.mean()

        # For inference, we might need to gather logits
        logits = None
        if not self.training:
            logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits_parallel)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )
