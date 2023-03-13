import math

import torch
import torch.nn.functional as F
from diffusers import models
from diffusers.models import cross_attention

def sum_over_dim(activation, name):

    return activation.sum(dim=1)

def sum_over_head(activation, name):

    return activation.sum(dim=0)

def low_mem(activation, name):

    return activation.cpu().half()

def interpolate(activation, name):

    n_heads, n_patches, n_dim = activation.shape

    h = w = int(math.sqrt(n_patches))

    if 'attn2' in name:

        activation = F.interpolate(torch.transpose(activation, 1, 2).view(n_heads, n_dim, h, w), size=(64, 64), mode='bicubic')

    else:
    
        activation = F.interpolate(torch.transpose(activation, 1, 2).view(1, n_heads, n_dim, h, w), size=(4096, 64, 64), mode='trilinear')[0]

    return activation

class AttentionHookModule(torch.nn.Module):

    def forward(self, probs):

        return probs

class _CrossAttnProcessor(cross_attention.CrossAttnProcessor, torch.nn.Module):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.attnprobshook = AttentionHookModule()
        self.attnhshook = AttentionHookModule()
        
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attnprobshook(attention_probs)

        hidden_states = torch.bmm(attention_probs, value)

        self.attnhshook(hidden_states)
        
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class _CrossAttention(cross_attention.CrossAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
     
        self.attnscoreshook = AttentionHookModule()
    
    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        self.attnscoreshook(attention_scores)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

        return attention_probs

cross_attention.CrossAttnProcessor = _CrossAttnProcessor
models.attention.CrossAttention = _CrossAttention

def group_by_type(attentions, tlen=None):

    self_attention = {}
    cross_attention = {}
     
    for key, value in list(attentions.items()):

        if 'attn1' in key:
            self_attention[key] = value
        elif 'attn2' in key:
            if tlen is not None:
                breakpoint()
                cross_attention[key] = value[..., 1:tlen+1]
            else:
                cross_attention[key] = value

        del attentions[key]

    return self_attention, cross_attention


def stack_attentions(trace_steps):

    attentions = {}

    keys = list(trace_steps[0].keys())

    for key in keys:

        _attentions = []

        for trace_step in trace_steps:

            _attentions.append(trace_step[key].output)
            del trace_step[key]

        attentions[key] = torch.stack(_attentions)

    return attentions