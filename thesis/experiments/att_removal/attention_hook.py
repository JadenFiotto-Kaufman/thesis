import math

import torch
import torch.nn.functional as F
from diffusers.models import attention

class AttentionHookModule(torch.nn.Module):

    def forward(self, probs):


        n_heads, n_patches, n_dim = probs.shape

        h = w = int(math.sqrt(n_patches))

        cross_attention = n_dim == 77

        return upsampled.half().cpu()

class _CrossAttention(attention.CrossAttention):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.attnhook = AttentionHookModule()
        
    def _attention(self, query, key, value, attention_mask=None):
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

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            if self.upcast_softmax:
                attention_scores = attention_scores.float()

            attention_probs = attention_scores.softmax(dim=-1)

            # cast back to the original dtype
            attention_probs = attention_probs.to(value.dtype)

            self.attnhook(attention_probs)

            # compute attention output
            hidden_states = torch.bmm(attention_probs, value)

            # reshape hidden_states
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

attention.CrossAttention = _CrossAttention

def group_by_type(attentions, t_len):

    self_attention = {}
    cross_attention = {}
     
    for key, value in list(attentions.items()):

        if 'attn1' in key:
            self_attention[key] = value
        elif 'attn2' in key:
            cross_attention[key] = value[:,:, 1:t_len+1]

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