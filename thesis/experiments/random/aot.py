import numpy as np
import torch
from matplotlib import pyplot as plt

from ... import util
from ...StableDiffuser import StableDiffuser, default_parser
from ..attention.attention import AttentionHookModule, group_by_type, stack_attentions

if __name__ == '__main__':

    prompt = "An indian woman wearing a red tophat standing on a boulder in a jungle"


    diffuser = StableDiffuser(seed=42).to(torch.device('cuda:1'))

    layers = set([module_name for module_name, module in diffuser.named_modules() if 'attnprobshook' in module_name and 'attn2' in module_name])

    images, trace_steps = diffuser(prompt,
        n_steps=50, 
        n_imgs=1,
        trace_args={'layers' : layers}
    )

    tokens = diffuser.text_tokenize([prompt])['input_ids'][0][1:]
    tokens = diffuser.text_detokenize(tokens)

    attentions = stack_attentions(trace_steps)

    self_attentions, cross_attentions = group_by_type(attentions)
    
    cross_attention_ts = []

    for key in cross_attentions.keys():

        cross_attentions[key] = cross_attentions[key].mean(dim=(1,2))

        cross_attention_ts.append(cross_attentions[key])

    cross_attention_ts = torch.stack(cross_attention_ts).mean(dim=0)

    tokens = ['<SOS>'] + tokens

    for i in range(len(tokens)):

        if i == 0:
            continue

        plt.plot(range(50), cross_attention_ts[:, i].cpu().numpy(), label=tokens[i])



    plt.legend(loc="upper right")
    plt.savefig('ayy.png')



