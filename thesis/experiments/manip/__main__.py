import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from functools import partial

from ...experiments.attention.attention import (AttentionHookModule,
                                                group_by_type, low_mem,
                                                stack_attentions)
from ...StableDiffuser import StableDiffuser

def edit_output(activations, name, trace, token_idx, removal_token_idx, scale):

    trace = trace[name].output

    activations[8:, token_idx] -= trace[8:, removal_token_idx] * scale

    return activations

def main(prompt, removal_prompt, outpath, scale, device, seed):

    diffuser = StableDiffuser(scheduler='LMS').to(torch.device(device)).half()

    layers = set([module_name for module_name, module in diffuser.named_modules() if (module_name.endswith('attnkeyhook') or module_name.endswith('attnvaluehook')) and 'attn2' in module_name])

    nsteps = 50

    os.makedirs(outpath, exist_ok=True)

    generator = torch.manual_seed(seed)

    images = diffuser(
            prompt,
            generator=generator,
            n_steps=nsteps, 
        )
    
    images[0][0].save(os.path.join(outpath, 'orig.png'))

    _, trace_steps = diffuser(
            removal_prompt,
            generator=generator,
            n_steps=1, 
            trace_args={'layers' : layers}
        )
    
    removal_trace = trace_steps[0]

    generator = torch.manual_seed(seed)

    tokens = diffuser.text_tokenize([prompt])['input_ids'][0][1:]
    tokens = diffuser.text_detokenize(tokens)

    removal_tokens = diffuser.text_tokenize([removal_prompt])['input_ids'][0][1:]
    removal_tokens = diffuser.text_detokenize(removal_tokens)

    eo = partial(edit_output, trace=removal_trace, token_idx = len(removal_tokens), removal_token_idx = len(removal_tokens), scale=scale)

    images, trace_steps = diffuser(
            prompt,
            generator=generator,
            n_steps=nsteps, 
            trace_args={'layers' : layers, 'edit_output' : eo}
        )
    
    images[0][0].save(os.path.join(outpath, 'edited.png'))
    

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('prompt')
    parser.add_argument('removal_prompt')
    parser.add_argument('outpath')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--scale', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)


    main(**vars(parser.parse_args()))

