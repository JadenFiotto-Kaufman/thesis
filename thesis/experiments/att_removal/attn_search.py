import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..attention.attention import AttentionHookModule, group_by_type, interpolate, low_mem, sum_over_dim, stack_attentions
import math
from ... import util
from ...StableDiffuser import StableDiffuser, default_parser

def edit_output(activation, name):

    activation = interpolate(activation, name).sum(dim=1)
    activation = low_mem(activation, name)

    return activation

def to_image(att):

    plt.figure(figsize=(5,5), dpi=200)
    plt.imshow(att, cmap='inferno', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout(pad=0)

    image = util.figure_to_image(plt.gcf())

    plt.close()

    return image


def main(prompt1, prompt2, outpath, seed):

    os.makedirs(outpath, exist_ok=True)

    diffuser = StableDiffuser(seed=seed).to(torch.device('cuda'))

    layers = set([module_name for module_name, module in diffuser.named_modules() if 'attnhshook' in module_name])

    images, trace_steps = diffuser(prompt1,
        n_steps=25, 
        reseed=True,
        trace_args={'layers' : layers, 'edit_output' : edit_output}
    )

    img1 = images[0][0]

    img1.save(os.path.join(outpath, 'p1.png'))

    attentions = stack_attentions(trace_steps)

    self_attentions1, cross_attentions1 = group_by_type(attentions)

    del trace_steps

    images, trace_steps = diffuser(prompt2,
        n_steps=25, 
        reseed=True,
        trace_args={'layers' : layers, 'edit_output' : edit_output}
    )

    img2 = images[0][0]

    img2.save(os.path.join(outpath, 'p2.png'))

    attentions = stack_attentions(trace_steps)

    self_attentions2, cross_attentions2 = group_by_type(attentions)

    del trace_steps



    for key in cross_attentions1:

        if 'unet.up_blocks.3.attentions.0.transformer_blocks.0' not in key:

            continue

        p1_ca = cross_attentions1[key].float().sum(dim=1)
        p2_ca = cross_attentions2[key].float().sum(dim=1)

        sa_key = key.replace('attn2', 'attn1')

        p1_sa = self_attentions1[sa_key].float().sum(dim=1)
        p2_sa = self_attentions2[sa_key].float().sum(dim=1)

        for ts in range(p1_ca.shape[0]):

            images = [
                [img1, to_image(p1_ca[ts]), to_image(p1_sa[ts])],
                [img2, to_image(p2_ca[ts]), to_image(p2_sa[ts])]
            ]

            util.image_grid(images, os.path.join(outpath, f"{key}.{ts}.png"), row_titles=[prompt1, prompt2], column_titles=['image', 'cross attn', 'self attn'])

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('prompt1')
    parser.add_argument('prompt2')
    parser.add_argument('outpath')
    parser.add_argument('--seed', type=int, default=42)

    main(**vars(parser.parse_args()))