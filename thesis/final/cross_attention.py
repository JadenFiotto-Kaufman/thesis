import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..experiments.attention.attention import AttentionHookModule, group_by_type, interpolate, low_mem, sum_over_dim, stack_attentions
from .. import util
from ..StableDiffuser import StableDiffuser

def edit_output(activation, name):

    activation = interpolate(activation, name)
    activation = low_mem(activation, name)

    return activation

def to_image(att, title, vmax, vmin):

    plt.figure(figsize=(5,5), dpi=200)
    plt.imshow(att, cmap='inferno', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout(pad=0)

    image = util.figure_to_image(plt.gcf())

    plt.close()

    return image


def main(prompt, outpath):

    os.makedirs(outpath, exist_ok=True)

    diffuser = StableDiffuser(scheduler='EA').to(torch.device('cuda:0')).half()

    layers = set([module_name for module_name, module in diffuser.named_modules() if 'attnprobshook' in module_name and 'attn2' in module_name])

    images, trace_steps = diffuser(prompt,
        generator=torch.manual_seed(50),
        n_steps=50, 
        trace_args={'layers' : layers, 'edit_output' : edit_output}
    )

    images[0][-1].save(os.path.join(outpath, 'image.png'))

    attentions = stack_attentions(trace_steps)

    self_attentions, cross_attentions = group_by_type(attentions)

    tokens = diffuser.text_tokenize([prompt])['input_ids'][0][1:]
    tokens = diffuser.text_detokenize(tokens)

    layers = cross_attentions.keys()
    cross_attentions = np.stack(list(cross_attentions.values()))

    attention_over_time = cross_attentions.mean(axis=0)

    attention_over_time = attention_over_time.mean(axis=1)

    vmin = attention_over_time[:,1:(len(tokens)+1)].min()

    vmax = attention_over_time[:,1:(len(tokens)+1)].max()

    aot_images = []

    for timestep in range(attention_over_time.shape[0]):

        token_images = []

        for token_idx in range(len(tokens)):

            token_images.append(to_image(attention_over_time[timestep, token_idx+1], tokens[token_idx], vmax, vmin))

        aot_images.append(util.image_grid([token_images]))

    util.to_gif(aot_images, os.path.join(outpath, 'aot.gif'))

    os.makedirs(outpath, exist_ok=True)

    for layer_idx, layer in enumerate(layers):

        attention_over_time = cross_attentions[layer_idx].mean(axis=1)

        vmin = attention_over_time[:,1:(len(tokens)+1)].min()

        vmax = attention_over_time[:,1:(len(tokens)+1)].max()

        aot_images = []

        for timestep in range(attention_over_time.shape[0]):

            token_images = []

            for token_idx in range(len(tokens)):

                token_images.append(to_image(attention_over_time[timestep, token_idx+1], tokens[token_idx], vmax, vmin))

            aot_images.append(util.image_grid([token_images]))

        util.to_gif(aot_images, os.path.join(outpath, f'{layer}.gif'))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('prompt')
    parser.add_argument('outpath')

    main(**vars(parser.parse_args()))