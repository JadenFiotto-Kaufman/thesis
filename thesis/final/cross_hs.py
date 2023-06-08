import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..experiments.attention.attention import AttentionHookModule, group_by_type, interpolate, low_mem, sum_over_dim, stack_attentions
from .. import util
from ..StableDiffuser import StableDiffuser

def edit_output(activation, name):

    activation = interpolate(activation, name)[1]
    activation = low_mem(activation, name)

    return activation

def to_image(att, vmax, vmin):

    plt.figure(figsize=(5,5), dpi=200)
    plt.imshow(att, cmap='inferno', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout(pad=0)

    image = util.figure_to_image(plt.gcf())

    plt.close()

    return image


def main(prompt, outpath):

    os.makedirs(outpath, exist_ok=True)

    diffuser = StableDiffuser(scheduler='EA').to(torch.device('cuda:0')).half()

    layers = set([module_name for module_name, module in diffuser.named_modules() if 'attnhshook' in module_name and 'attn2' in module_name])

    images, trace_steps = diffuser(prompt,
        generator=torch.manual_seed(50),
        n_steps=100, 
        return_steps=True,
        pred_x0=True,
        trace_args={'layers' : layers, 'edit_output' : edit_output}
    )

    images[0][-1].save(os.path.join(outpath, 'image.png'))

    attentions = stack_attentions(trace_steps)

    attentions = {key:np.absolute(value).mean(axis=1) for key, value in attentions.items()}

    layers = attentions.keys()
    attentions = np.stack(list(attentions.values()))

    scaled_path = os.path.join(outpath, 'scaled')
    unscaled_path = os.path.join(outpath, 'unscaled')
    mot_path = os.path.join(outpath, 'mot')
    total_path = os.path.join(outpath, 'total')

    os.makedirs(scaled_path, exist_ok=True)
    os.makedirs(unscaled_path, exist_ok=True)
    os.makedirs(mot_path, exist_ok=True)
    os.makedirs(total_path, exist_ok=True)

    for layer_idx, layer in enumerate(layers):

        attention_over_time = attentions[layer_idx]

        vmin = attention_over_time.min()

        vmax = attention_over_time.max()

        scaled_images = []
        unscaled_images = []

        for timestep in range(attention_over_time.shape[0]):

            scaled_images.append(util.image_grid([[to_image(attention_over_time[timestep], vmax, vmin), images[0][timestep]]]))
            unscaled_images.append(util.image_grid([[to_image(attention_over_time[timestep], None, None), images[0][timestep]]]))

        to_image(attention_over_time.mean(axis=0), None, None).save(os.path.join(total_path, f"{layer}.png"))
        to_image(attention_over_time.mean(axis=0), attentions.max(), attentions.min()).save(os.path.join(total_path, f"scaled_{layer}.png"))

        util.to_gif(scaled_images, os.path.join(scaled_path, f'{layer}.gif'))
        util.to_gif(unscaled_images, os.path.join(unscaled_path, f'{layer}.gif'))

        plt.plot(list(range(100)), attention_over_time.sum(axis=(1,2)))
        plt.savefig(os.path.join(mot_path, f"{layer}.png"))

        plt.clf()
        plt.close()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('prompt')
    parser.add_argument('outpath')

    main(**vars(parser.parse_args()))