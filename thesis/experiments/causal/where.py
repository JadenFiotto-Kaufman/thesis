import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from baukit import TraceDict
from scipy.special import softmax
from scipy.stats import entropy

from ...StableDiffuser import StableDiffuser
from ...util import figure_to_image, to_gif
from ..attention.attention import *

cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])


def update_key(key):

    key = key.replace('attentions', '')
    key = key.replace('unet.', '')
    key = key.replace('transformer_blocks', '')
    key = key.replace('processor', '')
    key = key.replace('attn2', '')
    key = key.replace('...', '.')
    key = key.replace('..', '.')

    return key

def compare(artist_td, blank_td, artist_tokens_idxs):

    comparison = {}

    for key in artist_td:

        if 'hshook' not in key:
            continue

        artist_act = artist_td[key].output
        blank_act = blank_td[key].output

        attention_scores = artist_td[key.replace('hshook', 'attnhook')]

        for head in range(artist_act.size(0)):

            if head <= 7:

                continue

            _key = key +  f'.{head}'

            cosine_distance = (1 - torch.nn.functional.cosine_similarity(artist_act[head], blank_act[head], dim=-1)).mean()
            euc_distance = torch.linalg.norm(artist_act[head] - blank_act[head])

            _key = update_key(_key)

            comparison[_key] = {
                'cos' : cosine_distance,
                'euc' : euc_distance ,
                'att' : attention_scores.output[head, :, artist_tokens_idxs].sum(dim=-1).mean()
            }

    return comparison

def plot_gif(comparisons, outpath):

    euc_path =  os.path.join(outpath, 'euc_overtime.gif')
    cos_path =  os.path.join(outpath, 'cos_overtime.gif')

    euc_image, cos_image, sort_idx = vis_comparison(comparisons[0])

    images = [euc_image]

    for comparison in  comparisons[1:]:

        euc_image, cos_image, _ = vis_comparison(comparison, sort_idx=sort_idx)

        images.append(euc_image)

    to_gif(images, euc_path)

    euc_image, cos_image, sort_idx = vis_comparison(comparisons[0], sort_by='cos')

    images = [cos_image]

    for comparison in  comparisons[1:]:

        euc_image, cos_image, _ = vis_comparison(comparison, sort_idx=sort_idx)

        images.append(cos_image)

    to_gif(images, cos_path)

def average_comparisons(comparisons):

    output = {}

    for comparison in comparisons:

        for key in comparison:

            if key not in output:

                output[key] = {
                    'euc' : comparison[key]['euc'] / len(comparisons), 
                    'cos' : comparison[key]['cos'] / len(comparisons),
                    'att' : comparison[key]['att'] / len(comparisons)
                    }

            else:

                output[key]['euc'] += comparison[key]['euc'] / len(comparisons)
                output[key]['cos'] += comparison[key]['cos'] / len(comparisons)
                output[key]['att'] += comparison[key]['att'] / len(comparisons)

    return output       

def vis_comparison(comparison, sort_idx = None, sort_by='euc'):

    euclidean_values = []
    cosine_values = []
    attn_values = []
    keys = list(comparison.keys())

    for key in comparison:

        euclidean_values.append(comparison[key]['euc'].cpu().numpy())
        cosine_values.append(comparison[key]['cos'].cpu().numpy())
        attn_values.append(comparison[key]['att'].cpu().numpy())

    euclidean_values = np.array(euclidean_values)
    cosine_values = np.array(cosine_values)
    attn_values = np.array(attn_values)

    attn_values = attn_values / attn_values.max()

    if sort_idx is None:
        if sort_by == 'euc':
            sort_idx = euclidean_values.argsort()
        elif sort_by == 'cos':
            sort_idx = cosine_values.argsort()

    keys = [keys[idx] for idx in sort_idx]

    colors = cmap(attn_values)

    plt.figure(figsize=(20,20))
    plt.barh(keys, euclidean_values[sort_idx], color=colors)
    plt.title(f"Entropy: {entropy(euclidean_values)}")
    plt.tight_layout()
    euc_image = figure_to_image(plt.gcf())
    plt.clf()

    plt.figure(figsize=(20,20))
    plt.barh(keys, cosine_values[sort_idx], color=colors)
    plt.title(f"Entropy: {entropy(cosine_values)}")
    plt.tight_layout()
    cos_image = figure_to_image(plt.gcf())
    plt.clf()

    return euc_image, cos_image, sort_idx


def main(artist, timesteps, seeds, n_steps, outpath):

    os.makedirs(outpath, exist_ok=True)

    diffuser = StableDiffuser().to(torch.device('cuda'))
    diffuser.set_scheduler_timesteps(n_steps)

    artist_prompt = f"{artist}"
    blank_prompt = ""

    artist_text_embeddings = diffuser.get_text_embeddings([artist_prompt],n_imgs=1)
    blank_text_embeddings = diffuser.get_text_embeddings([blank_prompt],n_imgs=1)

    artist_tokens = diffuser.text_tokenize(artist_prompt)['input_ids'][0]
    blank_tokens = diffuser.text_tokenize(blank_prompt)['input_ids'][0]

    artist_tokens_idxs = torch.argwhere(artist_tokens != blank_tokens)[:, 0]

    layers = []

    for name, module in diffuser.named_modules():

        # if 'lhook' in name and 'attn1' not in name:

        #     layers.append(name)

        if 'hshook'in name and 'attn1' not in name:

            layers.append(name)

        if 'attnhook' in name and 'attn1' not in name:

            layers.append(name)

    seed_comparisons = []

    for seed in seeds:

        seed_outpath = os.path.join(outpath, str(seed))
        os.makedirs(seed_outpath, exist_ok=True)

        diffuser.seed(seed)

        latents = diffuser.get_initial_latents(1, 512, 1)

        start_timestep = 0

        timestep_comparisons = []

        for end_timestep in timesteps:

            timestep_outpath = os.path.join(seed_outpath, str(end_timestep))
            os.makedirs(timestep_outpath, exist_ok=True)

            latents_steps, _ = diffuser.diffusion(
                latents,
                blank_text_embeddings,
                start_iteration=start_timestep,
                end_iteration=end_timestep
            )

            with torch.no_grad():

                with TraceDict(diffuser, layers=layers, retain_output=True) as artist_td:
                    artist_latents = diffuser.predict_noise(end_timestep, latents_steps[0], artist_text_embeddings)

                with TraceDict(diffuser, layers=layers, retain_output=True) as blank_td:
                    blank_latents = diffuser.predict_noise(end_timestep, latents_steps[0], blank_text_embeddings)

            latents = latents_steps[0]
            start_timestep = end_timestep

            comparison = compare(artist_td, blank_td, artist_tokens_idxs)

            euc_path = os.path.join(timestep_outpath, 'euclidean.png')
            cos_path = os.path.join(timestep_outpath, 'cosine.png')

            euc_image, cos_image, _ = vis_comparison(comparison)

            euc_image.save(euc_path)
            cos_image.save(cos_path)

            timestep_comparisons.append(comparison)

        plot_gif(timestep_comparisons, seed_outpath)

        timestep_comparison = average_comparisons(timestep_comparisons)

        euc_path = os.path.join(seed_outpath, 'euclidean.png')
        cos_path = os.path.join(seed_outpath, 'cosine.png')

        euc_image, cos_image, _ = vis_comparison(timestep_comparison)

        euc_image.save(euc_path)
        cos_image.save(cos_path)

        seed_comparisons.append(timestep_comparisons)

    
    seed_comparison = average_comparisons([comparison for timestep_comparisons in seed_comparisons for comparison in timestep_comparisons])

    euc_path = os.path.join(outpath, 'euclidean.png')
    cos_path = os.path.join(outpath, 'cosine.png')

    euc_image, cos_image, _ = vis_comparison(seed_comparison)

    euc_image.save(euc_path)
    cos_image.save(cos_path)

    overtime_comparisons = []

    for timestep in range(len(seed_comparisons[0])):

        comparisons = []

        for timestep_comparisons in seed_comparisons:

            comparisons.append(timestep_comparisons[timestep])

        overtime_comparisons.append(average_comparisons(comparisons))

    plot_gif(overtime_comparisons, outpath)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--artist', required=True)
    parser.add_argument('--timesteps', nargs='+', type=int, default=[5,10,15,20,25,30,35,40,45])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1,2,3,4,5, 6, 7, 8, 9, 10])
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--outpath', default='./')

    main(**vars(parser.parse_args()))