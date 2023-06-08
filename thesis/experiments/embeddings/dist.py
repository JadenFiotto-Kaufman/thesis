import os

import numpy as np
import pandas as pd
import torch

from ...StableDiffuser import StableDiffuser
from diffusers.models import cross_attention
from .get_embedding import get_embeddings

def edit_key(key):

    key = key.replace('unet.', '')
    key = key.replace('_blocks', '')
    key = key.replace('attentions.', '')
    key = key.replace('transformer.', '')
    key = key.replace('attn2.processor.', '')
    key = key.replace('attn', '')
    key = key.replace('hook', '')

    return key

@torch.no_grad()
def main(artist, styles, outpath, device):

    diffuser = StableDiffuser(scheduler='EA').to(torch.device(device))

    styles = [artist] + styles

    result = get_embeddings(diffuser, styles)

    score = [0] * len(styles[1:])

    # hmm = ['unet.down_blocks.2.attentions.0.transformer_blocks.0.attn2.4',
    # 'unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2.0',
    # 'unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2.7',
    # 'unet.up_blocks.1.attentions.1.transformer_blocks.0.attn2.0',
    # 'unet.up_blocks.1.attentions.1.transformer_blocks.0.attn2.6',
    # 'unet.up_blocks.1.attentions.2.transformer_blocks.0.attn2.5',
    # 'unet.up_blocks.1.attentions.2.transformer_blocks.0.attn2.6',
    # 'unet.up_blocks.2.attentions.1.transformer_blocks.0.attn2.5']


    for key in result:

        #print(f"==> {edit_key(key)}")

        data = result[key]

        key_dist= []

        for i in range(data.shape[0]):

            if i == 0: continue

            dist = np.linalg.norm(data[0] - data[i])

            #print(f"====> {styles[i]}: {dist}")

            score[i-1] += dist

            key_dist.append(dist)

        correct = np.array(key_dist).argmin() == 0

        if correct:

            print(key)

    idxs = np.array(score).argsort()

    styles = np.array(styles[1:])

    for idx in idxs:

        print(styles[idx], score[idx])




if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--artist')
    parser.add_argument('--styles', nargs='+')
    parser.add_argument('--outpath')
    parser.add_argument('--device', default='cuda:0')

    main(**vars(parser.parse_args()))

