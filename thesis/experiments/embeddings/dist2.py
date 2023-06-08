import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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

    dists = [[] for i in range(len(styles) -1)]

    for key in result:

        #print(f"==> {edit_key(key)}")

        data = result[key]

        for i in range(data.shape[0]):

            if i == 0: continue

            dist = np.linalg.norm(data[0] - data[i])

            #print(f"====> {styles[i]}: {dist}")

            dists[i-1].append(dist)



    os.makedirs(outpath, exist_ok=True)


    for i in range(len(dists)):

        dist = dists[i]

        prompt = styles[i+1]

        plt.plot(range(len(dist)), dist)

        plt.savefig(os.path.join(outpath, f"{prompt}.png"))
        plt.clf()

        plt.close()







if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--artist')
    parser.add_argument('--styles', nargs='+')
    parser.add_argument('--outpath')
    parser.add_argument('--device', default='cuda:0')

    main(**vars(parser.parse_args()))

