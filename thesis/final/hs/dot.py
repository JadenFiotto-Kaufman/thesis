import os

import numpy as np
import pandas as pd
import torch

from ...StableDiffuser import StableDiffuser
from diffusers.models import cross_attention
from .get_embedding import get_embeddings
import scipy
def cos_sim(a,b):

    return np.array([scipy.spatial.distance.cosine(a[i], b[0]) for i in range(a.shape[0])])


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
def main(inpath, outpath, device):

    diffuser = StableDiffuser(scheduler='EA').to(torch.device(device))

    df = pd.read_csv(inpath)

    result = get_embeddings(diffuser, df['prompt'])

    

    # hmm = ['unet.down_blocks.2.attentions.0.transformer_blocks.0.attn2.4',
    # 'unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2.0',
    # 'unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2.7',
    # 'unet.up_blocks.1.attentions.1.transformer_blocks.0.attn2.0',
    # 'unet.up_blocks.1.attentions.1.transformer_blocks.0.attn2.6',
    # 'unet.up_blocks.1.attentions.2.transformer_blocks.0.attn2.5',
    # 'unet.up_blocks.1.attentions.2.transformer_blocks.0.attn2.6',
    # 'unet.up_blocks.2.attentions.1.transformer_blocks.0.attn2.5']


    output = []

    for i, row in df.iterrows():

        prompt = row['prompt']

        print(f"=> '{prompt}'")

        for key in result:

            print(f"==> {edit_key(key)}")

            data = result[key]

            #distances = np.linalg.norm(data - data[[i]], axis=1)
            distances = cos_sim(data, data[[i]])

            sort_idx = distances.argsort()

            _output = [prompt, edit_key(key)]

            for di in range(5):

                _output.append(df['prompt'].iloc[sort_idx[di+1]])

                print(f"===> {distances[sort_idx[di+1]]}:'{df['prompt'].iloc[sort_idx[di+1]]}'")

            output.append(_output)

            columns = ['prompt', 'layer'] + [f"top{n+1}" for n in range(len(output[0]) - 2)]

    output = pd.DataFrame(output, columns=columns)

    breakpoint()




if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('inpath')
    parser.add_argument('outpath')
    parser.add_argument('--device', default='cuda:0')

    main(**vars(parser.parse_args()))

  