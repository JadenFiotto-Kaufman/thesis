import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import pickle
from diffusers.models import cross_attention

from ...StableDiffuser import StableDiffuser


def edit_key(key):

    key = key.replace('unet.', '')
    key = key.replace('_blocks', '')
    key = key.replace('attentions.', '')
    key = key.replace('transformer.', '')
    key = key.replace('attn2.processor.', '')
    key = key.replace('attn', '')
    key = key.replace('hook', '')

    return key

def plot(data, pca, outpath):

    clss = []
    keys = []
    _data = []

    for key in data.keys():

        clss.append(data[key]['cls'])
        _data.append(data[key]['data'])
        keys.append(key)

    data = np.stack(_data)
    clss = np.array(clss)
    keys = np.array(keys)

    data = pca.transform(data)

    for cls in np.unique(clss):

        idx = np.argwhere(clss == cls)[:, 0]

        plt.scatter([data[idx,0]], [data[idx,1]], label=cls)

        for i in idx:
            plt.text(data[i,0], data[i, 1], keys[i])
    
    plt.legend(loc='upper right')
    plt.savefig(f'{outpath}.png')
    plt.clf()
    plt.close()
    
@torch.no_grad()
def main(inpath, pcas_path, outpath, device):

    diffuser = StableDiffuser(scheduler='LMS').to(device)

    layers = [(module_name, module) for module_name, module in diffuser.named_modules() if isinstance(module, cross_attention.CrossAttention) and 'attn2' in module_name]

    data = pd.read_csv(inpath)

    with open(pcas_path, 'rb') as file:

        pcas = pickle.load(file)

    result = {}

    for i, row in data.iterrows():

        prompt = row['prompt']
        cls = row['cls']

        encoder_hidden_states = diffuser.get_text_embeddings([prompt], n_imgs=1)
        
        tokens = diffuser.text_tokenize([prompt])['input_ids'][0][1:]
        tokens = diffuser.text_detokenize(tokens)

        for key, attn in layers:

            value = attn.to_v(encoder_hidden_states)
            value = attn.head_to_batch_dim(value)
            value = value[8:]

            attention_probs = torch.zeros((8,1,77)).to(device)

            attention_probs[:,:,1:len(tokens)+1] = 1 / len(tokens)

            hidden_states = torch.bmm(attention_probs, value)
        
            hidden_states = hidden_states.cpu().numpy()[:,0]

            for head in range(hidden_states.shape[0]):

                _key = f"{key}.{head}"

                if _key not in result:

                    result[_key] = {prompt: {'data': hidden_states[head], 'cls':cls}}

                else:
                    result[_key][prompt] = {'data': hidden_states[head], 'cls':cls}

    os.makedirs(outpath, exist_ok=True)

    for key in result.keys():

        _outpath = os.path.join(outpath, f"{edit_key(key)}.png")

        plot(result[key], pcas[key], _outpath)



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('inpath')
    parser.add_argument('pcas_path')
    parser.add_argument('outpath')
    parser.add_argument('--device', default='cuda:0')

    main(**vars(parser.parse_args()))

