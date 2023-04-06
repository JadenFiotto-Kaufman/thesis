import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ...experiments.attention.attention import (AttentionHookModule,
                                                group_by_type, low_mem,
                                                stack_attentions)
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

def plot(data, outpath):

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

    data = PCA(n_components=2).fit_transform(data)

    for cls in np.unique(clss):

        idx = np.argwhere(clss == cls)[:, 0]

        plt.scatter([data[idx,0]], [data[idx,1]], label=cls)

        for i in idx:
            plt.text(data[i,0], data[i, 1], keys[i])
    
    plt.legend(loc='upper right')
    plt.savefig(f'{outpath}.png')
    plt.clf()
    plt.close()

def main(inpath, outpath, device):

    diffuser = StableDiffuser(scheduler='EA').to(torch.device(device)).half()

    layers = set([module_name for module_name, module in diffuser.named_modules() if (module_name.endswith('attnkeyhook') or module_name.endswith('attnvaluehook')) and 'attn2' in module_name])

    generator = torch.manual_seed(42)

    data = pd.read_csv(inpath)

    result = {}

    for i, row in data.iterrows():

        prompt = row['prompt']
        cls = row['cls']

        tokens = diffuser.text_tokenize([prompt])['input_ids'][0][1:]
        tokens = diffuser.text_detokenize(tokens)

        images, trace_steps = diffuser(
            prompt,
            generator=generator,
            n_steps=1, 
            trace_args={'layers' : layers}
        )

        trace = trace_steps[0]

        for key in trace:

            output = trace[key].output[8:, 1:len(tokens)+1, :].cpu().numpy().mean(axis=1)

            for head in range(output.shape[0]):

                _key = f"{key}.{head}"

                _key = edit_key(_key)

                if _key not in result:

                    result[_key] = {prompt: {'data': output[head], 'cls':cls}}

                else:
                    result[_key][prompt] = {'data': output[head], 'cls':cls}

    os.makedirs(outpath, exist_ok=True)

    for key in result.keys():

        _outpath = os.path.join(outpath, f"{key}.png")

        plot(result[key], _outpath)



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('inpath')
    parser.add_argument('outpath')
    parser.add_argument('--device', default='cuda:0')

    main(**vars(parser.parse_args()))

