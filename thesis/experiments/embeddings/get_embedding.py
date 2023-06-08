import os

import numpy as np
import torch
import json
import random
from ...StableDiffuser import StableDiffuser

from diffusers.models import cross_attention


def get_embeddings(diffuser, prompts):

    layers = [(module_name, module) for module_name, module in diffuser.named_modules() if isinstance(module, cross_attention.CrossAttention) and 'attn2' in module_name]

    results = {}

    for prompt in prompts:

        encoder_hidden_states = diffuser.get_text_embeddings([prompt], n_imgs=1)
        
        tokens = diffuser.text_tokenize([prompt])['input_ids'][0][1:]
        tokens = diffuser.text_detokenize(tokens)

        for key, attn in layers:

            value = attn.to_v(encoder_hidden_states)
            value = attn.head_to_batch_dim(value)
            value = value[8:]

            attention_probs = torch.zeros((8,1,77)).to(value.device)

            attention_probs[:,:,1:len(tokens)+1] = 1 / len(tokens)

            hidden_states = torch.bmm(attention_probs, value)
        
            hidden_states = hidden_states.cpu().numpy()[:,0]

            for head in range(hidden_states.shape[0]):

                _key = f"{key}.{head}"

                if _key not in results:

                    results[_key] = [hidden_states[head]]

                else:

                    results[_key].append(hidden_states[head])


    for key in results:

        results[key] = np.stack(results[key])
        
    return results


@torch.no_grad()
def main(inpath, outpath, device):

    device = torch.device(device)

    diffuser = StableDiffuser(scheduler='LMS').to(device)

    layers = [(module_name, module) for module_name, module in diffuser.named_modules() if isinstance(module, cross_attention.CrossAttention) and 'attn2' in module_name]

    os.makedirs(outpath, exist_ok=True)

    with open(inpath, 'r') as file:

        dataset = json.load(file)

    results = {}

    ids = []

    random.shuffle(dataset['annotations'])

    dataset['annotations'] = dataset['annotations'][:2000]

    for caption in dataset['annotations']:

        id = caption['id']
        caption = caption['caption']

        encoder_hidden_states = diffuser.get_text_embeddings([caption], n_imgs=1)
        
        tokens = diffuser.text_tokenize([caption])['input_ids'][0][1:]
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

                if _key not in results:

                    results[_key] = [hidden_states[head]]

                else:

                    results[_key].append(hidden_states[head])
        
        ids.append(id)

    ids = np.array(ids)

    for key in results:

        results[key] = np.stack(results[key])

    np.save(os.path.join(outpath, 'ids.npy'), ids)
    np.savez(os.path.join(outpath, 'embds.npz'), **results)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('inpath')
    parser.add_argument('outpath')
    parser.add_argument('--device', default='cuda:0')

    main(**vars(parser.parse_args()))

