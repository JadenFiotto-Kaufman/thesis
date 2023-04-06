import numpy as np
import torch
from matplotlib import pyplot as plt
import os
from .. import util
from ..experiments.attention.attention import AttentionHookModule, group_by_type, stack_attentions, low_mem
from ..StableDiffuser import StableDiffuser, default_parser
import pandas as pd
import itertools


def prepare_prompts(inpath, diffuser):

    medium_data = pd.read_csv(os.path.join(inpath, 'medium.csv'), names=['word'])['word']
    object_adj_data = pd.read_csv(os.path.join(inpath, 'object_adj.csv'), names=['word'])['word']
    object_data = pd.read_csv(os.path.join(inpath, 'object.csv'), names=['word'])['word']
    preps_data = pd.read_csv(os.path.join(inpath, 'preps.csv'), names=['word'])['word']
    setting_data = pd.read_csv(os.path.join(inpath, 'setting.csv'), names=['word'])['word']
    subject_data = pd.read_csv(os.path.join(inpath, 'subject.csv'), names=['word'])['word']

    combinations = list(itertools.product(*[medium_data, subject_data, preps_data, object_adj_data, object_data, ['in'], setting_data]))

    prompts = []
    token_idxs = []

    for combination in combinations:

        token_idx = 1

        _token_idxs = []

        for word in combination:

            tokens = diffuser.text_tokenize([word])['input_ids'][0][1:]
            tokens = diffuser.text_detokenize(tokens)

            __token_idxs = np.array([token_idx, len(tokens) + token_idx])

            token_idx += len(tokens)

            _token_idxs.append(__token_idxs)

        token_idxs.append(_token_idxs)
        prompts.append(combination)

    return prompts, token_idxs

def plot_attention_over_layer(cross_attentions, prompt_words, outpath):

    for timestep_idx in range(cross_attentions.shape[2]):
        for word_idx in range(cross_attentions.shape[1]):

            word_data = cross_attentions[:, word_idx, timestep_idx]

            plt.plot(range(word_data.shape[0]), word_data, label=prompt_words[word_idx])

        plt.ylim((0, cross_attentions.max()))
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(outpath, f"{timestep_idx}.png"))
        plt.clf()

    cross_attentions = cross_attentions.mean(dim=-1)

    for word_idx in range(cross_attentions.shape[1]):

        word_data = cross_attentions[:, word_idx]

        plt.plot(range(word_data.shape[0]), word_data, label=prompt_words[word_idx])

    plt.ylim((0, cross_attentions.max()))
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(outpath, f"all.png"))
    plt.clf()


def plot_attention_over_time(cross_attentions, prompt_words, outpath):

    for layer_idx in range(cross_attentions.shape[0]):
        for word_idx in range(cross_attentions.shape[1]):

            word_data = cross_attentions[layer_idx, word_idx, :]

            plt.plot(range(word_data.shape[0]), word_data, label=prompt_words[word_idx])

        plt.ylim((0, cross_attentions.max()))
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(outpath, f"{layer_idx}.png"))
        plt.clf()

    cross_attentions = cross_attentions.mean(dim=0)

    for word_idx in range(cross_attentions.shape[0]):

        word_data = cross_attentions[word_idx, :]

        plt.plot(range(word_data.shape[0]), word_data, label=prompt_words[word_idx])

    plt.ylim((0, cross_attentions.max()))
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(outpath, f"all.png"))
    plt.clf()



def combine_cross_attentions(cross_attentons):

    combined = {}

    for key in cross_attentons[0]:

        for i, _cross_attentions in enumerate(cross_attentons):

            if i == 0:

                combined[key] = _cross_attentions

            else:

                combined[key] += _cross_attentions

        combined[key] /= len(cross_attentons)

    return combined


def postprocess(cross_attentions, token_idxs):

    all_data = []

    for key in cross_attentions:

        data = cross_attentions[key].mean(dim=(1,2))

        token_data = []

        for _token_idxs in token_idxs:

            _token_data = data[:, _token_idxs[0]:_token_idxs[1]].max(dim=-1).values

            token_data.append(_token_data)

        token_data = torch.stack(token_data)

        all_data.append(token_data)

    all_data = torch.stack(all_data)

    return all_data

def evaluate(cross_attentions, prompt_words, outpath):

    _outpath = os.path.join(outpath, 'aot')

    os.makedirs(_outpath, exist_ok=True)

    plot_attention_over_time(cross_attentions, prompt_words, _outpath)

    _outpath = os.path.join(outpath, 'aol')

    os.makedirs(_outpath, exist_ok=True)

    plot_attention_over_layer(cross_attentions, prompt_words, _outpath)



def main(inpath, outpath, nsteps, seeds, device):

    diffuser = StableDiffuser(scheduler='EA').to(torch.device(device)).half()

    prompts, token_idxs = prepare_prompts(inpath, diffuser)

    layers = set([module_name for module_name, module in diffuser.named_modules() if 'attnprobshook' in module_name and 'attn2' in module_name])

    all_cross_attentions = []

    for i, prompt_words in enumerate(prompts):

        if i not in [1,10,20,30]:
            continue

        prompt = ' '.join(prompt_words)

        prompt_cross_attentions = []
    
        for seed in seeds:

            generator = torch.manual_seed(seed)

            images, trace_steps = diffuser(
                prompt,
                generator=generator,
                n_steps=nsteps, 
                trace_args={'layers' : layers}
            )

            attentions = stack_attentions(trace_steps)

            self_attentions, cross_attentions = group_by_type(attentions)

            cross_attentions = postprocess(cross_attentions, token_idxs[i])

            _outpath = os.path.join(outpath, str(i), str(seed))

            evaluate(cross_attentions, prompt_words, _outpath)

            _outpath = os.path.join(outpath, str(i), str(seed), f"{'_'.join(prompt_words)}.png")

            images[0][0].save(_outpath)

            prompt_cross_attentions.append(cross_attentions)

        prompt_cross_attentions = torch.stack(prompt_cross_attentions).mean(dim=0)

        _outpath = os.path.join(outpath, str(i))

        evaluate(cross_attentions, prompt_words, _outpath)

        all_cross_attentions.append(prompt_cross_attentions)

    all_cross_attentions = torch.stack(all_cross_attentions).mean(dim=0)

    evaluate(cross_attentions, ['medium_data', 'subject_data', 'preps_data', 'object_adj_data', 'object_data', 'in', 'setting_data'], outpath)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('inpath')
    parser.add_argument('outpath')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1,2,3,4,5,6,7,8,9,10])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--nsteps', type=int, default=50)

    main(**vars(parser.parse_args()))

