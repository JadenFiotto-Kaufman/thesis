import numpy as np
import torch
from matplotlib import pyplot as plt
import os
from ... import util
from ..attention.attention import AttentionHookModule, group_by_type, stack_attentions

from ...StableDiffuser import StableDiffuser

def main(prompt, outpath, seeds, device):

    os.makedirs(outpath, exist_ok=True)


    diffuser = StableDiffuser(scheduler='DDIM').to(torch.device(device))

    layers = set([module_name for module_name, module in diffuser.named_modules() if 'attnprobshook' in module_name and 'attn2' in module_name])

    tokens = diffuser.text_tokenize([prompt])['input_ids'][0][1:]
    tokens = diffuser.text_detokenize(tokens)

    data = {}

    
    for seed in seeds:

        generator = torch.manual_seed(seed)

        images, trace_steps = diffuser(
            prompt,
            generator=generator,
            n_steps=50, 
            n_imgs=1,
            trace_args={'layers' : layers}
        )

        attentions = stack_attentions(trace_steps)

        self_attentions, cross_attentions = group_by_type(attentions)

        for key in cross_attentions:

            cross_attention = cross_attentions[key].mean(dim=2)

            if key not in data:

                data[key] = cross_attention
        
            else:

                data[key] += cross_attention

    tokens = ['<SOS>'] + tokens

    all = torch.zeros((50,77)).to(device)

    _max = 0

    for key in data:

        data[key] = data[key] / len(seeds)
        _max = max(_max, data[key][:,:, 1:].max().item())

    for key in data:

        cross_attention = data[key]

        all += cross_attention.mean(dim=1)
                    
        for head in range(cross_attention.shape[1]):
                
            if head <= 7:
                continue

            for i in range(len(tokens)):

                if i == 0:
                    continue

                plt.plot(range(50), cross_attention[:, head, i].cpu().numpy(), label=tokens[i])
            plt.ylim(0, _max)
            plt.legend(loc="upper right")
            # plt.title(f"<SOS> prob: {}")
            plt.ylabel('Softmax Prob')
            plt.xlabel('Timestep')
            plt.savefig(os.path.join(outpath, f"{key}.{head}.png"))
            plt.clf()
            plt.close()

    all /= len(data)

    for i in range(len(tokens)):

        if i == 0:
            continue

        plt.plot(range(50), all[:, i].cpu().numpy(), label=tokens[i])

    plt.legend(loc="upper right")
    # plt.title(f"<SOS> prob: {}")
    plt.ylabel('Softmax Prob')
    plt.xlabel('Timestep')
    plt.savefig(os.path.join(outpath, f"all.png"))
    plt.clf()
    plt.close()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('prompt')
    parser.add_argument('outpath')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1,2,3,4,5,6,7,8,9,10])
    parser.add_argument('--device', default='cuda:0')

    main(**vars(parser.parse_args()))

