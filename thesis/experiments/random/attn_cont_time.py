import numpy as np
import torch
from matplotlib import pyplot as plt
import os
from ... import util
from ...StableDiffuser import StableDiffuser, default_parser
from ..attention.attention import AttentionHookModule, group_by_type, stack_attentions

def main(prompt, outpath, seeds, device):

    os.makedirs(outpath, exist_ok=True)


    diffuser = StableDiffuser().to(torch.device(device))

    layers = set([module_name for module_name, module in diffuser.named_modules() if module_name.endswith('attn2')])


    tokens = diffuser.text_tokenize([prompt])['input_ids'][0][1:]
    tokens = diffuser.text_detokenize(tokens)

    data = {}

    
    for seed in seeds:

        diffuser._seed = seed

        images, trace_steps = diffuser(
            prompt,
            reseed=True,
            n_steps=50, 
            n_imgs=1,
            trace_args={'layers' : layers}
        )

        attentions = stack_attentions(trace_steps)

        self_attentions, cross_attentions = group_by_type(attentions)

        for key in cross_attentions:
            cross_attention = cross_attentions[key].absolute().mean(dim=(1,2,3))

            if key not in data:

                data[key] = cross_attention
        
            else:

                data[key] += cross_attention

    tokens = ['<SOS>'] + tokens


    for key in data:

        cross_attention = data[key] / len(seeds)
                    
       
     
        plt.plot(range(50), cross_attention.cpu().numpy())

        # plt.title(f"<SOS> prob: {}")
        plt.ylabel('Softmax Prob')
        plt.xlabel('Timestep')
        plt.savefig(os.path.join(outpath, f"{key}.png"))
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

