import numpy as np
import torch
from matplotlib import pyplot as plt
import os
from ... import util
from ..attention.attention import AttentionHookModule, group_by_type, stack_attentions

from ...StableDiffuser import StableDiffuser
import random
def main(prompts, outpath, seeds, device):

    os.makedirs(outpath, exist_ok=True)


    diffuser = StableDiffuser(scheduler='DDIM').to(torch.device(device))

    layers = set([module_name for module_name, module in diffuser.named_modules() if 'attnprobshook' in module_name and 'attn2' in module_name])




    data = {}

    
    for seed in seeds:

        generator = torch.manual_seed(seed)

        prompt = prompts[random.randint(0, len(prompts)-1)]

        images, trace_steps = diffuser(
            prompt,
            generator=generator,
            n_steps=50, 
            n_imgs=1,
            trace_args={'layers' : layers}
        )

        attentions = stack_attentions(trace_steps)

        self_attentions, cross_attentions = group_by_type(attentions)


        tokens = diffuser.text_tokenize([prompt])['input_ids'][0][1:]
        tokens = diffuser.text_detokenize(tokens)


        for key in cross_attentions:

            cross_attention = cross_attentions[key].mean(dim=2)
            if key not in data:

                data[key] = cross_attention.mean(dim=(0,1))[1:len(tokens)+1].sum()
        
            else:

                data[key] += cross_attention.mean(dim=(0,1))[1:len(tokens)+1].sum()

      

        


    for key in data:

        data[key] = data[key] / len(seeds)


    isort = np.array(list(data.keys())).argsort()
    values = np.array(list(data.values()))[isort]
    keys = np.array(list(data.keys()))[isort]


    down = 0
    up = 0

    for i, key in enumerate(keys):

        if 'down' in key:

            keys[i] = f'down{down}'

            down += 1    

        elif 'up' in key:
            keys[i] = f'up{up}'

            up += 1

        else:

            keys[i] = 'mid'




    plt.plot(range(len(values)), values)
    ci = 1.96 * np.std(values)/np.sqrt(len(keys))
    plt.fill_between(range(len(values)), (values-ci), (values+ci), color='b', alpha=.1)
    #plt.ylim(0, _max)
    plt.legend(loc="upper right")
    # plt.title(f"<SOS> prob: {}")
    plt.ylabel('Softmax Prob')
    plt.xlabel('Layer')

    plt.xticks(range(len(values)), keys, size='small')
    plt.savefig(os.path.join(outpath, f"aol.png"))
    plt.clf()
    plt.close()
    breakpoint()
    all /= len(data)

    all = all[:,1:len(tokens)].sum(dim=1)

  
    plt.plot(range(50), all.cpu().numpy())
    # plt.title(f"<SOS> prob: {}")
    plt.ylabel('Softmax Prob')
    plt.xlabel('Timestep')
    plt.savefig(os.path.join(outpath, f"all.png"))
    plt.clf()
    plt.close()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('prompts', nargs='+')
    parser.add_argument('outpath')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    parser.add_argument('--device', default='cuda:0')

    main(**vars(parser.parse_args()))

