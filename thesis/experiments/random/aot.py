import numpy as np
import torch
from matplotlib import pyplot as plt
import os
from ... import util
from ...StableDiffuser import StableDiffuser, default_parser
from ..attention.attention import AttentionHookModule, group_by_type, stack_attentions

def main(outpath):

    os.makedirs(outpath, exist_ok=True)


    prompt = "A woman standing in a jungle with red coat"


    diffuser = StableDiffuser(seed=42).to(torch.device('cuda:1'))

    layers = set([module_name for module_name, module in diffuser.named_modules() if 'attnscoreshook' in module_name and 'attn2' in module_name])

    images, trace_steps = diffuser(prompt,
        n_steps=50, 
        n_imgs=1,
        trace_args={'layers' : layers}
    )

    images[0][0].save(os.path.join(outpath, 'image.png'))

    tokens = diffuser.text_tokenize([prompt])['input_ids'][0][1:]
    tokens = diffuser.text_detokenize(tokens)

    attentions = stack_attentions(trace_steps)

    self_attentions, cross_attentions = group_by_type(attentions)
    
    cross_attention_ts = []

    for key in cross_attentions.keys():

        cross_attention_ts.append(cross_attentions[key].mean(dim=(1,2)))

    cross_attention_ts = torch.stack(cross_attention_ts).mean(dim=0)

    tokens = ['<SOS>'] + tokens


    for key in cross_attentions:

        cross_attention = cross_attentions[key].mean(dim=2)

        for head in range(cross_attention.shape[1]):
            
            if head <= 7:
                continue

            for i in range(len(tokens)):

                if i == 0:
                    continue

                plt.plot(range(50), cross_attention[:, head, i].cpu().numpy(), label=tokens[i])

            plt.legend(loc="upper right")
            plt.savefig(os.path.join(outpath, f"{key}.{head}.png"))
            plt.clf()
            plt.close()

    for i in range(len(tokens)):

        if i == 0:
            continue

        plt.plot(range(50), cross_attention_ts[:, i].cpu().numpy(), label=tokens[i])

    plt.legend(loc="upper right")
    plt.savefig(os.path.join(outpath, f"all.png"))
    plt.clf()
    plt.close()

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('outpath')

    main(**vars(parser.parse_args()))

