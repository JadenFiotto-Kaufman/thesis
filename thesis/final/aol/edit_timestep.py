
import os

import torch
from baukit import TraceDict
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from ... import util
from ...StableDiffuser import StableDiffuser


def remove(activations, name):

    activations[:] = 0.0

    return activations

@torch.no_grad()
def run(diffuser, prompt_text_embeddings, layers):

    latents = diffuser.get_initial_latents(1, 512, 1, generator = torch.manual_seed(123123))

    for iteration in tqdm(range(100)):

        with TraceDict(diffuser,layers=layers, edit_output=remove, retain_output=False) as td:

            noise_pred = diffuser.predict_noise(
                iteration, 
                latents, 
                prompt_text_embeddings)
        
        output = diffuser.scheduler.step(noise_pred, diffuser.scheduler.timesteps[iteration], latents)
        
        latents = output.prev_sample

        #pred_original_samples.append(output.pred_original_sample.cpu())

    return diffuser.to_image(diffuser.decode(latents).cpu().float())[0]

def get_o(all_layers, layers):

    return [layer for layer in all_layers if layer not in layers]


def main(prompt, outpath):

    diffuser = StableDiffuser(scheduler='EA').to(torch.device('cuda:0')).half()
    diffuser.set_scheduler_timesteps(100)

    layers = set([module_name for module_name, module in diffuser.named_modules() if module_name.endswith('attn2')])

    up_layers = [layer for layer in layers if 'up' in layer]
    down_layers = [layer for layer in layers if 'down' in layer]
    high_res = ['unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2',
                'unet.down_blocks.0.attentions.1.transformer_blocks.0.attn2',
                'unet.up_blocks.3.attentions.0.transformer_blocks.0.attn2',
                'unet.up_blocks.3.attentions.1.transformer_blocks.0.attn2',
                'unet.up_blocks.3.attentions.2.transformer_blocks.0.attn2'
                ]
    
    low_res = ['unet.mid_block.attentions.0.transformer_blocks.0.attn2',
                'unet.down_blocks.2.attentions.0.transformer_blocks.0.attn2',
                'unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2',
                'unet.up_blocks.1.attentions.0.transformer_blocks.0.attn2',
                'unet.up_blocks.1.attentions.1.transformer_blocks.0.attn2',
                'unet.up_blocks.1.attentions.2.transformer_blocks.0.attn2',
                ]

    prompt_text_embeddings = diffuser.get_text_embeddings([prompt],n_imgs=1)

    os.makedirs(outpath, exist_ok=True)

    run(diffuser, prompt_text_embeddings, layers).save(os.path.join(outpath, 'rall.png'))

    run(diffuser, prompt_text_embeddings, []).save(os.path.join(outpath, 'control.png'))

    run(diffuser, prompt_text_embeddings, down_layers).save(os.path.join(outpath, f'r_down.png')) 
    run(diffuser, prompt_text_embeddings, up_layers).save(os.path.join(outpath, f'r_up.png')) 
    run(diffuser, prompt_text_embeddings, low_res).save(os.path.join(outpath, f'r_lowres.png')) 
    run(diffuser, prompt_text_embeddings, high_res).save(os.path.join(outpath, f'r_highres.png')) 


    for layer in layers:

        run(diffuser, prompt_text_embeddings, [layer]).save(os.path.join(outpath, f'r_{layer}.png')) 

    for layer in layers:

        run(diffuser, prompt_text_embeddings, get_o(layers, [layer])).save(os.path.join(outpath, f'o_{layer}.png')) 

    run(diffuser, prompt_text_embeddings, get_o(layers, down_layers)).save(os.path.join(outpath, f'o_down.png')) 
    run(diffuser, prompt_text_embeddings, get_o(layers, up_layers)).save(os.path.join(outpath, f'o_up.png')) 
    run(diffuser, prompt_text_embeddings, get_o(layers, low_res)).save(os.path.join(outpath, f'o_lowres.png')) 
    run(diffuser, prompt_text_embeddings, get_o(layers, high_res)).save(os.path.join(outpath, f'o_highres.png')) 

    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt')
    parser.add_argument('outpath')

    main(**vars(parser.parse_args()))