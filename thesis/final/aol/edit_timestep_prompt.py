
import os

import torch
from matplotlib import pyplot as plt

import torch
from diffusers.models import cross_attention
from functools import partial
from ...StableDiffuser import StableDiffuser


class _CrossAttnProcessor(cross_attention.CrossAttnProcessor):

    blank_text_embeddings = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):

        encoder_hidden_states =  _CrossAttnProcessor.blank_text_embeddings

        return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask)



def reset(diffuser):

    for module_name, module in diffuser.named_modules():
    
        if module_name.endswith('attn2'):

            module.processor = cross_attention.CrossAttnProcessor()



def run(diffuser, prompt, layers):

    for module_name, module in diffuser.named_modules():
    
        if module_name in layers:

            module.processor = _CrossAttnProcessor()


    images = diffuser(
            prompt,
            generator=torch.manual_seed(123123),
            n_steps=100, 
            n_imgs=1,
        )
    

    reset(diffuser)


    return images[0][0]

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

    blank_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=1)

    _CrossAttnProcessor.blank_text_embeddings = blank_text_embeddings

    os.makedirs(outpath, exist_ok=True)

    run(diffuser, prompt, []).save(os.path.join(outpath, 'control.png'))

    run(diffuser, prompt, down_layers).save(os.path.join(outpath, f'r_down.png')) 
    run(diffuser, prompt, up_layers).save(os.path.join(outpath, f'r_up.png')) 
    run(diffuser, prompt, low_res).save(os.path.join(outpath, f'r_lowres.png')) 
    run(diffuser, prompt, high_res).save(os.path.join(outpath, f'r_highres.png')) 


    for layer in layers:

        run(diffuser, prompt, [layer]).save(os.path.join(outpath, f'r_{layer}.png')) 

    for layer in layers:

        run(diffuser, prompt, get_o(layers, [layer])).save(os.path.join(outpath, f'o_{layer}.png')) 

    run(diffuser, prompt, get_o(layers, down_layers)).save(os.path.join(outpath, f'o_down.png')) 
    run(diffuser, prompt, get_o(layers, up_layers)).save(os.path.join(outpath, f'o_up.png')) 
    run(diffuser, prompt, get_o(layers, low_res)).save(os.path.join(outpath, f'o_lowres.png')) 
    run(diffuser, prompt, get_o(layers, high_res)).save(os.path.join(outpath, f'o_highres.png')) 
    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt')
    parser.add_argument('outpath')

    main(**vars(parser.parse_args()))