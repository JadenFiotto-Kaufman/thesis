import numpy as np
import torch
from matplotlib import pyplot as plt
import os
from ... import util
from ..attention.attention import AttentionHookModule, group_by_type, stack_attentions
from functools import partial
from ...StableDiffuser import StableDiffuser

def manipulate(data, name, name_to_head, artist_tokens_idxs):

    key, value, encoder_hidden_states, attn = data

    encoder_hidden_states = encoder_hidden_states.clone()

    if 'attn2' in name:

        for module_name in name_to_head:
            
            if module_name in name:

                encoder_hidden_states[1, artist_tokens_idxs] = encoder_hidden_states[0, artist_tokens_idxs]

                uncond_key = attn.to_k(encoder_hidden_states)
                uncond_value = attn.to_v(encoder_hidden_states)
                uncond_key = attn.head_to_batch_dim(uncond_key)
                uncond_value = attn.head_to_batch_dim(uncond_value)

                for head in name_to_head[module_name]:

                    key[head] = uncond_key[head]
                    value[head] = uncond_value[head]

                if len(name_to_head[module_name]) == 0:

                    key[:] = uncond_key[:]
                    value[:] = uncond_value[:]

                break


    return key, value


def zap_head(activation, name, name_to_head, artist_tokens_idxs):

    for key in name_to_head:
        
        if key in name:

            for head in name_to_head[key]:

                activation[head] = activation[head-8]

            if len(name_to_head[key]) == 0:

                activation[8:] = activation[:8]

            return activation

def main(prompt, outpath, device):

    os.makedirs(outpath, exist_ok=True)

    diffuser = StableDiffuser(scheduler='DDIM').to(torch.device(device))

    layers = set([module_name for module_name, module in diffuser.named_modules() if 'manipkeyvalue' in module_name])
 
    name_to_head = {
        'unet.down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor' : [],
        'unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor' : [],
        'unet.down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor' : [],
        'unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor' : [],
        'unet.down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor' : [],
        'unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor' : [],
        # 'unet.mid_block.attentions.0.transformer_blocks.0.attn2.processor' : [],
        #'unet.up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor' : [14, 13,12,11],
        #'unet.up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor' : [14, 10, 8],
        #'unet.up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor' : [15, 14, 13, 11, 9, 8],
        # 'unet.up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor' : [],
        # 'unet.up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor' : [],
        #'unet.up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor' : [13, 10, 9],
        # 'unet.up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor' : [],
        # 'unet.up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor' : [],
        # 'unet.up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor' : [],
    }

    artist = 'Van Gogh\'s'

    prompt_tokens = diffuser.text_tokenize(prompt)['input_ids'][0]
    rem_tokens = diffuser.text_tokenize(prompt.replace(artist, ''))['input_ids'][0]

    artist_tokens_idxs = torch.argwhere(prompt_tokens != rem_tokens)[:, 0]

    edit_fn = partial(manipulate, name_to_head=name_to_head, artist_tokens_idxs=artist_tokens_idxs)

    seed = 23

    generator = torch.manual_seed(seed)

    images, _ = diffuser(
        prompt,
        n_steps=50, 
        n_imgs=1,
        generator=generator,
        trace_args={'layers' : layers, 'edit_output': partial(manipulate, name_to_head={}, artist_tokens_idxs=[]), 'retain_output': False}
    )

    orig_image = images[0][0]

    generator = torch.manual_seed(seed)

    images, trace_steps = diffuser(
        prompt,
        n_steps=50, 
        n_imgs=1,
        generator=generator,
        trace_args={'layers' : layers, 'edit_output': edit_fn, 'retain_output': False}
    )

    edited_image = images[0][0]

    util.image_grid([[orig_image, edited_image]], outpath=os.path.join(outpath, 'image.png'))




if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('prompt')
    parser.add_argument('outpath')
    parser.add_argument('--device', default='cuda:0')

    main(**vars(parser.parse_args()))

