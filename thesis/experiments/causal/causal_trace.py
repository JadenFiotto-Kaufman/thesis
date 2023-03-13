import os

import pandas as pd
import torch
import re
from ...StableDiffuser import StableDiffuser
from .caching import ActivationCache
from .CLIP import CLIP


def add_noise_to_conditional_text_embedding(embedding, prompt_tokens_idxs, generator, variance=.1):

    embedding[1][prompt_tokens_idxs] = embedding[1][prompt_tokens_idxs] + (variance**0.5)*torch.randn(embedding[1][prompt_tokens_idxs].shape, generator=generator).to(embedding.device)

    return embedding


def eval(clip, prompt, images, outpath):

    images[0][0].save(outpath)

    logits, probs = clip([prompt], images[0][0])

    clip_score = logits[0][0].item()

    return clip_score

def diffusion(diffuser, nsteps, latents, text_embeddings, trace_args=None):

    latents_steps, trace_steps = diffuser.diffusion(
        latents,
        text_embeddings,
        end_iteration=nsteps,
        guidance_scale=7.5,
        trace_args=trace_args
    )

    latents_steps = [diffuser.decode(latents.cuda()) for latents in latents_steps]
    images_steps = [diffuser.to_image(latents) for latents in latents_steps]

    return images_steps

def main(prompt, nsteps, outpath, module_regex):
    
    seed = 42
    variance = 2.75

    os.makedirs(outpath, exist_ok=True)

    device = torch.device('cuda')

    diffuser = StableDiffuser(seed=seed).to(device)
    diffuser.set_scheduler_timesteps(nsteps)
    
    clip = CLIP(device)

    layers = []

    for module_name, module in diffuser.named_modules():

        match = re.search(module_regex, module_name)
        
        if match is not None:
            print(f"=> {module_name}")

            layers.append(module_name)

    images, _ = diffuser(prompt,
                reseed=True,
                trace_args={'layers': layers, 
                            'edit_output' : ActivationCache.cache,
                            'retain_output':False},
                guidance_scale=7.5,
                n_steps=nsteps
                )
    
    original_clip_score = eval(clip, prompt, images, os.path.join(outpath,'original.png'))

    print(f"Original CLIP score: {original_clip_score}")

    diffuser.seed(diffuser._seed)

    latents = diffuser.get_initial_latents(1, 512, 1)

    prompt_tokens = diffuser.text_tokenize(prompt)['input_ids'][0]
    blank_tokens = diffuser.text_tokenize("")['input_ids'][0]

    prompt_tokens_idxs = torch.argwhere(prompt_tokens != blank_tokens)[:, 0]

    text_embeddings = diffuser.get_text_embeddings([prompt],n_imgs=1)

    corrupted_text_embeddings = add_noise_to_conditional_text_embedding(text_embeddings, prompt_tokens_idxs, diffuser.generator, variance=variance)
    
    images = diffusion(diffuser, nsteps, latents, corrupted_text_embeddings)

    corrupted_clip_score = eval(clip, prompt, images, os.path.join(outpath,'corrupted.png'))

    print(f"Corrupted CLIP score: {corrupted_clip_score}\nDifference: {original_clip_score - corrupted_clip_score}")

    data = []

    for layer in layers:

        ActivationCache.reset()

        trace_args = {
            'layers': [layer], 
            'edit_output' : ActivationCache.load,
            'retain_output':False}

        images = diffusion(diffuser, nsteps, latents, corrupted_text_embeddings, trace_args=trace_args)

        _corrupted_clip_score = eval(clip, prompt, images, os.path.join(outpath,f'{layer}.png'))
        difference = original_clip_score - _corrupted_clip_score

        print(f"{layer} CLIP score: {_corrupted_clip_score}\nDifference: {difference}")
    
        data.append((prompt, seed, variance, layer, difference))

    results_path = os.path.join(outpath, 'result.csv')

    if not os.path.exists(results_path):

        pd.DataFrame(data, columns=['prompt', 'seed', 'variance', 'layer', 'diff']).to_csv(results_path, index=False)

    else:

        pd.DataFrame(data, columns=['prompt', 'seed', 'variance', 'layer', 'diff']).to_csv(results_path, index=False, header=False)





if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--outpath', required=True)
    parser.add_argument('--module_regex', required=True)
    parser.add_argument('--nsteps', type=int, default=50)

    main(**vars(parser.parse_args()))

    