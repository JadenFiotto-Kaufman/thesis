
import os

import torch
from baukit import TraceDict
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from ... import util
from ...StableDiffuser import StableDiffuser


@torch.no_grad()
def run(diffuser, prompt_text_embeddings, blank_text_embeddings, remove_iterations, layers):

    latents = diffuser.get_initial_latents(1, 512, 1, generator = torch.manual_seed(50))

    for iteration in tqdm(range(100)):

        noise_pred = diffuser.predict_noise(
            iteration, 
            latents, 
            blank_text_embeddings if iteration in remove_iterations else prompt_text_embeddings)
        
        output = diffuser.scheduler.step(noise_pred, diffuser.scheduler.timesteps[iteration], latents)
        
        latents = output.prev_sample

        #pred_original_samples.append(output.pred_original_sample.cpu())

    return diffuser.to_image(diffuser.decode(latents).cpu().float())[0]


def main(prompt, outpath):

    diffuser = StableDiffuser(scheduler='EA').to(torch.device('cuda:0')).half()
    diffuser.set_scheduler_timesteps(100)

    layers = set([module_name for module_name, module in diffuser.named_modules() if module_name.endswith('attn2')])

    prompt_text_embeddings = diffuser.get_text_embeddings([prompt],n_imgs=1)
    blank_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=1)

    os.makedirs(outpath, exist_ok=True)

    q1 = list(range(0,10))
    q2 = list(range(10,30))
    q3 = list(range(30,60))
    q4 = list(range(60,100))

    run(diffuser, prompt_text_embeddings, blank_text_embeddings, [], layers).save(os.path.join(outpath, 'control.png'))

    run(diffuser, prompt_text_embeddings, blank_text_embeddings,q1, layers).save(os.path.join(outpath, 'rq1.png'))      

    run(diffuser, prompt_text_embeddings, blank_text_embeddings,q2, layers).save(os.path.join(outpath, 'rq2.png'))      

    run(diffuser, prompt_text_embeddings, blank_text_embeddings,q3, layers).save(os.path.join(outpath, 'rq3.png'))      

    run(diffuser, prompt_text_embeddings, blank_text_embeddings,q4, layers).save(os.path.join(outpath, 'rq4.png'))      

    run(diffuser, prompt_text_embeddings, blank_text_embeddings, range(0, 27), layers).save(os.path.join(outpath, 'rearly.png')) 

    run(diffuser, prompt_text_embeddings,blank_text_embeddings,list(range(0,100)), layers).save(os.path.join(outpath, 'rall.png'))   

    run(diffuser, prompt_text_embeddings, blank_text_embeddings,q2 + q3 + q4, layers).save(os.path.join(outpath, 'oq1.png'))      

    run(diffuser, prompt_text_embeddings, blank_text_embeddings,q1 + q3 + q4, layers).save(os.path.join(outpath, 'oq2.png'))      

    run(diffuser, prompt_text_embeddings, blank_text_embeddings,q2 + q1 + q4, layers).save(os.path.join(outpath, 'oq3.png'))      

    run(diffuser, prompt_text_embeddings, blank_text_embeddings,q2 + q3 + q1, layers).save(os.path.join(outpath, 'oq4.png'))  

    run(diffuser, prompt_text_embeddings, blank_text_embeddings, range(26,100), layers).save(os.path.join(outpath, 'oearly.png'))    
   
 
    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt')
    parser.add_argument('outpath')

    main(**vars(parser.parse_args()))