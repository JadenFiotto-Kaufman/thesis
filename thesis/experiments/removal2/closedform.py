
import os

import torch
from tqdm import tqdm
from baukit import TraceDict
from ...StableDiffuser import StableDiffuser
from ..fine_tuning.finetuning import FineTunedModel
from ... import util

def evaluate(diffuser, finetuner, outpath, seed, prompt):

    diffuser = diffuser.eval().half()
    finetuner = finetuner.eval().half()

    generator = torch.manual_seed(seed)
    orig = diffuser(prompt, generator=generator)[0][0]
    generator = torch.manual_seed(seed)
    with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    util.image_grid([[orig, new]], outpath=os.path.join(outpath,'prompt.png'))

    prompt = "Thomas Kinkade inspired depiction of a peaceful park"
    generator = torch.manual_seed(seed)
    orig = diffuser(prompt, generator=generator)[0][0]
    generator = torch.manual_seed(seed)
    with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    util.image_grid([[orig, new]], outpath=os.path.join(outpath,'tk.png'))

    prompt = "Car next to tree"
    generator = torch.manual_seed(seed)
    orig = diffuser(prompt, generator=generator)[0][0]
    generator = torch.manual_seed(seed)
    with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    util.image_grid([[orig, new]], outpath=os.path.join(outpath,'car.png'))
    
    prompt= "A landscape of a wheat field under a stormy sky, with the thick brushstrokes and bold colors characteristic of Van Gogh's style."
    generator = torch.manual_seed(seed)
    orig = diffuser(prompt, generator=generator)[0][0]
    generator = torch.manual_seed(seed)
    with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    util.image_grid([[orig, new]], outpath=os.path.join(outpath,'vg.png'))
    
    prompt= "Starry Night."
    generator = torch.manual_seed(seed)
    orig = diffuser(prompt, generator=generator)[0][0]
    generator = torch.manual_seed(seed)
    with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    util.image_grid([[orig, new]], outpath=os.path.join(outpath,'sn.png'))

    prompt= "A view of a bridge over a river, with the play of light on the water and the reflections on the surface, similar to Monet's series of paintings of the Thames river."
    generator = torch.manual_seed(seed)
    orig = diffuser(prompt, generator=generator)[0][0]
    generator = torch.manual_seed(seed)
    with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    util.image_grid([[orig, new]], outpath=os.path.join(outpath,'mon.png'))
    
    prompt= "A vibrant, swirling depiction of a starry night sky over a peaceful village, inspired by Vincent van Gogh's 'The Starry Night.'"
    generator = torch.manual_seed(4862)
    orig = diffuser(prompt, generator=generator)[0][0]
    generator = torch.manual_seed(4862)
    with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    util.image_grid([[orig, new]], outpath=os.path.join(outpath,'sn2.png'))
    
    diffuser = diffuser.train().float()
    finetuner = finetuner.train().float()

def main(prompt, outpath,  reg_scale, device):
   
    diffuser = StableDiffuser(scheduler='DDIM').to(device)
    diffuser.set_scheduler_timesteps(50)

    modules = [".*attn2.*to_k$", ".*attn2.*to_v$"]

    finetuner = FineTunedModel(diffuser, modules)

    with torch.no_grad():

        text_embeddings = diffuser.get_text_embeddings([prompt],n_imgs=1)

        latents = diffuser.get_initial_latents(1, 512, 1)


        with TraceDict(
            diffuser,
            layers=finetuner.ft_modules.keys(),
            retain_output=True,
            retain_input=True
        ) as trace:
        
            _ = diffuser.predict_noise(0, latents, text_embeddings, guidance_scale=1)           

        for key in trace:

            neutral_output = trace[key].output[0].cpu()

            positive_input = trace[key].input[1].cpu()

            old_weights = finetuner.orig_modules[key].weight.cpu()

            new_weights = torch.matmul(
                reg_scale * old_weights + torch.matmul(neutral_output.T,positive_input) , 
                torch.inverse(reg_scale * torch.eye(positive_input.shape[1]) + torch.matmul(positive_input.T,positive_input)))

            finetuner.ft_modules[key].weight[:] = new_weights

    os.makedirs(outpath, exist_ok=True) 

    evaluate(diffuser, finetuner, outpath, 4324, prompt)

    torch.save(finetuner.state_dict(), os.path.join(outpath, 'checkpoint.pth'))
 

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--outpath', type=str, default='./')
    parser.add_argument('--reg_scale',type=float, default=0)
    parser.add_argument('--device',  default='cuda')
    
    main(**vars(parser.parse_args()))
    
   