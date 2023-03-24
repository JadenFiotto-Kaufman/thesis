
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

    # generator = torch.manual_seed(seed)
    # orig = diffuser(prompt, generator=generator)[0][0]
    # generator = torch.manual_seed(seed)
    # with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    # util.image_grid([[orig, new]], outpath=os.path.join(outpath,'prompt.png'))

    prompt = "Thomas Kinkade inspired depiction of a peaceful park"
    generator = torch.manual_seed(seed)
    orig = diffuser(prompt, generator=generator)[0][0]
    generator = torch.manual_seed(seed)
    with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    util.image_grid([[orig, new]], outpath=os.path.join(outpath,'tk.png'))


    

    # prompt = "Car next to tree"
    # generator = torch.manual_seed(seed)
    # orig = diffuser(prompt, generator=generator)[0][0]
    # generator = torch.manual_seed(seed)
    # with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    # util.image_grid([[orig, new]], outpath=os.path.join(outpath,'car.png'))
    
    # prompt= "A landscape of a wheat field under a stormy sky, with the thick brushstrokes and bold colors characteristic of Van Gogh's style."
    # generator = torch.manual_seed(seed)
    # orig = diffuser(prompt, generator=generator)[0][0]
    # generator = torch.manual_seed(seed)
    # with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    # util.image_grid([[orig, new]], outpath=os.path.join(outpath,'vg.png'))
    
    # prompt= "Starry Night."
    # generator = torch.manual_seed(seed)
    # orig = diffuser(prompt, generator=generator)[0][0]
    # generator = torch.manual_seed(seed)
    # with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    # util.image_grid([[orig, new]], outpath=os.path.join(outpath,'sn.png'))

    # prompt= "A view of a bridge over a river, with the play of light on the water and the reflections on the surface, similar to Monet's series of paintings of the Thames river."
    # generator = torch.manual_seed(seed)
    # orig = diffuser(prompt, generator=generator)[0][0]
    # generator = torch.manual_seed(seed)
    # with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    # util.image_grid([[orig, new]], outpath=os.path.join(outpath,'mon.png'))
    
    # prompt= "A vibrant, swirling depiction of a starry night sky over a peaceful village, inspired by Vincent van Gogh's 'The Starry Night.'"
    # generator = torch.manual_seed(4862)
    # orig = diffuser(prompt, generator=generator)[0][0]
    # generator = torch.manual_seed(4862)
    # with finetuner: new = diffuser(prompt, generator=generator)[0][0]
    # util.image_grid([[orig, new]], outpath=os.path.join(outpath,'sn2.png'))
    
    diffuser = diffuser.train().float()
    finetuner = finetuner.train().float()

def main(prompt, outpath, batch_size, start_guidance, reg_scale, iterations, lr, device, nsteps=50):
   
    diffuser = StableDiffuser(scheduler='DDIM').to(device)
    diffuser.set_scheduler_timesteps(50)
    diffuser.train()

    modules = [".*attn2.*to_k$", ".*attn2.*to_v$"]

    finetuner = FineTunedModel(diffuser, modules)

    optimizer = torch.optim.Adam(finetuner.parameters(), lr=lr)

    pbar = tqdm(range(iterations))

    with torch.no_grad():

        neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=batch_size)
        positive_text_embeddings = diffuser.get_text_embeddings([prompt],n_imgs=batch_size)

        latents = diffuser.get_initial_latents(batch_size, 512, 1)

        with TraceDict(
                diffuser,
                layers=finetuner.ft_modules.keys(),
                retain_output=True
            ) as original_trace:
            
                _ = diffuser.predict_noise(0, latents, neutral_text_embeddings, guidance_scale=1)

    

    losses = []

    for i in pbar:
        

        optimizer.zero_grad()

        with finetuner:
    
            with TraceDict(
                diffuser,
                layers=finetuner.ft_modules.keys(),
                retain_output=True
            ) as erased_trace:
            
                _ = diffuser.predict_noise(0, latents, positive_text_embeddings, guidance_scale=1)
            
        losses = []

        for key in erased_trace:

            erased_value = erased_trace[key].output
            original_value = original_trace[key].output

            loss = torch.nn.functional.mse_loss(erased_value, original_value)
            losses.append(loss)

        value_loss = torch.stack(losses).sum()

        losses = []

        for key in finetuner.ft_modules:

            ft_weight = finetuner.ft_modules[key].weight
            orig_weight = finetuner.orig_modules[key].weight

            loss = torch.linalg.matrix_norm(ft_weight - orig_weight)

            losses.append(loss)

        regularization_loss = reg_scale * torch.stack(losses).sum()

        loss = value_loss + regularization_loss

        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        if i % 9999989 == 0 and i > 0:

            _outpath = os.path.join(outpath, 'training', str(i))

            os.makedirs(_outpath, exist_ok=True) 

            evaluate(diffuser,finetuner, _outpath, 4234, prompt)

    os.makedirs(outpath, exist_ok=True) 

    diffuser = diffuser.eval()
    finetuner = finetuner.eval()

    torch.save(finetuner.state_dict(), os.path.join(outpath, 'checkpoint.pth'))
 

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--outpath', type=str, default='./')
    parser.add_argument('--start_guidance', type=float, default=3)
    parser.add_argument('--reg_scale',type=float, default=0)
    parser.add_argument('--iterations',  type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--nsteps', type=int, default=50)
    parser.add_argument('--device',  default='cuda')
    parser.add_argument('--batch_size', type=int, default=2)
    
    main(**vars(parser.parse_args()))
    
   