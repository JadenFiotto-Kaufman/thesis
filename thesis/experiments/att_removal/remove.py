
import os

import torch
from tqdm import tqdm

from ...StableDiffuser import StableDiffuser
from ..fine_tuning.finetuning import FineTunedModel
 

def main(prompt, outpath, modules, batch_size, start_guidance, negative_guidance, iterations, lr, device, nsteps=50):
   
    diffuser = StableDiffuser(scheduler='DDIM').to(device)
    diffuser.train()

    finetuner = FineTunedModel(diffuser, modules)

    optimizer = torch.optim.Adam(finetuner.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()

    pbar = tqdm(range(iterations))

    with torch.no_grad():

        neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=batch_size)
        positive_text_embeddings = diffuser.get_text_embeddings([prompt],n_imgs=batch_size)

    losses = []

    for i in pbar:
        
        with torch.no_grad():

            diffuser.set_scheduler_timesteps(nsteps)

            optimizer.zero_grad()

            iteration = torch.randint(1, nsteps - 1, (1,)).item()

            latents = diffuser.get_initial_latents(batch_size, 512, 1)

            with finetuner:

                latents_steps, _ = diffuser.diffusion(
                    latents,
                    positive_text_embeddings,
                    start_iteration=0,
                    end_iteration=iteration,
                    guidance_scale=start_guidance, 
                    show_progress=False
                )

            diffuser.set_scheduler_timesteps(1000)

            iteration = int(iteration / nsteps * 1000)
            
            positive_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=1)
            neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)

        with finetuner:
            negative_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=1)

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False

        loss = criteria(negative_latents, neutral_latents - (negative_guidance*(positive_latents - neutral_latents))) #loss = criteria(e_n, e_0) works the best try 5000 epochs
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

    os.makedirs(outpath, exist_ok=True) 

    diffuser = diffuser.eval()
    finetuner = finetuner.eval()

    seed = 99

    torch.save(finetuner.state_dict(), os.path.join(outpath, 'checkpoint.pth'))
    generator = torch.manual_seed(seed)
    diffuser(prompt, generator=generator)[0][0].save(os.path.join(outpath,'prmtO.png'))
    generator = torch.manual_seed(seed)
    with finetuner: diffuser(prompt, generator=generator)[0][0].save(os.path.join(outpath,'promptN.png'))
    prompt = "Thomas Kinkade inspired depiction of a peaceful park"
    generator = torch.manual_seed(seed)
    diffuser(prompt, generator=generator)[0][0].save(os.path.join(outpath,'tkO.png'))
    generator = torch.manual_seed(seed)
    with finetuner: diffuser(prompt, generator=generator)[0][0].save(os.path.join(outpath,'tkN.png'))
    prompt = "Car next to tree"
    generator = torch.manual_seed(seed)
    diffuser(prompt, generator=generator)[0][0].save(os.path.join(outpath,'carO.png'))
    generator = torch.manual_seed(seed)
    with finetuner: diffuser(prompt, generator=generator)[0][0].save(os.path.join(outpath,'carN.png'))
    prompt= "A landscape of a wheat field under a stormy sky, with the thick brushstrokes and bold colors characteristic of Van Gogh's style."
    generator = torch.manual_seed(seed)
    diffuser(prompt, generator=generator)[0][0].save(os.path.join(outpath,'vgO.png'))
    generator = torch.manual_seed(seed)
    with finetuner: diffuser(prompt, generator=generator)[0][0].save(os.path.join(outpath,'vgN.png'))
    prompt= "Starry Night."
    generator = torch.manual_seed(seed)
    diffuser(prompt, generator=generator)[0][0].save(os.path.join(outpath,'snO.png'))
    generator = torch.manual_seed(seed)
    with finetuner: diffuser(prompt, generator=generator)[0][0].save(os.path.join(outpath,'snN.png'))
       

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--modules', type=str, required=True, nargs='+')
    parser.add_argument('--outpath', type=str, default='./')
    parser.add_argument('--start_guidance', type=float, default=3)
    parser.add_argument('--negative_guidance',type=float, default=1)
    parser.add_argument('--iterations',  type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--nsteps', type=int, default=50)
    parser.add_argument('--device',  default='cuda')
    parser.add_argument('--batch_size', type=int, default=2)
    
    main(**vars(parser.parse_args()))
    
   