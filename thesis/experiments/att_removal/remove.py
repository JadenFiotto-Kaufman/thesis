
from ...StableDiffuser import StableDiffuser
import torch
import random
from ..fine_tuning.finetuning import FineTunedModel
from tqdm import tqdm


def main(prompt, modules, batch_size, start_guidance, negative_guidance, iterations, lr, nsteps=50):
   
    diffuser = StableDiffuser().to(torch.device('cuda'))
    diffuser.set_scheduler_timesteps(nsteps)
    diffuser.train()

    finetuner = FineTunedModel(diffuser, modules)

    optimizer = torch.optim.Adam(finetuner.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()

    pbar = tqdm(range(iterations))

    for i in pbar:
        
        with torch.no_grad():

            positive_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=batch_size)
            neutral_text_embeddings = diffuser.get_text_embeddings([prompt],n_imgs=batch_size)

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
            
            positive_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=start_guidance)
            neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=start_guidance)

        with finetuner:
            negative_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=start_guidance)

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False

        loss = criteria(negative_latents, neutral_latents - (negative_guidance*neutral_latents - neutral_latents)) #loss = criteria(e_n, e_0) works the best try 5000 epochs
        loss.backward()

        optimizer.step()

    torch.save(finetuner.state_dict(), 'checkpoint.pth')
    diffuser._seed = 40
    diffuser('Art in the style of Pablo Picasso', reseed=True)[0][0].save('orig.png')
    with finetuner: diffuser('Art in the style of Pablo Picasso', reseed=True)[0][0].save('new.png')

    breakpoint()
       

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--modules', type=str, required=True)
    parser.add_argument('--start_guidance', type=float, default=3)
    parser.add_argument('--negative_guidance',type=float, default=1)
    parser.add_argument('--iterations',  type=int, default=500)
    parser.add_argument('--lr', type=int, default=1e-5)
    parser.add_argument('--nsteps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    
    main(**vars(parser.parse_args()))
    
   