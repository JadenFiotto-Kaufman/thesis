
import os

import torch
from tqdm import tqdm

from ...StableDiffuser import StableDiffuser
from ..fine_tuning.head_tuning import HeadTuner
from ... import util

def evaluate(diffuser, finetuner, iteration, outpath, seed, prompt):

    diffuser = diffuser.eval()
    finetuner = finetuner.eval()

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


    diffuser = diffuser.train()
    finetuner = finetuner.train()




def main(prompt, outpath, batch_size, start_guidance, negative_guidance, iterations, lr, device, nsteps=50):
   
    diffuser = StableDiffuser(scheduler='DDIM').to(device)
    diffuser.train()

    # #Down
    # modules = [
    #     ['unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2', []],
    #     ['unet.down_blocks.0.attentions.1.transformer_blocks.0.attn2', []],
    #     ['unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2', []],
    #     ['unet.down_blocks.1.attentions.1.transformer_blocks.0.attn2', []],
    #     ['unet.down_blocks.2.attentions.0.transformer_blocks.0.attn2', []],
    #     ['unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2', []],
    # ]
    #All
    modules = [
        ['unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2', []],
        ['unet.down_blocks.0.attentions.1.transformer_blocks.0.attn2', []],
        ['unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2', []],
        ['unet.down_blocks.1.attentions.1.transformer_blocks.0.attn2', []],
        ['unet.down_blocks.2.attentions.0.transformer_blocks.0.attn2', []],
        ['unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2', []],
        ['unet.mid_block.attentions.0.transformer_blocks.0.attn2', []],
        ['unet.up_blocks.1.attentions.0.transformer_blocks.0.attn2', []],
        ['unet.up_blocks.1.attentions.1.transformer_blocks.0.attn2', []],
        ['unet.up_blocks.1.attentions.2.transformer_blocks.0.attn2', []],
        ['unet.up_blocks.2.attentions.0.transformer_blocks.0.attn2', []],
        ['unet.up_blocks.2.attentions.1.transformer_blocks.0.attn2', []],
        ['unet.up_blocks.2.attentions.2.transformer_blocks.0.attn2', []],
        ['unet.up_blocks.3.attentions.0.transformer_blocks.0.attn2', []],
        ['unet.up_blocks.3.attentions.1.transformer_blocks.0.attn2', []],
        ['unet.up_blocks.3.attentions.2.transformer_blocks.0.attn2', []],
    ]



    # All20
    # modules = [
    #     ['unet.down_blocks.1.attentions.1.transformer_blocks.0.attn2', [8]],
    #     ['unet.down_blocks.2.attentions.0.transformer_blocks.0.attn2', [10, 11]],
    #     ['unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2', [8, 14]],
    #     ['unet.mid_block.attentions.0.transformer_blocks.0.attn2', [8]],
    #     ['unet.up_blocks.1.attentions.0.transformer_blocks.0.attn2', [11, 12, 13, 14]],
    #     ['unet.up_blocks.1.attentions.1.transformer_blocks.0.attn2', [8, 14]],
    #     ['unet.up_blocks.1.attentions.2.transformer_blocks.0.attn2', [8, 9, 11, 14]],
    #     ['unet.up_blocks.2.attentions.2.transformer_blocks.0.attn2', [9,10,13]],
    #     ['unet.up_blocks.3.attentions.0.transformer_blocks.0.attn2', [11]],
    # ]

    # Down10
    # modules = [
    #     ['unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2', [15]],
    #     ['unet.down_blocks.1.attentions.1.transformer_blocks.0.attn2', [8]],
    #     ['unet.down_blocks.2.attentions.0.transformer_blocks.0.attn2', [10, 11]],
    #     ['unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2', [8, 12, 14]],
    # ]

    # Down20
    # modules = [
    #     ['unet.down_blocks.1.attentions.1.transformer_blocks.0.attn2', [8]],
    #     ['unet.down_blocks.2.attentions.0.transformer_blocks.0.attn2', [10, 11]],
    #     ['unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2', [8,14]],
    # ]

    # Down5 + mid
    # modules = [
    #     ['unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2', [9]],
    #     ['unet.down_blocks.0.attentions.1.transformer_blocks.0.attn2', [8, 11 ,12, 13, 15]],
    #     ['unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2', [15, 10, 13]],
    #     ['unet.down_blocks.1.attentions.1.transformer_blocks.0.attn2', [8]],
    #     ['unet.down_blocks.2.attentions.0.transformer_blocks.0.attn2', [10, 11, 9, 13]],
    #     ['unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2', [8, 11, 12, 14, 15]],
    #     #['unet.mid_block.attentions.0.transformer_blocks.0.attn2', [8, 9,10,11,12,13,14,15]],
    # ]

    # All5
    # modules = [
    #     ['unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2', [9]],
    #     ['unet.down_blocks.0.attentions.1.transformer_blocks.0.attn2', [8, 11 ,12, 13, 15]],
    #     ['unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2', [15, 10, 13]],
    #     ['unet.down_blocks.1.attentions.1.transformer_blocks.0.attn2', [8]],
    #     ['unet.down_blocks.2.attentions.0.transformer_blocks.0.attn2', [10, 11, 9, 13]],
    #     ['unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2', [8, 11, 12, 14, 15]],
    #     ['unet.mid_block.attentions.0.transformer_blocks.0.attn2', [8, 13, 15]],
    #     ['unet.up_blocks.1.attentions.0.transformer_blocks.0.attn2', [9, 10, 11, 12, 13, 14, 15]],
    #     ['unet.up_blocks.1.attentions.1.transformer_blocks.0.attn2', [8,9, 10,  11, 12, 13, 14]],
    #     ['unet.up_blocks.1.attentions.2.transformer_blocks.0.attn2', [8, 9, 11, 13, 14, 15]],
    #     ['unet.up_blocks.2.attentions.2.transformer_blocks.0.attn2', [8, 9,10,13]],
    #     ['unet.up_blocks.3.attentions.0.transformer_blocks.0.attn2', [11, 12, 15]],
    #     ['unet.up_blocks.3.attentions.2.transformer_blocks.0.attn2', [12, 13, 15]],
    # ]

    # Down5Random
    # modules = [
    #     ['unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2', [8, 12, 13]],
    #     ['unet.down_blocks.0.attentions.1.transformer_blocks.0.attn2', [9, 10 ,14]],
    #     ['unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2', [8, 9, 12]],
    #     ['unet.down_blocks.1.attentions.1.transformer_blocks.0.attn2', [9, 10, 11]],
    #     ['unet.down_blocks.2.attentions.0.transformer_blocks.0.attn2', [8, 12, 14, 15]],
    #     ['unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2', [9, 10, 13]],
    # ]

    finetuner = HeadTuner(diffuser, modules)

    optimizer = torch.optim.Adam(finetuner.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()

    pbar = tqdm(range(iterations))

    with torch.no_grad():

        neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=batch_size)
        positive_text_embeddings = diffuser.get_text_embeddings([prompt],n_imgs=batch_size)

    losses = []

    seed = 99

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

        if i % 25 == 0:

            _outpath = os.path.join(outpath, 'training', str(i))

            os.makedirs(_outpath, exist_ok=True) 

            evaluate(diffuser,finetuner, i, _outpath, seed, prompt)

    torch.save(finetuner.state_dict(), os.path.join(outpath, 'checkpoint.pth'))
   
       

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--outpath', type=str, default='./')
    parser.add_argument('--start_guidance', type=float, default=3)
    parser.add_argument('--negative_guidance',type=float, default=1)
    parser.add_argument('--iterations',  type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--nsteps', type=int, default=50)
    parser.add_argument('--device',  default='cuda')
    parser.add_argument('--batch_size', type=int, default=2)
    
    main(**vars(parser.parse_args()))
    
   