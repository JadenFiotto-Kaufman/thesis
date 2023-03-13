
import torch
from ...StableDiffuser import StableDiffuser
from ... import util
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

def heatmap(data):

    plt.figure(figsize=(5,5), dpi=200)
    plt.imshow(data, cmap='inferno', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout(pad=0)

    image = util.figure_to_image(plt.gcf())

    plt.close()

    return image



@torch.no_grad()
def main():

    prompt = "An indian woman wearing a red tophat standing on a boulder in a jungle"

    diffuser = StableDiffuser(seed=42).to(torch.device('cuda:1')).half()
    diffuser.set_scheduler_timesteps(50)
    # diffuser.scheduler.timesteps[:] = diffuser.scheduler.timesteps[3]
    diffuser.seed(diffuser._seed)

    latents = diffuser.get_initial_latents(1, 512, 1)

    text_embeddings = diffuser.get_text_embeddings([prompt],n_imgs=1)

    latents_steps = []

    for iteration in tqdm(range(50)):

        noise_pred = diffuser.predict_noise(
            iteration, 
            latents, 
            text_embeddings)
        
        

        # compute the previous noisy sample x_t -> x_t-1
        output = diffuser.scheduler.step(noise_pred, iteration, latents)
        
        latents = output.prev_sample

        latents_steps.append(output.pred_original_sample.cpu())

    control = diffuser.to_image(diffuser.decode(latents))
    control[0].save('image.png')

        

        

    # latents_steps = [diffuser.decode(latents.to(diffuser.unet.device)) for latents in latents_steps]
    diffs = torch.concat(latents_steps).mean(dim=1).absolute().cpu().numpy()

    images = []

    for i in range(diffs.shape[0]):

        images.append(heatmap(diffs[i]))

    # images_steps = [diffuser.to_image(latents) for latents in latents_steps]

    # images_steps = list(zip(*images_steps))

    util.to_gif(images=images, path='test.gif')        

    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()


    main(**vars(parser.parse_args()))