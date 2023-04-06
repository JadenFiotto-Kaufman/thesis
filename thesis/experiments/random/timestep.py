
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
def main(prompt, outpath):


    diffuser = StableDiffuser(scheduler='EA').to(torch.device('cuda:0')).half()
    diffuser.set_scheduler_timesteps(50)
    #diffuser.scheduler.timesteps[:] = diffuser.scheduler.timesteps[20]

    latents = diffuser.get_initial_latents(1, 512, 1)

    text_embeddings = diffuser.get_text_embeddings([prompt],n_imgs=1)

    pred_original_samples = []

    for iteration in tqdm(range(50)):

        noise_pred = diffuser.predict_noise(
            iteration, 
            latents, 
            text_embeddings)
        
        output = diffuser.scheduler.step(noise_pred, diffuser.scheduler.timesteps[iteration], latents)
        
        latents = output.prev_sample

        pred_original_samples.append(output.pred_original_sample.cpu())

    pred_images = [diffuser.to_image(diffuser.decode(data.to('cuda:0')).cpu().float()) for data in pred_original_samples]
    diffs = torch.concat(pred_original_samples).absolute().diff(dim=0).mean(dim=1).cpu().numpy()
   
    images = []

    for i in range(diffs.shape[0]):

        hm = heatmap(diffs[i])
        image = pred_images[i+1][0]
        hm = hm.resize(image.size)

        images.append(util.get_concat_h(hm, image))

    util.to_gif(images=images, path=outpath)        

    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt')
    parser.add_argument('outpath')


    main(**vars(parser.parse_args()))