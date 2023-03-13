import argparse

import torch
from baukit import TraceDict
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from . import util
from .LMSDiscreteScheduler import LMSDiscreteScheduler


def default_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('prompts', type=str, nargs='+')
    parser.add_argument('outpath', type=str)

    parser.add_argument('--images', type=str, nargs='+', default=None)
    parser.add_argument('--nsteps', type=int, default=1000)
    parser.add_argument('--nimgs', type=int, default=1)
    parser.add_argument('--start_itr', type=int, default=0)
    parser.add_argument('--return_steps', action='store_true', default=False)
    parser.add_argument('--pred_x0', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    return parser


class StableDiffuser(torch.nn.Module):

    def __init__(self,
                seed=None
        ):

        super().__init__()

        self._seed = seed

        # Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae")
        
        # Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14")
        
        # The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet")
        
        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        self.generator = torch.Generator()

        if self._seed is not None:

            self.seed(seed)

    def seed(self, seed):

        self.generator = torch.manual_seed(seed)

    def get_noise(self, batch_size, img_size):

        param = list(self.parameters())[0]

        return torch.randn(
            (batch_size, self.unet.in_channels, img_size // 8, img_size // 8),
            generator=self.generator).type(param.dtype).to(param.device)

    def add_noise(self, latents, noise, step):

        return self.scheduler.add_noise(latents, noise, torch.tensor([self.scheduler.timesteps[step]]))

    def text_tokenize(self, prompts):

        return self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

    def text_detokenize(self, tokens):

        return [self.tokenizer.decode(token) for token in tokens if token != self.tokenizer.vocab_size - 1]

    def text_encode(self, tokens):

        return self.text_encoder(tokens.input_ids.to(self.unet.device))[0]

    def decode(self, latents):

        return self.vae.decode(1 / 0.18215 * latents).sample

    def encode(self, tensors):

        return self.vae.encode(tensors).latent_dist.mode() * 0.18215

    def to_image(self, image):

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def set_scheduler_timesteps(self, n_steps):
        self.scheduler.set_timesteps(n_steps, device=self.unet.device)

    def get_initial_latents(self, n_imgs, img_size, n_prompts):

        noise = self.get_noise(n_imgs, img_size).repeat(n_prompts, 1, 1, 1)

        latents = noise * self.scheduler.init_noise_sigma

        return latents

    def get_text_embeddings(self, prompts, n_imgs):

        text_tokens = self.text_tokenize(prompts)

        text_embeddings = self.text_encode(text_tokens)

        unconditional_tokens = self.text_tokenize([""] * len(prompts))

        unconditional_embeddings = self.text_encode(unconditional_tokens)

        text_embeddings = torch.cat([unconditional_embeddings, text_embeddings]).repeat_interleave(n_imgs, dim=0)

        return text_embeddings

    def predict_noise(self,
             iteration,
             latents,
             text_embeddings,
             guidance_scale=7.5
             ):

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latents = torch.cat([latents] * 2)
        latents = self.scheduler.scale_model_input(
            latents, iteration)

        # predict the noise residual
        noise_prediction = self.unet(
            latents, self.scheduler.timesteps[iteration], encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_prediction_uncond, noise_prediction_text = noise_prediction.chunk(2)
        noise_prediction = noise_prediction_uncond + guidance_scale * \
            (noise_prediction_text - noise_prediction_uncond)

        return noise_prediction

    @torch.no_grad()
    def diffusion(self,
                  latents,
                  text_embeddings,
                  end_iteration=1000,
                  start_iteration=0,
                  return_steps=False,
                  pred_x0=False,
                  trace_args=None,                  
                  show_progress=True,
                  **kwargs):

        latents_steps = []
        trace_steps = []

        trace = None

        for iteration in tqdm(range(start_iteration, end_iteration), disable=not show_progress):

            if trace_args:

                trace = TraceDict(self, **trace_args)

            noise_pred = self.predict_noise(
                iteration, 
                latents, 
                text_embeddings,
                **kwargs)

            # compute the previous noisy sample x_t -> x_t-1
            output = self.scheduler.step(noise_pred, iteration, latents)

            if trace_args:

                trace.close()

                trace_steps.append(trace)

            latents = output.prev_sample

            if return_steps or iteration == end_iteration - 1:

                output = output.pred_original_sample if pred_x0 else latents

                if return_steps:
                    latents_steps.append(output.cpu())
                else:
                    latents_steps.append(output)

        return latents_steps, trace_steps

    @torch.no_grad()
    def __call__(self,
                 prompts,
                 img_size=512,
                 n_steps=50,
                 n_imgs=1,
                 end_iteration=None,
                 reseed=False,
                 **kwargs
                 ):

        assert 0 <= n_steps <= 1000

        if not isinstance(prompts, list):

            prompts = [prompts]

        self.set_scheduler_timesteps(n_steps)

        if reseed:

            self.seed(self._seed)

        latents = self.get_initial_latents(n_imgs, img_size, len(prompts))

        text_embeddings = self.get_text_embeddings(prompts,n_imgs=n_imgs)

        end_iteration = end_iteration or n_steps

        latents_steps, trace_steps = self.diffusion(
            latents,
            text_embeddings,
            end_iteration=end_iteration,
            **kwargs
        )

        latents_steps = [self.decode(latents.to(self.unet.device)) for latents in latents_steps]
        images_steps = [self.to_image(latents) for latents in latents_steps]

        images_steps = list(zip(*images_steps))

        if trace_steps:

            return images_steps, trace_steps

        return images_steps


if __name__ == '__main__':

    parser = default_parser()

    args = parser.parse_args()

    diffuser = StableDiffuser(seed=args.seed).to(torch.device(args.device)).half()

    images = diffuser(args.prompts,
                      n_steps=args.nsteps,
                      n_imgs=args.nimgs,
                      start_iteration=args.start_itr,
                      return_steps=args.return_steps,
                      pred_x0=args.pred_x0
                      )

    util.image_grid(images, args.outpath)