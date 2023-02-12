import torch
from ....StableDiffuser import StableDiffuser
from deeplearning.models import Model
from ..finetuning import FineTunedModel

class DMSTNegationModel(Model, FineTunedModel):

    def __init__(self,
                 nsteps,
                 prompt_batch_size,
                 **kwargs):

        Model.__init__(self)

        diffuser = StableDiffuser()
        diffuser.set_scheduler_timesteps(nsteps)

        FineTunedModel.__init__(self, diffuser, **kwargs)

        self.diffuser = self.model

        self.nsteps = nsteps
        self.prompt_batch_size = prompt_batch_size


    def forward(self, word):

        word = word[0]

        with torch.no_grad():
    
            iteration = torch.randint(
                    1, self.nsteps - 1, (1,)).item()

            positive_text_embeddings = self.diffuser.get_text_embeddings(
                [word], n_imgs=self.prompt_batch_size).detach()
            neutral_text_embeddings = self.diffuser.get_text_embeddings(
                [''], n_imgs=self.prompt_batch_size).detach()

            latents = self.diffuser.get_initial_latents(self.prompt_batch_size, 512, 1).detach()

            with self:

                latents = self.diffuser.diffusion(
                    latents, positive_text_embeddings, end_iteration=iteration, show_progress=False)[0][0]

            positive_noise = self.diffuser.predict_noise(
                iteration, latents, positive_text_embeddings)
            neutral_noise = self.diffuser.predict_noise(
                iteration, latents, neutral_text_embeddings)

        with self:

            positive_text_embeddings = self.diffuser.get_text_embeddings(
                [word], n_imgs=self.prompt_batch_size).detach()

            new_noise = self.diffuser.predict_noise(
                iteration, latents, positive_text_embeddings)

        neutral_noise.requires_grad = False
        positive_noise.requires_grad = False

        return new_noise, neutral_noise, positive_noise
    
    def args(parser):

        parser.add_argument('--modules', type=str, nargs='+', required=True)
        parser.add_argument('--frozen_modules', type=str, nargs='+', default=[])
        parser.add_argument('--nsteps', type=int, default=50)
        parser.add_argument('--prompt_batch_size', type=int, default=1)

        super(DMSTNegationModel, DMSTNegationModel).args(parser)