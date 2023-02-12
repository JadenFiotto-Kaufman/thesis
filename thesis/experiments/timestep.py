from .. import utils
from ..StableDiffuser import StableDiffuser as _StableDiffuser, default_parser
import torch


class StableDiffuser(_StableDiffuser):
    
    def __init__(self, timestep_itr, sigma_itr, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.timestep_itr = timestep_itr
        self.sigma_itr = sigma_itr


    def set_scheduler_timesteps(self, n_steps):
        super().set_scheduler_timesteps(n_steps)

        if self.timestep_itr is not None:

            self.scheduler.timesteps[:] = self.scheduler.timesteps[args.timestep_itr]


        if self.sigma_itr is not None:

            self.scheduler.sigmas[:] = self.scheduler.sigmas[args.sigma_itr]



if __name__ == '__main__':

    parser = default_parser()

    parser.add_argument('--timestep_itr', type=int, default=None)
    parser.add_argument('--sigma_itr', type=int, default=None)

    args = parser.parse_args()

    diffuser = StableDiffuser(args.timestep_itr, args.sigma_itr, torch.device(args.device), seed=args.seed)


    images = diffuser(args.prompts,
                      n_steps=args.nsteps,
                      n_imgs=args.nimgs,
                      start_iteration=args.start_itr,
                      return_steps=args.return_steps,
                      pred_x0=args.pred_x0
                      )

    utils.image_grid(images, args.outpath)
