
import torch
from ..fine_tuning.finetuning import FineTunedModel
from ...StableDiffuser import StableDiffuser
import os
def main(ckpt_path, outpath, modules):

    os.makedirs(outpath, exist_ok=True)

    ckpt = torch.load(ckpt_path)

    diffuser = StableDiffuser().to(torch.device('cuda'))
    diffuser.set_scheduler_timesteps(50)
    diffuser.eval()

    finetuner = FineTunedModel(diffuser, modules)
    finetuner.load_state_dict(ckpt)


    while True:

        prompt = input("Enter prompt: ")

        diffuser._seed = 40
        diffuser(prompt, reseed=True)[0][0].save('orig.png')
        with finetuner: diffuser(prompt, reseed=True)[0][0].save('new.png')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('ckpt_path')
    parser.add_argument('outpath')
    parser.add_argument('--modules', nargs='+')

    main(**vars(parser.parse_args()))