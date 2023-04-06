
import os

import numpy as np
import pandas as pd
import torch

from ...StableDiffuser import StableDiffuser


def edit_output(activation, name):

    activation[:] = 0.0

    return activation


def main(inpath, outpath, device):

    diffuser = StableDiffuser(scheduler='LMS').to(torch.device(device)).half()

    layers = set([module_name for module_name, module in diffuser.named_modules() if module_name.endswith('attn2')])

    generator = torch.manual_seed(42)

    os.makedirs(outpath, exist_ok=True)

    prompt = "Van Gogh"
    nsteps = 50

    images = diffuser(
        prompt,
        generator=generator,
        n_steps=nsteps, 
    )

    images[0][0].save(os.path.join(outpath, f"orig.png"))

    for layer in layers:

        generator = torch.manual_seed(42)

        images, trace_steps = diffuser(
            prompt,
            generator=generator,
            n_steps=nsteps, 
            trace_args={'layers' : [layer], 'edit_output': edit_output}
        )

        images[0][0].save(os.path.join(outpath, f"{layer}.png"))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('inpath')
    parser.add_argument('outpath')
    parser.add_argument('--device', default='cuda:0')

    main(**vars(parser.parse_args()))

