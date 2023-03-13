import torch
from ...StableDiffuser import StableDiffuser
from .attention_hook import AttnHookModule

def edit_output(output, name):

    if 'up' in name:

        output[:] = 0.0



    return output



def main():

    nsteps = 50

    diffuser = StableDiffuser(seed=42).to(torch.device('cuda'))

    layers = []

    for name, module in diffuser.named_modules():

        if 'hshook'in name and 'attn1' not in name:

            layers.append(name)

    images = diffuser(['yellow car'],
                        reseed=True,
                        guidance_scale=7.5,
                        n_steps=nsteps
                        )
    images[0][0].save(f'new_control.jpg')

    for layer in layers:
        AttnHookModule.TS = layer

        images, trace_steps = diffuser(['yellow car'],
                        reseed=True,
                        trace_args={'layers': layers, 
                                    'edit_output' : edit_output,
                                    'retain_output':False},
                        guidance_scale=7.5,
                        n_steps=nsteps
                        )
    

        images[0][0].save(f'up.jpg')


if __name__ == '__main__':

    main()