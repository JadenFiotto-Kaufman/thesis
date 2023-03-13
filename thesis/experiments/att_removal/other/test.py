import torch
from ...StableDiffuser import StableDiffuser
from .attention_hook import AttnHookModule

ATTN_CACHE = {}
ATTN_TS = {}

def reset():

    for key in ATTN_CACHE:

        ATTN_TS[key] = 0


def edit_output(output, name):

    if AttnHookModule.SAVE:
        
        if name not in ATTN_CACHE:

            ATTN_CACHE[name] = [output]

        else:

            ATTN_CACHE[name].append(output.cpu())

    if AttnHookModule.LOAD and ATTN_TS[name] < AttnHookModule.TS:
   
        output = ATTN_CACHE[name][ATTN_TS[name]].cuda()

        ATTN_TS[name] += 1

    return output



def main():

    nsteps = 50

    diffuser = StableDiffuser(seed=50).to(torch.device('cuda'))

    layers = []

    for name, module in diffuser.named_modules():

        if 'hshook'in name and 'attn1' not in name:

            layers.append(name)

    AttnHookModule.SAVE=True

    images, trace_steps = diffuser(['illustration of a woman'],
                    reseed=True,
                    trace_args={'layers': layers, 
                                'edit_output' : edit_output,
                                'retain_output':False},
                    guidance_scale=7.5,
                    n_steps=nsteps
                    )
                    
    
    images[0][0].save('control.jpg')
    
    AttnHookModule.SAVE = False
    AttnHookModule.LOAD = True

    for ts in [1, 2, 3, 4, 5, 8, 10, 15, 20]:

        AttnHookModule.TS = ts
        AttnHookModule.LOAD = True

        reset()

        images, trace_steps = diffuser(['illustration of a woman in the style of Kelly McKernan'],
                        reseed=True,
                        trace_args={'layers': layers, 
                                    'edit_output' : edit_output,
                                    'retain_output':False},
                        guidance_scale=7.5,
                        n_steps=nsteps
                        )
        

    
        images[0][0].save(f'{ts}_new.jpg')

        del images 
        del trace_steps


if __name__ == '__main__':

    main()