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

    diffuser = StableDiffuser(seed=42).to(torch.device('cuda'))

    layers = []

    for name, module in diffuser.named_modules():

        if 'hshook'in name and 'attn1' not in name:

            layers.append(name)

    AttnHookModule.SAVE=True

    car_trace_steps = []

    for i in range(10):

        images, trace_steps = diffuser(['car'],
                        reseed=i==0,
                        trace_args={'layers': layers, 
                                    'edit_output' : edit_output,
                                    'retain_output':True},
                        guidance_scale=7.5,
                        n_steps=nsteps
                        )
        
        car_trace_steps.append(trace_steps)
    


    pre_steps = trace_steps
    
    AttnHookModule.SAVE = False
    AttnHookModule.LOAD = True

    for ts in [53]:

        AttnHookModule.TS = ts
        AttnHookModule.LOAD = True

        reset()

        images, trace_steps = diffuser(['road next to a tree'],
                        reseed=True,
                        trace_args={'layers': layers, 
                                    'edit_output' : edit_output,
                                    'retain_output':True},
                        guidance_scale=7.5,
                        end_iteration=55,
                        n_steps=nsteps
                        )
        

        post_steps = trace_steps

        diffs = []

        for key in pre_steps[53]:

            pre_output = pre_steps[53][key].output
            post_output = post_steps[53][key].output

            diff = torch.linalg.norm(pre_output - post_output, dim=-1).mean(dim=-1)

            diffs.append(diff)

        breakpoint()


if __name__ == '__main__':

    main()