import torch
from ...StableDiffuser import StableDiffuser
from .attention_hook import AttnHookModule


def main():

    nsteps = 50

    diffuser = StableDiffuser(seed=42).to(torch.device('cuda'))

    layers = []

    for name, module in diffuser.named_modules():

        if 'hshook'in name and 'attn1' not in name:

            layers.append(name)

    car_trace_steps = []

    for i in range(10):

        images, trace_steps = diffuser(['car'],
                        reseed=i==0,
                        trace_args={'layers': layers, 
                                    'retain_output':True},
                        guidance_scale=7.5,
                        n_steps=nsteps
                        )
        
        car_trace_steps.append(trace_steps)

    dts = {}

    for key in car_trace_steps[0][0]:

        dts[key] = torch.stack([step[key].output for trace_steps in car_trace_steps for step in trace_steps]).flatten(start_dim=1, end_dim=2).mean(dim=0).flatten()

    
    blank_trace_steps = []

    for i in range(10):

        images, trace_steps = diffuser([''],
                        reseed=i==0,
                        trace_args={'layers': layers, 
                                    'retain_output':True},
                        guidance_scale=7.5,
                        n_steps=nsteps
                        )
        
        blank_trace_steps.append(trace_steps)

    bdts = {}

    for key in blank_trace_steps[0][0]:

        bdts[key] = torch.stack([step[key].output for trace_steps in blank_trace_steps for step in trace_steps]).flatten(start_dim=1, end_dim=2).mean(dim=0).flatten()

    outs = []

    for key in dts:

        outs.append(torch.linalg.norm(dts[key] - bdts[key]))

    breakpoint()



if __name__ == '__main__':

    main()