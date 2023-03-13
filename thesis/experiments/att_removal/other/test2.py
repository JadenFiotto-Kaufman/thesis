import torch
from ...StableDiffuser import StableDiffuser
from .attention_hook import AttnHookModule
import matplotlib.pyplot as plt
ATTN_CACHE = {}
ATTN_TS = {}

def reset():

    for key in ATTN_CACHE:

        ATTN_TS[key] = 0


def edit_output(output, name):

    if 'hshook' in name:

        if AttnHookModule.SAVE:
            
            if name not in ATTN_CACHE:

                ATTN_CACHE[name] = [output]

            else:

                ATTN_CACHE[name].append(output.cpu())

        if AttnHookModule.LOAD and ATTN_TS[name] < AttnHookModule.TS:
    
            output = ATTN_CACHE[name][ATTN_TS[name]].cuda()

            ATTN_TS[name] += 1

    return output

def compare(pre, post, key):

    shape = pre.shape[-1]

    pre = pre.flatten()
    post = post.flatten()

    value = (1 - torch.nn.functional.cosine_similarity(pre ,post, dim=-1)).mean()

    key = key.replace('attentions', '')
    key = key.replace('unet', '')
    key = key.replace('transformer_blocks', '')
    key = key.replace('processor', '')
    key = key.replace('attn2', '')
    key = key.replace('...', '.')
    key = key.replace('..', '.')

    key += f'.{shape}'

    return value, key


def evaluate(pre, post, ts):

    pre = pre[ts]
    post = post[ts]

    keys = []
    values = []

    for key in pre:

        _pre = pre[key].output
        _post = post[key].output

        if 'hshook' in key:

            for i in range(_pre.size(0)):

                if i <= 7:

                    continue

                value, _key = compare(_pre[i], _post[i], key + f'.{i}')

                keys.append(_key)
                values.append(value)

        else:
            
            value, key = compare(_pre[1], _post[1], key)

            values.append(value)
            keys.append(key)
            

    values = torch.stack(values)

    sort_idx = values.argsort()

    values = values[sort_idx].cpu()
    keys = [keys[idx] for idx in sort_idx]

    plt.figure(figsize=(20,20))

    plt.barh(keys, values)

    plt.tight_layout()

    plt.savefig(f'change_cos_{ts}.jpg')
    



def main():

    nsteps = 50

    diffuser = StableDiffuser(seed=50).to(torch.device('cuda'))

    breakpoint()

    layers = []

    for name, module in diffuser.named_modules():

        if 'lhook' in name and 'attn1' not in name:

            layers.append(name)

        if 'hshook'in name and 'attn1' not in name:

            layers.append(name)

    AttnHookModule.SAVE=True

    images, pre_trace_steps = diffuser(['illustration in the style of Kelly McKernan'],
                    reseed=True,
                    trace_args={'layers': layers, 
                                'edit_output' : edit_output,
                                'retain_output':True},
                    guidance_scale=7.5,
                    n_steps=nsteps
                    )
                    
    
    images[0][0].save('control.jpg')
    
    AttnHookModule.SAVE = False
    AttnHookModule.LOAD = True

    for ts in [5, 10, 15, 20, 25, 30, 35]:

        AttnHookModule.TS = ts
        AttnHookModule.LOAD = True

        reset()

        images, post_trace_steps = diffuser(['illustration'],
                        reseed=True,
                        trace_args={'layers': layers, 
                                    'edit_output' : edit_output,
                                    'retain_output':True},
                        guidance_scale=7.5,
                        n_steps=nsteps
                        )
        

    
        images[0][0].save(f'{ts}_new.jpg')

        evaluate(pre_trace_steps, post_trace_steps, ts)




if __name__ == '__main__':

    main()