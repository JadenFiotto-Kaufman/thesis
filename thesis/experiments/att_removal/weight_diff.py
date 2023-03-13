
import torch
from ..fine_tuning.finetuning import FineTunedModel
from ...StableDiffuser import StableDiffuser
import os
import numpy as np
import matplotlib.pyplot as plt

def main(ckpt_path, outpath, modules):

    os.makedirs(outpath, exist_ok=True)

    ckpt = torch.load(ckpt_path)

    diffuser = StableDiffuser().to(torch.device('cuda'))
    diffuser.set_scheduler_timesteps(50)
    diffuser.eval()

    finetuner = FineTunedModel(diffuser, modules)
    finetuner.load_state_dict(ckpt)

    keys = []
    eucs = []
    coss = []

    for key in finetuner.ft_modules:

        ft_module = finetuner.ft_modules[key]
        orig_module = finetuner.orig_modules[key]

        for (ft_name, ft_param), (orig_name, orig_param) in zip(ft_module.named_parameters(), orig_module.named_parameters()):

            if 'weight' in ft_name:

                name = f"{key}.{ft_name}"

                if 'to_out' not in ft_name:

                    for head in range(8):

                        _name =  f"{name}.{head}"

                        inner_dim = ft_param.shape[0]
                        head_dim = inner_dim // 8

                        start = head_dim * head
                        end = head_dim * (head + 1)

                        euc = torch.linalg.norm(ft_param[start:end].flatten() - orig_param[start:end].flatten()).item()
                        cos = (1 - torch.nn.functional.cosine_similarity(ft_param[start:end].flatten(), orig_param[start:end].flatten(), dim=-1)).item()

                        keys.append(_name)
                        eucs.append(euc)
                        coss.append(cos)

    euclidean_values = np.array(eucs)
    cosine_values = np.array(coss)

    sort_idx = euclidean_values.argsort()

    _keys = [keys[idx] for idx in sort_idx]

    plt.figure(figsize=(60,60))
    plt.barh(_keys, euclidean_values[sort_idx])
    plt.ylim
    plt.tight_layout()
    plt.savefig('euc.png')
    plt.clf()

    sort_idx = cosine_values.argsort()

    _keys = [keys[idx] for idx in sort_idx]

    plt.figure(figsize=(60,60))
    plt.barh(_keys, cosine_values[sort_idx])
    plt.ylim
    plt.tight_layout()
    plt.savefig('cos.png')
    plt.clf()

                       




if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('ckpt_path')
    parser.add_argument('outpath')
    parser.add_argument('--modules', nargs='+')

    main(**vars(parser.parse_args()))