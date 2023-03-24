

import torch
from tqdm import tqdm
import os
from ...StableDiffuser import StableDiffuser
from ... import util
from ..fine_tuning.finetuning import FineTunedModel
import pandas as pd

def main(inpath, outpath, model_path, device, nsteps=50):
   
    diffuser = StableDiffuser(scheduler='DDIM').to(device).half()
    diffuser.eval()

    state_dict = torch.load(model_path)

    finetuner = FineTunedModel(diffuser, [f"{key}$" for key in list(state_dict.keys())])
    finetuner.load_state_dict(state_dict)

    data = pd.read_csv(inpath)

    os.makedirs(os.path.join(outpath, 'ESD'), exist_ok=True)
    os.makedirs(os.path.join(outpath, 'SD'), exist_ok=True)


    for row in tqdm(data.iterrows()):
        row = row[1]
        prompt = row['prompt']
        seed = row['evaluation_seed']
        generator = torch.manual_seed(seed)

        original = diffuser(prompt, generator=generator)[0][0]
        generator = torch.manual_seed(seed)
        with finetuner: 
            new = diffuser(prompt, generator=generator)[0][0]

        
        original.save(os.path.join(outpath, 'SD', f"{row['case_number']}.png"))
        new.save(os.path.join(outpath, 'ESD', f"{row['case_number']}.png"))
        util.image_grid([[original, new]], column_titles=[prompt, ""], outpath=os.path.join(outpath, f"{row['case_number']}.png"))

        

       

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', type=str)
    parser.add_argument('outpath', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('--nsteps', type=int, default=50)
    parser.add_argument('--device',  default='cuda')
    
    main(**vars(parser.parse_args()))
    
   