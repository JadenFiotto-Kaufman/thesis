import torch
from matplotlib import pyplot as plt
import numpy as np
from ... import utils
from ...StableDiffuser import StableDiffuser, default_parser
from . import AttentionHookModule, group_by_type, stack_attentions

if __name__ == '__main__':

    parser = default_parser()

    args = parser.parse_args()

    diffuser = StableDiffuser(torch.device(args.device), seed=args.seed)

    params = set([module_name for module_name, module in diffuser.unet.named_modules() if isinstance(module, AttentionHookModule)])

    images, trace_steps = diffuser(args.prompts,
        n_steps=args.nsteps, 
        n_imgs=args.nimgs, 
        start_iteration=args.start_itr,
        return_steps=args.return_steps,
        pred_x0=args.pred_x0, 
        trace_layers=params
    )

    tokens = diffuser.text_tokenize(args.prompts)['input_ids'][0][1:]
    tokens = diffuser.text_detokenize(tokens)

    attentions = stack_attentions(trace_steps)

    self_attentions, cross_attentions = group_by_type(attentions, len(tokens))

    shape = np.array(self_attentions[list(self_attentions.keys())[0]].shape)

    self_attentions_sum = torch.zeros(tuple(shape[[0, 3, 4]]))

    for key in self_attentions.keys():

        self_attentions_sum += self_attentions[key].sum(dim=(1,2))

    self_attention_images = []

    for i in range(len(self_attentions_sum)):

        plt.figure(figsize=(5,5), dpi=200)
        plt.imshow(self_attentions_sum[i], cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.tight_layout(pad=0)

        self_attention_images.append(utils.figure_to_image(plt.gcf()))

        plt.close()

    utils.image_grid([self_attention_images], args.outpath)

