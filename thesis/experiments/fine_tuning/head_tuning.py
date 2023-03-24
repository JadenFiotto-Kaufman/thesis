from .finetuning import FineTunedModel
from functools import partial
import torch
class HeadTuner(FineTunedModel):


    def __init__(self, model, modules, frozen_modules=[]):

        _modules = []
        module_head_map = {}

        for module, heads in modules:

            _modules.append(f"{module}.to_q")
            _modules.append(f"{module}.to_k")
            _modules.append(f"{module}.to_v")
            _modules.append(f"{module}.to_out.0")

            #module_head_map[f"{module}.to_q"] = heads
            module_head_map[f"{module}.to_k"] = heads
            module_head_map[f"{module}.to_v"] = heads
            module_head_map[f"{module}.to_out.0"] = heads

        super().__init__(model, _modules, frozen_modules)

        for module_name, module in self.ft_modules.items():

            if module_name in module_head_map:

                module.weight.register_hook(partial(HeadTuner.backward_head, heads=module_head_map[module_name]))

    @staticmethod
    def backward_head(grad, heads):

        if len(heads) != 0:

            head_dim = 40

            mask = torch.full((grad.shape[0],), 1, dtype=torch.bool)

            for head in heads:

                start = head * head_dim
                end = start + head_dim

                mask[start:end] = False
                
            grad[mask] = 0.

        return grad