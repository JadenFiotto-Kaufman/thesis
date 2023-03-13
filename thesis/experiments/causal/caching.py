
class ActivationCache:

    CACHE = {}
    TIMESTEPS = {}

    @staticmethod
    def reset():

        for key in ActivationCache.TIMESTEPS:
 
            ActivationCache.TIMESTEPS[key] = 0
    
    @staticmethod
    def cache(activations, name):

        if name not in ActivationCache.CACHE:

            ActivationCache.CACHE[name] = [activations.cpu()]     
            ActivationCache.TIMESTEPS[name] = 0
        
        else:

            ActivationCache.CACHE[name].append(activations.cpu())

        return activations

    @staticmethod
    def load(activations, name):

        activations = ActivationCache.CACHE[name][ActivationCache.TIMESTEPS[name]].cuda()
        ActivationCache.TIMESTEPS[name] += 1

        return activations
       

    