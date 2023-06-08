import os

import numpy as np

import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm


def main(inpath, outpath):

    os.makedirs(outpath, exist_ok=True)

    embeddings = dict(np.load(inpath))

    models = {}

    for key in embeddings:

        data = embeddings[key]

        model = PCA()

        model.fit(data)

        models[key] = model
    
    with open(os.path.join(outpath, 'pca.pkl'), 'wb') as file:

        pickle.dump(models, file)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('inpath')
    parser.add_argument('outpath')

    main(**vars(parser.parse_args()))

