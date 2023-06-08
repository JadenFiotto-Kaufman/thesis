import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from ...StableDiffuser import StableDiffuser
from .get_embedding import get_embeddings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def edit_key(key):

    key = key.replace('unet.', '')
    key = key.replace('_blocks', '')
    key = key.replace('attentions.', '')
    key = key.replace('transformer.', '')
    key = key.replace('attn2.processor.', '')
    key = key.replace('attn', '')
    key = key.replace('hook', '')

    return key


def train(embeddings, targets, epochs, lr = .0003):

    model = torch.nn.Sequential(torch.nn.Linear(embeddings.shape[1], 1),  torch.nn.Sigmoid())
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    crit = torch.nn.BCELoss()

    for epoch in range(epochs):

        for i in range(embeddings.shape[0]):

            optim.zero_grad()

            embedding = embeddings[[i]]
            target = targets[[i]][None, :]

            pred = model(embedding)

            loss = crit(pred, target)

            loss.backward()

            optim.step()

    return model
            
@torch.no_grad()
def pred(embeddings, model):


    preds = []

    for i in range(embeddings.shape[0]):

        embedding = embeddings[[i]]

        pred = model(embedding)

        preds.append(pred.item())

    return torch.Tensor(preds)


def main(inpath, cls, outpath, device):

    diffuser = StableDiffuser(scheduler='EA').to(torch.device(device))

    df = pd.read_csv(inpath)

    embeddings = get_embeddings(diffuser, df['prompt'])

    targets = torch.Tensor(df['cls'] == cls)

    recalls = []

    _recalls = []

    for key in embeddings.keys():

        _embeddings = torch.from_numpy(embeddings[key])

        model = train(_embeddings, targets, 30)

        preds = pred(_embeddings, model) > .5

        acc = accuracy_score(targets, preds)
        pre = precision_score(targets, preds)
        rec = recall_score(targets, preds)


        _recalls.append(rec)


        if len(_recalls) == 8:

            recalls.append(np.array(_recalls).mean())

            _recalls = []

        

    plt.plot(range(len(recalls)), recalls)

    plt.savefig('../test.png')










if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('inpath')
    parser.add_argument('cls')
    parser.add_argument('outpath')
    parser.add_argument('--device', default='cuda:0')

    main(**vars(parser.parse_args()))
