
import pdb

import pandas as pd
from deeplearning.datasets import Dataset
from deeplearning.util import str2bool

class DMSTDataset(Dataset):
    def __init__(self, 
        dataset_path,
        source=None, 
        heldout=None, 
        segmentable=None, 
        objects=None,
        negation=False,
        n_examples=None,
        random_sample_amount=None,
        **kwargs):

        super().__init__(**kwargs)
        self.random_sample_amount = random_sample_amount
        self.dataset = pd.read_csv(dataset_path, keep_default_na=False)
        self.negation = negation

        if source is not None:
            if type(source) == str:
                source = [source]

            if n_examples is not None:
                if len(n_examples) < len(source):
                    raise ValueError('Specify the number of examples for all the data source!')
                self.dataset = pd.concat([self.dataset[self.dataset["source"] == s].iloc[:int(n)] for s,n in zip(source, n_examples)])
            else:
                self.dataset = self.dataset[self.dataset["source"].isin(source)]
        if heldout is not None:
            if heldout:
                self.dataset = self.dataset[self.dataset["heldout_word"]]
            else:
                self.dataset = self.dataset[~self.dataset["heldout_word"]]
        if segmentable is not None:
            if segmentable:
                self.dataset = self.dataset[self.dataset["segmentable"]]
            else:
                self.dataset = self.dataset[~self.dataset["segmentable"]]
        if objects is not None:
           self.dataset = self.dataset[self.dataset['object'].isin(objects)]

        if self.dataset_type != Dataset.DatasetType.predict:

            if negation:

                self.to_return = [
                    'positive_prompt', 
                    'neutral_prompt', 
                    'negative_prompt', 
                    'negation_token']

            else:

                self.to_return = [
                    'positive_prompt', 
                    'neutral_prompt', 
                    'positive_prompt', 
                    'object']

        else:

            self.to_return = [
                    'case_number', 
                    'source', 
                    'positive_prompt', 
                    'neutral_prompt', 
                    'negative_prompt', 
                    'object',
                    'evaluation_seed']        

    def __len__(self):

        if self.dataset_type == Dataset.DatasetType.validate:
            return 1

        if self.random_sample_amount is not None:
            return self.random_sample_amount
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.random_sample_amount is not None:
            item = self.dataset.sample(1).iloc[0]
        else:
            item = self.dataset.iloc[idx]
        out = [getattr(item, x) for x in self.to_return]

        if self.dataset_type == Dataset.DatasetType.predict:
            return out

        return out, 0

    @staticmethod
    def args(parser):

        parser.add_argument('--dataset_path',type=str, required=True)
        parser.add_argument('--source', nargs='+', type=str, default=None, const=None)
        parser.add_argument('--heldout', type=str2bool, nargs='?', default=None, const=True)
        parser.add_argument('--segmentable', type=str2bool, nargs='?',default=None, const=True)
        parser.add_argument('--negation', action='store_true')
        parser.add_argument('--n_examples', type=int, default=None, nargs='+')
        parser.add_argument('--random_sample_amount', type=int, default=None)
        super(DMSTDataset, DMSTDataset).args(parser)

if __name__ == '__main__':
    dd = EvaluationDataset()
    pdb.set_trace()
    dd = EvaluationDataset(source='bare_nouns')
    print(dd.dataset)
    dd = EvaluationDataset(source=['bare_nouns', 'coco_captions_with_no'])
    print(dd.dataset)
    dd = EvaluationDataset(source='bare_nouns', heldout=False)
    print(dd.dataset)
    print(dd[0])
    print('\n\n\n')
    dd = EvaluationDataset(source='bare_nouns', heldout=False, to_return=['evaluation_seed', 'negative_prompt'])
    print(dd.dataset)
    print(dd[0])