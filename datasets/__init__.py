from datasets import imdb62, c50
import torch

def select(dataset, config):
    if dataset == 'imdb62':
        path = 'datasets\my_datasets\imdb62\imdb62.txt'
        return imdb62.get(config, path)
    
    if dataset == 'c50':
        path = 'datasets\my_datasets\c50'
        return c50.get(config, path)
    
    raise NotImplementedError('Dataset {} not implemented!'.format(dataset))