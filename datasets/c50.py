import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os

def get(config, dataset_folder_path):
    X_global, y_global = load_dataset(dataset_folder_path)
    X_train, X_test, y_train, y_test = train_test_split(X_global, y_global, test_size=config.test_size, stratify=y_global, random_state=config.seed)

    dataset_train = DatasetC50(texts=X_train, labels=y_train, tokenizer=config.tokenizer, max_length=config.token_size)
    dataset_test = DatasetC50(texts=X_test, labels=y_test, tokenizer=config.tokenizer, max_length=config.token_size)

    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size)
    dl_test  = torch.utils.data.DataLoader(dataset_test, batch_size=config.batch_size)

    n_classes = len(np.unique(y_train))

    return dl_train, dl_test, n_classes

def load_dataset(dataset_folder_path):
    authors = []
    texts = []
    for folder in os.listdir(dataset_folder_path):
        folder_path = os.path.join(dataset_folder_path, folder)
        for author in os.listdir(folder_path):
            author_path = os.path.join(folder_path, author)
            for text_file_name in os.listdir(author_path):
                text_file_path = os.path.join(author_path, text_file_name)
                text = open(text_file_path, 'r').read()
                authors.append(author)
                texts.append(text)
    labels = np.unique(authors, return_inverse=True)[1]

    return texts, labels

class DatasetC50(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256, padding='max_length', truncation=True):
        self.texts = texts
        self.labels = [int(label) for label in labels]
        self.encodings = tokenizer(texts, max_length=max_length, padding=padding, truncation=truncation)
  
    def __getitem__(self, i):
        item = {}
        item = {key : torch.tensor(val[i]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item
  
    def __len__(self):
        return len(self.labels)
  
    def get_labels(self):
        return self.labels

    def get_texts(self):
        return self.texts


if __name__ == '__main__':
    path = 'my_datasets\C50'
    import sys, transformers, torch
    parent = os.path.abspath('..')
    sys.path.insert(1, parent)
    import argparse, parameters

    parser = argparse.ArgumentParser()
    parser = parameters.training_parameters(parser)
    config = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:', device)

    print('Importing tokenizer...')
    tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(config.base_model)

    config.tokenizer = tokenizer
    print(get(config, path))