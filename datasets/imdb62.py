import torch
import numpy as np
from sklearn.model_selection import train_test_split

def get(config, dataset_path):
    X_global, y_global = load_dataset_imdb62(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X_global, y_global, test_size=config.test_size, stratify=y_global, random_state=config.seed)

    dataset_train = DatasetIMDB62(texts=X_train, labels=y_train, tokenizer=config.tokenizer, max_length=config.token_size)
    dataset_test = DatasetIMDB62(texts=X_test, labels=y_test, tokenizer=config.tokenizer, max_length=config.token_size)

    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size)
    dl_test  = torch.utils.data.DataLoader(dataset_test, batch_size=config.batch_size)

    n_classes = len(np.unique(y_train))

    return dl_train, dl_test, n_classes


def load_dataset_imdb62(data_path):
    file = open(data_path, 'r', encoding='utf8')
    lines = file.readlines()
    authors = []
    texts = []
    for line in lines:
        line = line.split('\n')[0]
        line = line.split('\t')
        author = line[1]
        content = line[5]
        if (author != '') and (content != ''):
            authors.append(author)
            texts.append(content)
        
    authors = np.unique(authors, return_inverse=True)[1]

    return texts, authors

class DatasetIMDB62(torch.utils.data.Dataset):
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
