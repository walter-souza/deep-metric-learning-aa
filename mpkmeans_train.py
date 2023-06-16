import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import random
from dml_distilbert import DMLDistilBert
import transformers
from tqdm import tqdm
import datasets
import argparse, parameters


def get_dataloader_y(dl):
    all_labels = []
    with torch.no_grad():
        for ii,item in enumerate(dl):
            labels = np.array(item['labels'])
            all_labels = np.concatenate((all_labels, labels))
    return all_labels


def get_model_output(dl, model, device):
    xall = None
    yall = []
    with torch.no_grad():
        for ii,item in enumerate(dl):
            input_ids = item['input_ids'].to(device)
            attention_mask = item['attention_mask'].to(device)
            labels = item['labels'].to(device)
            labels = labels.long()

            embeddings = model(input_ids, attention_mask)
            if xall is None:
                xall = embeddings.data.cpu().numpy()
            else:
                xall = np.concatenate((xall, embeddings.data.cpu().numpy()))
            yall = np.concatenate((yall, labels.data.cpu().numpy()))  

    return xall, yall  


def get_indexes(dl_train, model, device, neighbors):
    x_all, y_true = get_model_output(dl_train, model, device)
    xtrain, xtest, ytrain, ytest = train_test_split(x_all, y_true, test_size=0.33, random_state=123, stratify=np.array(y_true))

    knn = KNeighborsClassifier(n_neighbors=neighbors, weights='distance')
    knn.fit(xtrain, ytrain)
    y_pred = knn.predict(xtest)

    compared = np.array(y_true == y_pred)
    indexes = np.where(compared == False)
    return indexes

def balance_index(indexes, dl_train):
    y_true = dl_train.y_true #TODO get y_true from dl_train
    uniques, labels_count = np.unique(y_true[indexes], return_counts=True)
    max_count = max(labels_count)
    balanced = indexes.to_list()
    for i, label in enumerate(uniques):
        cur_count = labels_count[i]
        if max_count - cur_count > 0:
            samples = random.sample(np.range(len(y_true)[y_true == label]), max_count - cur_count)
        balanced += samples.to_list()

    return balanced

def main():
    parser = argparse.ArgumentParser()
    parser = parameters.training_parameters(parser)
    config = parser.parse_args()

    config.embedding_size = 256
    config.token_size = 256
    config.batch_size = 65

    model_path = '_results\deep_metric_learning\imdb62\DMLDistilBert_imdb62_arcface_es256_ts64_bs64_s0\model.pth'

    tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(config.base_model)
    config.tokenizer = tokenizer
    model = torch.load(model_path)
    print('Getting dataset')
    dl_train, dl_test, n_classes = datasets.select(config.dataset, config)
    print('Done!')
    print(get_dataloader_y(dl_train))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    indexes = get_indexes(dl_train, model, device, 7)
    balanced_indexes = balance_index(indexes, dl_train)

    # create ml and cl
    # initialize mpckmeans
    # train mpckmeans
    # test on splited test dl[2]
    # get metric results

    pass

if __name__ == '__main__':
    main()