import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from pytorch_metric_learning import miners, losses
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from utils.data_visualization import *
from utils.constants import *
import sklearn.neighbors
import sklearn.metrics as metrics
import datasets
import utils

import argparse, parameters

def data_classification_report(dl, model, step, device, neighbors):
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
    
    xtrain, xtest, ytrain, ytest = train_test_split(xall, yall, test_size=0.33, random_state=123,stratify=np.array(yall))
        
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=neighbors,weights='distance')
    knn.fit(xtrain,ytrain)

    ypred = knn.predict(xtest)
    ypredp = knn.predict_proba(xtest)

    acc = metrics.accuracy_score(ytest, ypred)
    precision = metrics.precision_score(ytest,ypred,average='macro')
    recall = metrics.recall_score(ytest,ypred,average='macro')
    f1 = metrics.f1_score(ytest,ypred,average='macro')
    top_k_2 = metrics.top_k_accuracy_score(ytest,ypredp,k=2)
    top_k_3 = metrics.top_k_accuracy_score(ytest,ypredp,k=3)
    top_k_4 = metrics.top_k_accuracy_score(ytest,ypredp,k=4)
        
    mtr = {"acc_"+step: acc, 
            "precision_"+step: precision, 
            "recall_"+step: recall, 
            "f1_"+step: f1, 
            "top_k_2_"+step: top_k_2, 
            "top_k_3_"+step: top_k_3, 
            "top_k_4_"+step: top_k_4}
    report = metrics.classification_report(ytest, ypred, output_dict=True, zero_division=0)
    cm = confusion_matrix(ytest, ypred)
    return mtr, report, cm

def main():
    ########## CONFIG ##########
    parser = argparse.ArgumentParser()
    parser = parameters.training_parameters(parser)
    config = parser.parse_args()

    if (config.model_path == ''):
        load_folder = 'results/deep_metric_learning/{datasetname}/{modelname}_{datasetname}_{loss}_es{es}_ts{ts}_bs{bs}_s{seed}'.format(
            datasetname=config.dataset,
            modelname=config.modelname,
            loss=config.loss,
            es=config.embedding_size,
            ts=config.token_size,
            bs=config.batch_size,
            seed=config.seed)
    else:
        load_folder = config.model_path

    ########## MODEL AND TOKENIZER ##########
    tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(config.base_model)
    model_path = load_folder + '/model.pth'
    model = torch.load(model_path)
    config.tokenizer = tokenizer
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ########## DATASET AND DATALOADER ##########
    _, dl_test, n_classes = datasets.select(config.dataset, config)
    config.n_classes = n_classes
    print('n_classes:', config.n_classes)

    ########## CLASSIFICATION REPORT ##########
    utils.dml_classification_report.dml_data_classification_report(model, dl_test, device, 7, 0, 'dml_test', 'test')
  
if __name__ == '__main__':
    main()