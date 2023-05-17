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
    tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    # model = torch.load('/home/walter/TCC/DeepMetricLearningV3/models/dml_distilbert.pth')
    model = torch.load('/home/walter/TCC/DeepMetricLearningV3/models/dml_imdb_ArcFaceLoss/dml_distilbert.pth')
    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    X_test,  y_test  = load_dataset_imdb62('/home/walter/TCC/datasets/imdb62/imdb62_test.txt')
    dataset_test = DatasetIMDB62(texts=X_test, labels=y_test, tokenizer=tokenizer, max_length=TOKEN_MAX_LENGTH)
    dl_test  = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE)

    model.eval()

    neighbors = 3
    mtr, report, cm = data_classification_report(dl_test, model, 'test', device, neighbors=neighbors)
    
    for (metric, value) in mtr.items():
        print('{}: {}'.format(metric, value))

    plt.figure(figsize=(30,20))
    cm_plot = sns.heatmap(cm, annot=True)
    fig = cm_plot.get_figure()
    fig.savefig('dml_cm_test_n{}.png'.format(neighbors))

if __name__ == '__main__':
    main()