import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from dml_distilbert_v3 import DistilBertV3
import utils
import criteria
from criteria.partial_fc import *
import math


import argparse, parameters

########## CONFIG ##########
parser = argparse.ArgumentParser()
parser = parameters.training_parameters(parser)
config = parser.parse_args()

########## CONSTANTS ##########
DATASET_REDUCTION = 0.1
SEED = 21
BATCH_SIZE = 64
EMBEDDING_SIZE = 256   #TODO: Alterar no modelo para ser dinâmico
TOKEN_MAX_LENGTH = 128 #TODO: Alterar no modelo para ser dinâmico
EPOCHS = 20
BASE_MODEL = "distilbert-base-uncased"

########## MODEL AND TOKENIZER ##########
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
model = DistilBertV3(EMBEDDING_SIZE, TOKEN_MAX_LENGTH, config=BASE_MODEL)

########## DATASET AND DATALOADER ##########
X_global, y_global = utils.dataset.load_dataset_imdb62('datasets/imdb62/imdb62.txt')
X_train, X_test, y_train, y_test = utils.dataset.train_test_split(X_global, y_global, test_size=0.25, stratify=y_global, random_state=SEED)

dataset_train = utils.dataset.DatasetIMDB62(texts=X_train, labels=y_train, tokenizer=tokenizer, max_length=TOKEN_MAX_LENGTH)
dataset_test = utils.dataset.DatasetIMDB62(texts=X_test, labels=y_test, tokenizer=tokenizer, max_length=TOKEN_MAX_LENGTH)

dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE)
dl_test  = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE)

n_classes = len(np.unique(y_train))
config.n_classes = n_classes
print('n_classes:', n_classes)

########## LOSS AND OPTIMIZER ##########
loss_function = criteria.select(config.loss, config)
print('loss:', config.loss)

optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': loss_function.parameters()}], lr=1e-4, weight_decay=5e-4)

########## GPU ##########
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device:', device)
model.to(device)
loss_function.to(device)

''' =============== Training ==============='''
model.train()
loss_function.train()
batch_list_loss = []

path_file = 'data_visualization/'
minimun_loss = math.inf
for epoch in range(EPOCHS):
    data_iterator = tqdm(dl_train, desc='Epoch {} Training...'.format(epoch))
    current_batch_loss = []
    
    ''' =============== Visualization ==============='''
    name_file = 'train_epoch_' + str(epoch) + '.png'
    utils.dataset.view_data(dl_train, model, path_file, name_file, device)

    for ii, item in enumerate(data_iterator):
        input_ids = item['input_ids'].to(device)
        attention_mask = item['attention_mask'].to(device)
        labels = item['labels'].to(device)
        labels = labels.long()

        embeddings = model(input_ids, attention_mask)

        loss = loss_function(embeddings, labels)

        optimizer.zero_grad()

        loss.backward()
        current_batch_loss.append(loss.item())
        batch_list_loss.append(loss.item())

        data_iterator.set_description('Epoch {} Training... (loss {})'.format(epoch, round(loss.item(), 4)))
        
        optimizer.step()

    if (np.mean(current_batch_loss) < minimun_loss):
        ''' =============== Save Model ==============='''
        print('Previous loss: {} | New loss: {} | Saving model...'.format(minimun_loss, np.mean(current_batch_loss)))
        save_name_state = 'models/dml_distilbert_state_dict.pth'
        save_name_model = 'models/dml_distilbert.pth'
        torch.save(model.state_dict(), save_name_state)
        torch.save(model, save_name_model)
        
        minimun_loss = np.mean(current_batch_loss)
        
    plt.figure(figsize=(30,20))
    plt.plot(batch_list_loss)
    plt.savefig(path_file + 'batch_list_loss_all.png')
    plt.clf()
    plt.cla()
    plt.close()

    plt.figure(figsize=(30,20))
    plt.plot(batch_list_loss[::int(len(X_train)/BATCH_SIZE)])
    plt.savefig(path_file + 'batch_list_loss.png')
    plt.clf()
    plt.cla()
    plt.close()
        

