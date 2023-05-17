import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
from tqdm import tqdm
from dml_distilbert_v3 import DistilBertV1

import utils
import criteria
import datasets

import math
import argparse, parameters

########## CONFIG ##########
parser = argparse.ArgumentParser()
parser = parameters.training_parameters(parser)
config = parser.parse_args()

########## MODEL AND TOKENIZER ##########
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(config.base_model)
model = DistilBertV1(config.embedding_size, config.token_size, config=config.base_model)
config.tokenizer = tokenizer

########## DATASET AND DATALOADER ##########
dl_train, dl_test, n_classes, dataset_size = datasets.select(config.dataset, config)
config.n_classes = n_classes
print('n_classes:', config.n_classes)

########## LOSS ##########
loss_function = criteria.select(config.loss, config)
print('loss:', config.loss)

########## OPTIMIZER ##########
optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': loss_function.parameters()}], lr=1e-4, weight_decay=5e-4)

########## GPU ##########
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device:', device)
model.to(device)
loss_function.to(device)

########## TRAINING ##########
model.train()
loss_function.train()
batch_list_loss = []

path_file = 'data_visualization/'
minimun_loss = math.inf
for epoch in range(config.n_epochs):
    data_iterator = tqdm(dl_train, desc='Epoch {} Training...'.format(epoch))
    current_batch_loss = []
    
    ########## VISUALIZATION ##########
    name_file = 'train_epoch_' + str(epoch) + '.png'
    utils.data_visualization.view_data(dl_train, model, path_file, name_file, device)

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
        ########## SAVE MODEL ##########
        print('Previous loss: {} | New loss: {} | Saving model...'.format(minimun_loss, np.mean(current_batch_loss)))
        save_name_model = 'results/deep_metric_learning/{model.name}/dataset/{model.name}_{config.dataset}_{config.loss}_es{config.embedding_size}_ts{config.token_size}_bs{config.batch_size}_s{config.seed}.pth'
        name_file = 'train_epoch_' + str(epoch) + '.png'
        utils.data_visualization.view_data(dl_train, model, path_file, name_file, device)
        
        # save_name_model = 'models/dml_distilbert.pth'
        # torch.save(model.state_dict(), save_name_state)
        torch.save(model, save_name_model)
        
        minimun_loss = np.mean(current_batch_loss)
        
    plt.figure(figsize=(30,20))
    plt.plot(batch_list_loss)
    plt.savefig(path_file + 'batch_list_loss_all.png')
    plt.clf()
    plt.cla()
    plt.close()

    plt.figure(figsize=(30,20))
    plt.plot(batch_list_loss[::int(len(dataset_size)/config.batch_size)])
    plt.savefig(path_file + 'batch_list_loss.png')
    plt.clf()
    plt.cla()
    plt.close()
        

