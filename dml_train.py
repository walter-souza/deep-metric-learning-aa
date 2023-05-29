import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
from tqdm import tqdm
from dml_distilbert import DMLDistilBert

import utils
import criteria
import datasets
import os
import math
import argparse, parameters

########## CONFIG ##########
parser = argparse.ArgumentParser()
parser = parameters.training_parameters(parser)
config = parser.parse_args()

########## MODEL AND TOKENIZER ##########
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(config.base_model)
model = DMLDistilBert(config.embedding_size, config.token_size, config=config.base_model)
config.tokenizer = tokenizer
config.modelname = model.name

########## DATASET AND DATALOADER ##########
dl_train, dl_test, n_classes = datasets.select(config.dataset, config)
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

########## SAVE FOLDER ##########
config.save_path = save_folder = 'results/deep_metric_learning/{datasetname}/{modelname}_{datasetname}_{loss}_es{es}_ts{ts}_bs{bs}_s{seed}'.format(
    datasetname=config.dataset,
    modelname=config.modelname,
    loss=config.loss,
    es=config.embedding_size,
    ts=config.token_size,
    bs=config.batch_size,
    seed=config.seed)
print(config.save_path)
########## VISUALIZATION ##########
data_visualization_path = '{}/data_visualization/'.format(config.save_path)
if not os.path.exists(data_visualization_path):
    os.makedirs(data_visualization_path)
data_visualization_name = 'before_train.png'

utils.data_visualization.view_data(dl_train, model, data_visualization_path, data_visualization_name, device)

path_file = 'data_visualization/'
minimun_loss = math.inf
for epoch in range(config.n_epochs):
    data_iterator = tqdm(dl_train, desc='Epoch {} Training...'.format(epoch))
    current_batch_loss = []    
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
        minimun_loss = np.mean(current_batch_loss)
        
        data_visualization_path = '{}/data_visualization/'.format(config.save_path)
        data_visualization_name = 'train_epoch_{}.png'.format(epoch)
        utils.data_visualization.view_data(dl_train, model, data_visualization_path, data_visualization_name, device)
        
        # save_name_model = 'models/dml_distilbert.pth'
        # torch.save(model.state_dict(), save_name_state)
        save_name_model = '{}/model.pth'.format(config.save_path)
        torch.save(model, save_name_model)

        model.eval()
        utils.dml_classification_report.dml_data_classification_report(model, dl_test, device, 7, epoch, config.save_path, 'train')
        model.train()
        
        
    plt.figure(figsize=(30,20))
    plt.plot(batch_list_loss)
    plt.savefig(config.save_path + '/batch_list_loss_all.png')
    plt.clf()
    plt.cla()
    plt.close()

    plt.figure(figsize=(30,20))
    plt.plot(batch_list_loss[::len(dl_train)])
    plt.savefig(config.save_path + '/batch_list_loss.png')
    plt.clf()
    plt.cla()
    plt.close()
        

