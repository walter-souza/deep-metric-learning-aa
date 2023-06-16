import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
from tqdm import tqdm
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

########## TOKENIZER ##########
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(config.base_model)
config.tokenizer = tokenizer

########## DATASET AND DATALOADER ##########
dl_train, dl_test, n_classes = datasets.select(config.dataset, config)
config.n_classes = n_classes
print('n_classes:', config.n_classes)

########## MODEL ##########
model = transformers.AutoModelForSequenceClassification.from_pretrained(config.base_model, num_labels=config.n_classes)
config.modelname = 'TraditionalDistilBert'

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
save_folder = '_results/traditional_classification/{datasetname}/{modelname}_{datasetname}_ts{ts}_bs{bs}_s{seed}'.format(
    datasetname=config.dataset,
    modelname=config.modelname,
    ts=config.token_size,
    bs=config.batch_size,
    seed=config.seed)
config.save_path = save_folder
print('Save path:', config.save_path)

########## VISUALIZATION ##########
# print('Generating visualization before training...')
# data_visualization_path = '{}/data_visualization/'.format(config.save_path)
# if not os.path.exists(data_visualization_path):
#     os.makedirs(data_visualization_path)
# data_visualization_name = 'before_train.png'
# utils.data_visualization.traditional_view_data(dl_train, model, data_visualization_path, data_visualization_name, device)
# print('Done!')

minimun_loss = math.inf
for epoch in range(config.n_epochs):
    data_iterator = tqdm(dl_train, desc='Epoch {} Training...'.format(epoch))
    current_batch_loss = []  
    for ii, item in enumerate(data_iterator):
        input_ids = item['input_ids'].to(device)
        attention_mask = item['attention_mask'].to(device)
        labels = item['labels'].to(device)
        labels = labels.long()

        output = model(input_ids, attention_mask, labels=labels)

        loss = output.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        current_batch_loss.append(loss.item())
        batch_list_loss.append(loss.item())

        data_iterator.set_description('Epoch {} Training... (loss {})'.format(epoch, round(loss.item(), 4)))

    if (np.mean(current_batch_loss) < minimun_loss):
        ########## SAVE MODEL ##########
        print('Previous loss: {} | New loss: {} | Saving model...'.format(minimun_loss, np.mean(current_batch_loss)))
        minimun_loss = np.mean(current_batch_loss)
        
        data_visualization_path = '{}/data_visualization/'.format(config.save_path)
        data_visualization_name = 'train_epoch_{}.png'.format(epoch)
        utils.data_visualization.traditional_view_data(dl_train, model, data_visualization_path, data_visualization_name, device)
        
        save_name_model = '{}/model.pth'.format(config.save_path)
        torch.save(model, save_name_model)

        model.eval()
        utils.traditional_classification_report.traditional_data_classification_report(model, dl_test, device, epoch, config.save_path, 'test')
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
        

