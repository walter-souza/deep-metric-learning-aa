import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from pytorch_metric_learning import miners, losses
from dml_distilbert_v3 import DistilBertV3
from utils.dataset_utils import *
from loss.partial_fc import *
import math
from torch import distributed


''' =============== Constants ==============='''
DATASET_REDUCTION = 0.1
SEED = 21
BATCH_SIZE = 100
NUM_CLASSES = 62
EMBEDDING_SIZE = 256   #TODO: Alterar no modelo para ser dinâmico
TOKEN_MAX_LENGTH = 128 #TODO: Alterar no modelo para ser dinâmico
EPOCHS = 20
BASE_MODEL = "distilbert-base-uncased"

''' =============== Model and Tokenizer ==============='''
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
model = DistilBertV3(EMBEDDING_SIZE, TOKEN_MAX_LENGTH, config=BASE_MODEL)

''' =============== Dataset and DataLoader ==============='''
X_train, y_train = load_dataset_imdb62('/home/walter/TCC/datasets/imdb62/imdb62_train.txt')
X_test,  y_test  = load_dataset_imdb62('/home/walter/TCC/datasets/imdb62/imdb62_test.txt')

# X_global, y_global = load_dataset_imdb62('/home/walter/TCC/datasets/imdb62/imdb62.txt')
# _, X, _, y = train_test_split(X_global, y_global, test_size=DATASET_REDUCTION, shuffle=True, stratify=y_global, random_state=SEED)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

dataset_train = DatasetIMDB62(texts=X_train, labels=y_train, tokenizer=tokenizer, max_length=TOKEN_MAX_LENGTH)
dataset_test = DatasetIMDB62(texts=X_test, labels=y_test, tokenizer=tokenizer, max_length=TOKEN_MAX_LENGTH)

dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE)
dl_test  = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE)

''' =============== Loss and Optimizer ==============='''
# loss_function = losses.ArcFaceLoss(num_classes=NUM_CLASSES, embedding_size=EMBEDDING_SIZE, margin=28.6, scale=64)
# loss_function = losses.TripletMarginLoss()
loss_function = losses.MultiSimilarityLoss()
'''world_size = 1
rank = 0
distributed.init_process_group(
    backend="nccl",
    init_method="tcp://127.0.0.1:12584",
    rank=rank,
    world_size=world_size,
)
margin_loss = CombinedMarginLoss(64,1,0.5,0,0)
loss_function = PartialFC_V2(margin_loss=margin_loss, embedding_size=EMBEDDING_SIZE, num_classes=NUM_CLASSES, sample_rate=0.2)'''

optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': loss_function.parameters()}], lr=1e-4, weight_decay=5e-4)

''' =============== GPU Config ==============='''
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print('Device:', device)

model.to(device)
loss_function.to(device)

''' =============== Training ==============='''
model.train()
loss_function.train()
batch_list_loss = []

path_file = '/home/walter/TCC/DeepMetricLearningV3/data_visualization/'
minimun_loss = math.inf
for epoch in range(EPOCHS):
    data_iterator = tqdm(dl_train, desc='Epoch {} Training...'.format(epoch))
    current_batch_loss = []
    
    ''' =============== Visualization ==============='''
    name_file = 'train_epoch_' + str(epoch) + '.png'
    view_data(dl_train, model, path_file, name_file, device)

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
        save_name_state = '/home/walter/TCC/DeepMetricLearningV3/models/dml_distilbert_state_dict.pth'
        save_name_model = '/home/walter/TCC/DeepMetricLearningV3/models/dml_distilbert.pth'
        torch.save(model.state_dict(), save_name_state)
        torch.save(model, save_name_model)
        
        minimun_loss = np.mean(current_batch_loss)
        
    plt.figure(figsize=(30,20))
    plt.plot(batch_list_loss)
    plt.savefig(path_file + 'batch_list_loss_all.png')
    plt.clf()

    plt.figure(figsize=(30,20))
    plt.plot(batch_list_loss[::int(len(X_train)/BATCH_SIZE)])
    plt.savefig(path_file + 'batch_list_loss.png')
    plt.clf()
        

