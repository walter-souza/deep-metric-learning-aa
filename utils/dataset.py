import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.model_selection import train_test_split


''' =============== IMDB62 =============== ''' 

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

''' =============== Data Visualization =============== ''' 
def view_data(dl, model, path_file, name_file, device):

  model.eval()
  
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

  tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
  tsne_results = tsne.fit_transform(xall)
  df_tsne = pd.DataFrame(tsne_results)
  df_tsne['label'] = yall
  df_tsne.columns = ['X', 'Y', 'label']
  plt.figure(figsize=(30,20))
  qtd_classes = len(np.unique(yall))
  sns_plot = sns.scatterplot(
    x="X", y="Y",
    hue="label",
    palette=sns.color_palette("hls", qtd_classes),
    data=df_tsne,
    legend="full",
    alpha=0.7
  )
  plt.savefig(path_file + name_file)
  plt.clf()
  plt.cla()
  plt.close()

  model.train()