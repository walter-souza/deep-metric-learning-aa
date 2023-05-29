import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

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

  tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
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

  if not os.path.exists(path_file):
    os.makedirs(path_file)

  plt.savefig(path_file + name_file)
  plt.clf()
  plt.cla()
  plt.close()

  model.train()