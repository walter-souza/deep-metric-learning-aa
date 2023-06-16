import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils.data_visualization import *
import sklearn.metrics as metrics
from utils import dataset_utils
from utils import metrics_utils

def data_classification_report(dl, model, step, device):
    xall, yall = dataset_utils.get_traditional_model_outputs(dl, model, device)

    y_predict = np.argmax(xall, axis=-1)
    y_true = yall

    mtr = metrics_utils.get_metrics(y_true, y_predict, step)
    report = metrics.classification_report(y_true, y_predict, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_predict)
    return mtr, report, cm

def traditional_data_classification_report(model, data_loader, device, epoch, save_path, step):
    model.eval()
    neighbors = 3
    mtr, report, cm = data_classification_report(data_loader, model, step, device)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if epoch == 0:
        result_file = open('{}/results.csv'.format(save_path), 'w+')
        result_file.write('epoch,')
        result_file.write(','.join([str(key) for key in mtr.keys()]) + '\n')  
    else:
        result_file = open('{}/results.csv'.format(save_path), 'a+')  
    result_file.write('{},'.format(epoch))
    result_file.write(','.join([str(value) for value in mtr.values()]) + '\n')

    cms_path = '{}/confusion_matrix'.format(save_path)
    if not os.path.exists(cms_path):
        os.makedirs(cms_path)

    np.savetxt('{}/traditional_cm_test_epoch{}_n{}.csv'.format(cms_path, epoch, neighbors), cm, delimiter=',', fmt='%i')

    plt.figure(figsize=(30,20))
    cm_plot = sns.heatmap(cm, annot=True)
    fig = cm_plot.get_figure()
    fig.savefig('{}/traditional_cm_test_epoch{}_n{}.png'.format(cms_path, epoch, neighbors))
    
    model.train()