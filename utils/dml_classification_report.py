import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from utils import dataset_utils
from utils import metrics_utils
import os

def data_classification_report(dl, model, step, device, neighbors):
    xall, yall = dataset_utils.get_dml_model_outputs(dl, model, device)
    
    xtrain, xtest, ytrain, ytest = train_test_split(xall, yall, test_size=0.33, random_state=123, stratify=np.array(yall))
        
    knn = KNeighborsClassifier(n_neighbors=neighbors, weights='distance')
    knn.fit(xtrain,ytrain)

    ypred = knn.predict(xtest)
    
    mtr = metrics_utils.get_metrics(ytest, ypred, step)
    report = metrics.classification_report(ytest, ypred, output_dict=True, zero_division=0)
    cm = confusion_matrix(ytest, ypred)
    return mtr, report, cm


def dml_data_classification_report(model, data_loader, device, neighbors, epoch, save_path, step):
    model.eval()
    mtr, report, cm = data_classification_report(data_loader, model, step, device, neighbors=neighbors)
    
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

    np.savetxt('{}/dml_cm_test_epoch{}_n{}.csv'.format(cms_path, epoch, neighbors), cm, delimiter=',', fmt='%i')

    plt.figure(figsize=(30,20))
    cm_plot = sns.heatmap(cm, annot=True)
    fig = cm_plot.get_figure()
    fig.savefig('{}/dml_cm_test_epoch{}_n{}.png'.format(cms_path, epoch, neighbors))
    
    model.train()

if (__name__ == '__main__'):
    import sys, transformers, torch
    parent = os.path.abspath('.')
    sys.path.insert(1, parent)
    from dml_distilbert import DMLDistilBert
    import datasets, argparse, parameters

    parser = argparse.ArgumentParser()
    parser = parameters.training_parameters(parser)
    config = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:', device)

    print('Importing tokenizer...')
    tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(config.base_model)

    print('Importing model...')
    model = DMLDistilBert(config.embedding_size, config.token_size, config=config.base_model)
    model.to(device)
    config.tokenizer = tokenizer

    print('Importing dataset...')
    dl_train, dl_test, n_classes = datasets.select('imdb62', config)

    print('Classification report...')
    dml_data_classification_report(model, dl_test, device, 3, 0, '_results/unit_tests', 'test')

    print('Done!')
