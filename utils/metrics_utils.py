import sklearn.metrics as metrics

def get_metrics(true, predict, step):
    acc = metrics.accuracy_score(true, predict)
    precision = metrics.precision_score(true, predict, average='macro')
    recall = metrics.recall_score(true, predict, average='macro')
    f1 = metrics.f1_score(true, predict, average='macro')
    
    mtr = {"accuracy_"+step: acc, 
            "precision_"+step: precision, 
            "recall_"+step: recall, 
            "f1_"+step: f1}
    return mtr