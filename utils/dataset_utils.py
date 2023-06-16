import torch
import numpy as np

def get_dml_model_outputs(dl, model, device):
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

    return xall, yall  

def get_traditional_model_outputs(dl, model, device):
    xall = None
    y_true = []
    with torch.no_grad():
        for ii,item in enumerate(dl):
            input_ids = item['input_ids'].to(device)
            attention_mask = item['attention_mask'].to(device)
            labels = item['labels'].to(device)
            # labels = labels.long()

            output = model(input_ids, attention_mask)

            if xall is None:
                xall = output.logits.cpu().data.numpy()
            else:    
                xall = np.concatenate((xall, output.logits.cpu().data.numpy()))
            y_true = np.concatenate((y_true, labels.cpu().data.numpy()))
    

    y_true = y_true.astype(int)

    return xall, y_true