import torch
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Functions
def tensorify(data, input_type='file'):
    if input_type == 'file':
        dfl = pd.read_csv(data)

        features = dfl.iloc[:, :-1]
        labels = dfl.iloc[:, -1]
        features, labels = torch.tensor(features.values).float(), torch.tensor(labels.values).long()
        # features = features.view(-1, 2, 15)
        features = features.view(-1, 2, features.shape[1] // 2)
        #features= np.array(features)

        print(type(features), type(labels))
        
        labels = labels.view(-1, 1)
        
        #labels= np.array(labels)


        return features.transpose(2, 1), labels
    elif input_type == 'sample':
        features = torch.tensor(data).float()
        # features = features.view(-1, 2, 15)
        features = features.view(-1, 2, features.shape[1] // 2)
        features = features.transpose(2, 1)
        return features
    else:
        raise ValueError(f'input_type can either be "file", "sample". Cannot be {input_type}')


def predict(model, samples, return_prob=False):
    sample, ytrue= samples
    with torch.no_grad():
        ypred = torch.nn.functional.softmax(model(sample))
    
    result = ypred[:,1].tolist()
    
    # result = ypred.view(-1).tolist()[1]
    if return_prob:
        return result
    else:
        y_pred=[1 if x >= 0.5 else 0 for x in result]
        return y_pred, ytrue


def get_metrics(ytrue, ypred):
    acc = accuracy_score(y_true=ytrue, y_pred=ypred)
    pre = precision_score(y_true=ytrue, y_pred=ypred)
    rec = recall_score(y_true=ytrue, y_pred=ypred)
    f1s = f1_score(y_true=ytrue, y_pred=ypred)
    cfm = confusion_matrix(y_true=ytrue, y_pred=ypred)
    tn, fp, fn, tp= confusion_matrix(y_true=ytrue, y_pred=ypred).ravel()
    print(tn, fp, fn, tp)
    return acc, pre, rec, f1s, cfm


def process_sample(model, feature_list, return_prob=False):
    return predict(model, tensorify(feature_list, input_type='file'), return_prob=return_prob)