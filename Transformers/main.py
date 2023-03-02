import logging
import os
import sys
import pandas as pd
from glob import glob
import datetime
import numpy as np
import math

import torch
import torch.nn as nn

import pandas as pd
from sklearn.model_selection import train_test_split

from model_utils import get_metrics, tensorify, process_sample, predict
from data_utils import CropDataset
from cropclassifier import TransformerModel

import logging

loggerupds = logging.getLogger('update')


import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning) 


# ------------------------------- Train the model  -----------------------------------------------------

def load_model(model_path):
    # load the trained model and model related parameters
    model = TransformerModel()
    model.load_state_dict(torch.load(model_path))
    return model

def validate(model, dataset):

    y_pred = []
    y_true = []
    with torch.no_grad():
        for batchX, batchy in dataset.get_next_batch(use_data='validation', batch_size=10):
            ypred = model(batchX)
            #print("y_pred:", ypred)
            yp = ypred.argmax(axis=1).view(-1).numpy().tolist()

            yt = batchy.view(-1).numpy().tolist()
            #print("yt:", yt)
           
            y_pred.extend(yp)
            y_true.extend(yt)
    
    return get_metrics(ytrue=y_true, ypred=y_pred)

def train(train_fname, num_epochs, model_file, model, lr=0.001):

    dataset = CropDataset(train_fname)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

    best_f1 = 0
    best_metrics = []
    for e in range(num_epochs):
        losses = []
        epoch_num = e + 1
        for batchX, batchy in dataset.get_next_batch(batch_size=100):
            optimizer.zero_grad()
            ypred = model(batchX)
            loss = loss_function(ypred, batchy.view(-1))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        acc, pre, rec, f1s, cfm = validate(model, dataset)
        scheduler.step()
        if f1s > best_f1:
            best_f1 = f1s
            best_metrics = [acc, pre, rec, f1s, cfm]
            torch.save(model.state_dict(), model_file)
        print(f'EPOCH {epoch_num} :: Train Loss: {sum(losses)/len(losses):.6f} :: Validation Metrics: A[{acc:.2f}], P[{pre:.2f}], R[{rec:.2f}], F[{f1s:.2f}]')

    print(f'==========')
    print(f'BEST MODEL')
    print(model_file)
    print(f'Validation Metrics: A[{best_metrics[0]:.2f}], P[{best_metrics[1]:.2f}], R[{best_metrics[2]:.2f}], F[{best_metrics[3]:.2f}]')


    y_pred = []
    y_true = []
    with torch.no_grad():
        for batchX, batchy in dataset.get_next_batch(use_data='test'):
            ypred = model(batchX)
            yp = ypred.argmax(axis=1).view(-1).numpy().tolist()
            yt = batchy.view(-1).numpy().tolist()
            y_pred.extend(yp)
            y_true.extend(yt)
    acc, pre, rec, f1s, cfm = get_metrics(ypred=y_pred, ytrue=y_true)
    print(f'Test Metrics: A[{acc:.2f}], P[{pre:.2f}], R[{rec:.2f}], F[{f1s:.2f}]')
    

if __name__ == "__main__":
  
    print('test')
    finetune = False
    # load pretrained model
    if finetune:
        model_dir = "C:/Users/Lenovo/Downloads/cropProject/ModelTraining_Expts/Nizamabad_14pts"
        model_path = os.path.join(model_dir, 'savedmodel.pth')
        model = load_model(model_path)
    else:
        model = TransformerModel()
    
  
    ### Training ###
  
    # launch  model training
    model_dir = "/home/ubuntu/Amuthan_Agrix_Codes/Amuthan_Agirx_Model_training/trained_models/cotton/"
    model_path = os.path.join(model_dir, 'transformer_new_cotton_layers128_lr0.01_epoch400.pth')  
    
    db_csvpath = os.path.join("/home/ubuntu/Amuthan_Agrix_Codes/Interns_model_performance_analysis/Experiement-2/performance_analysis_test","nalgond_cotton_points-16122022-averageing.csv")

    
    num_epochs = 400
    lr = 1e-2
    
    train(train_fname=db_csvpath, num_epochs=num_epochs, model_file=model_path,model=model, lr=lr)
    '''
    ### testing ###
    db_csvpath = os.path.join("/home/ubuntu/Amuthan_Agrix_Codes/Interns_model_performance_analysis/Experiement-2/performance_analysis_test","nalgond_cotton_points-16122022-averageing.csv")
    
    model.load_state_dict(torch.load(model_path))
    print(model)
    nb_classes = 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    tl=[]
    pl=[]


    y_pred, y_true = process_sample(model, db_csvpath)
    
    #print(f"y_pred: {y_pred}, y_true: {y_true}")
    acc, pre, rec, f1s, cfm=get_metrics(y_true, y_pred )
    print( print(f'Test Metrics: A[{acc:.2f}], P[{pre:.2f}], R[{rec:.2f}], F[{f1s:.2f}]'))
    print(cfm)
    '''