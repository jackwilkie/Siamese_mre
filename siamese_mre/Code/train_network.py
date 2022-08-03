# -*- coding: utf-8 -*-
"""
Train Siamese Network in Pytorch

Created on Tue Aug  2 20:52:47 2022

@author: jackw
"""

#------------------------- User specified parameters --------------------------

#specify model hyperparameters
hyperparameters = {  
                   'epochs' : 2000,
                   'batch_size' : 30000, 
                   'learning_rate' : 0.001,
                   'j' : 5,
                   'm' : 1.0,
                   'l2' : 0.001,
                   'metric' : 'euclidean_distance'
                   }

ep_log_interval = 100
pair_log_interval = 6000
pairs = 30000


dataset = 'CICIDS2017'
class_names = ['Benign', 'DoS (Hulk)', 'DoS (SlowLoris)', 'FTP', 'SSH']
architecture = 'De(25):Dr(0.1):De(20):Dr(0.05):De(15)'
data_path = 'C:\\Users\\jackw\\Desktop\\siamese_mre\\data\\cicids2017.csv'

'''

dataset = 'NSLKDD'
class_names = ['Normal', 'Dos', 'Probe', 'R2L', 'U2R']
architecture = 'De(98):Dr(0.1):de(79):Dr(0.1):de(59):Dr(0.1):de(39):Dr(0.1):de(20)'
data_path = 'C:\\Users\\jackw\\Desktop\\siamese_mre\\data\\nslkdd.csv'
'''

#---------------------- End of User specified parameters ----------------------

#set current directory
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import numpy as np
import torch as T

from read_data import get_data
from pair_functions import get_pairs

from custom_pytorch_functions import euclidean_distance, ContrastiveLoss,Siamese_Dataset,SiameseNet,Siamese_Train

from accuracy import Evaluate_Oneshot_Accuracy

import plot_functions



device = T.device('cpu')


#--------------------- Import Data and Create Data Loaders --------------------

x_train, y_train, x_test, y_test, x_val, y_val = get_data(data_path, 'label')  #import dataset
#x_train, y_train, x_val, y_val = get_data(data_path, 'label')

#generate data pairs
left_samples_train, right_samples_train, y_train_pairs = get_pairs(x_train, y_train, pairs, seed = 9320824, log_interval=pair_log_interval)
left_samples_val, right_samples_val, y_val_pairs = get_pairs(x_val, y_val, 3277, seed = 32852769, log_interval=pair_log_interval)


#create train and validation datasets
train_ds = Siamese_Dataset(left_samples_train, right_samples_train, y_train_pairs)
val_ds = Siamese_Dataset(left_samples_val, right_samples_val, y_val_pairs)


#creat data loaders
train_dl = T.utils.data.DataLoader(train_ds, batch_size=hyperparameters['batch_size'], shuffle=True)
val_dl = T.utils.data.DataLoader(val_ds, batch_size=hyperparameters['batch_size'], shuffle=True)



#--------------------------- Create and Train Model ---------------------------

siamese_network = SiameseNet(input_size = len(x_train[0]), architecture = architecture).to(device)  #instantiate network


#loss_func = ContrastiveLoss()  #define loss function using euc distance
loss_func = ContrastiveLoss(metric = euclidean_distance, m = hyperparameters['m'])  #define loss function using snr distance

optimiser = T.optim.Adam(siamese_network.parameters(), lr=hyperparameters['learning_rate'] , weight_decay=hyperparameters['l2'])  #define optimiser

history = Siamese_Train(siamese_network, optimiser, loss_func, train_dl, val_dl, hyperparameters['epochs'], ep_log_interval = ep_log_interval)  #train model
plot_functions.plot_loss_curve(history)  #plot loss cureve

#------------------------------- Evaluate Model -------------------------------

#x_test = np.load('C:\\Users\\jackw\\Desktop\\x.npy')
#y_test = np.load('C:\\Users\\jackw\\Desktop\\y.npy')

#generate and plot confusion matrix
cm = Evaluate_Oneshot_Accuracy(siamese_network, x_train, y_train, x_test, y_test, metric = euclidean_distance, j = hyperparameters['j'])
plot_functions.plot_cm(cm,class_names, title = f'{dataset} Confusion Matrix')


acc = np.trace(cm)/np.sum(cm)  #find model accuracy

#find recall scores
recall_scores = [cm[i][i]/np.sum(cm[i]) for i in range(len(cm))]  #calculate recall value for each class
     
