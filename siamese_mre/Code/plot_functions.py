# -*- coding: utf-8 -*-
"""
Functions to visualise data

Created on Tue May 31 15:58:05 2022

@author: jackw
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#plot model loss curves
def plot_loss_curve(history, save_path = ''):
    #summarize history for loss
    plt.figure()
    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    if len(save_path) > 0:
        plt.savefig(save_path + "_loss_curve.png")
    
    plt.show()
    

def plot_cm (cm, c_names, cmap = 'Blues', cbar = False, save_path = None, title = None, xyplotlabels = None):
    '''
    inspired by:
        https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
    
    '''    
    
    #find performance metrics
    acc = np.trace(cm)/np.sum(cm)  #find model accuracy
    recall_scores = [cm[i][i]/np.sum(cm[i]) for i in range(len(cm))]  #calculate recall value for each class
    percentge_values = ((cm.T/cm.sum(axis=1)).T)*100  #divide rows by row sum and scale to percentage
    
    labels = []
    
    for r, row in enumerate(cm):
        row_labels = [f'{int(cm[r][i])}\n{round(percentge_values[r][i],2)}%' for i in range(len(row))]
      
        labels.append(row_labels)
    
    
    #plot confusion matrix
    plt.figure()
    
    sns.heatmap(cm, annot= labels, cmap=cmap, cbar = cbar, xticklabels=c_names, yticklabels=c_names, fmt = '')
    
    if title is not None:
        plt.title(title)
    
    
    if xyplotlabels is not None:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + f'\nAccuracy = {round(acc*100,2)}%')
    else:
        plt.xlabel(f'Accuracy = {round(acc*100,2)}%')
     
    
    #sns.heatmap(cm, annot= labels, cmap=cmap, cbar = cbar, xticklabels=c_names, yticklabels=c_names, fmt = '')
    #plt.show()
    
    if save_path is not None :
        plt.savefig(save_path, transparent = True)
        
    plt.show()