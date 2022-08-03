# -*- coding: utf-8 -*-
"""
Import and return data

Created on Tue Aug  2 20:38:13 2022

@author: jackw
"""

#set directory and import required libraies
import os

#set current directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def get_data(data_path, target = 'label', exclude = None, seed = 385929325):
    
    print('importing data')

    data_df = pd.read_csv(data_path)  #read data into df
    data_df = data_df.iloc[: , 1:]  #drop index columns
        
    
    #get target data as numpy array
    y = np.array(data_df[[target]]) #target variable is the malwares class
    y = y.flatten()

    #get x data
    df = data_df.drop(columns = [target])
    x = np.array(df)

    #get excluded class
    if exclude != None:
        excluded_indicies = list(np.where(y == exclude)[0])  #get indicies of excluded class
        
        # get arrays of excluded class
        x_e = np.array([x[i] for i in excluded_indicies])
        y_e = np.array([y[i] for i in excluded_indicies])
        
        #remove excluded class from original data
        x = np.delete(x,excluded_indicies, axis = 0)
        y = np.delete(y,excluded_indicies, axis = 0)
        

    #split into 45% train, 45% test and 10% val data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.45, random_state=seed, shuffle = True, stratify = y)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.182, random_state=seed, shuffle = True, stratify = y_train)
    
    #np.save('C:\\Users\\jackw\\Desktop\\x',x_test)
    #np.save('C:\\Users\\jackw\\Desktop\\y',y_test)
    
    #add excluded class to test data
    if exclude != None:
        x_test = np.concatenate((x_test, x_e), axis=0)
        y_test = np.concatenate((y_test, y_e), axis=0)

    
    #return datasets
    return x_train, y_train, x_test, y_test, x_val, y_val
    #return x_train, y_train, x_val, y_val


