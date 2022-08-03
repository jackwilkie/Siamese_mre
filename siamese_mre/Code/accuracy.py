# -*- coding: utf-8 -*-
"""
Evaluate oneshot accuracy of network

Created on Tue Aug  2 21:01:30 2022

@author: jackw
"""

#import required libraries
import numpy as np
import torch as T
import random
from custom_pytorch_functions import euclidean_distance

'''
def Evaluate_Oneshot_Accuracy(model, x_test, y_test, j = 5, metric = euclidean_distance, eval_batch_size = 30000, rnd_seed = 382948234, log_interval=1000, verbose = True, device = 'cpu'):
    
    #initalise variables
    random.seed(246546568)  #initalise seed for selecting random samples
    
    
    n_classes = len(np.unique(y_test))  #get number of unique classes
    
    #excluded_classes = list(set(np.unique(y_test)) - set(np.unique(y_train)))  #excluded classes are those featured in test data but not trained on 
    n_per_class = int(eval_batch_size/n_classes)  #caluclate number of samples to evaluate per class
    
    excluded_classes =[]
    
    n_correct = 0  #initalise correct prediction counter
    count = 0   #initalise 
    
    cm = np.zeros((n_classes,n_classes))  #initalise model confusion matric
    
    if log_interval < 0:
        raise ValueError('Invalid Log Rate')
    
    else:
        log_interval = int(log_interval)
        
    #convert data to tensors
    #x_train_data = T.tensor(x_train, dtype=T.float32).to(device)
    x_test_data = T.tensor(x_test, dtype=T.float32).to(device)
    
    
    #if len(excluded_classes) > 0:
    #    x_comp_data = T.tensor(x_e_comp, dtype=T.float32).to(device)
    
    
    model.eval()
    
    with T.no_grad():
        
        #get data embeddings
       # train_embeddings = model.feed(x_train_data)  #get train embeddings
        test_embeddings = model.feed(x_test_data)  #get test embeddings
        
        #return train_embeddings, test_embeddings
        
       # if len(excluded_classes) > 0:
       #     comp_embeddings = model.feed(x_comp_data)
        
        #perform model evaluation
        for c in range(n_classes):  #iterate over classes
            
            class_correct = 0
            class_count = 0
        
            #report current class
            if verbose:
                print('predicting class ' + str(c))
            
              
            #generate all predictions for current class
            for i in range(n_per_class):
              
                #periodically report evaluation progress
                if verbose and log_interval != 0:          
                    if i % log_interval == 0 and i > 0:
                        print('predicted ' + str(i) + ' samples') 
                        print('class accuracy = ' + str((class_correct*100)/class_count))
              
              
                #generate test prediction
                votes = np.zeros(j)   #initalise number of class votes for pedictions    
                #print(random.choice(np.argwhere(y_test==c))[0])
                x1 = test_embeddings[random.choice(np.argwhere(y_test==c))[0]]#get random instance of test class
                #print(x1)
                
                for pair in range(j):  #repeat prediction j times
                         
                    pairs_dist = np.zeros(n_classes)
                      
                    for c2 in range(n_classes):  #iterate over classes again to generate comparison pair for each class
          
                          
                        if not c2 in excluded_classes:
                            #x2 = train_embeddings[random.choice(np.argwhere(y_train==c2))[0]] #get random instance of test class
                            x2 = test_embeddings[random.choice(np.argwhere(y_test==c2))[0]]
                        #else:
                        #    x2 = comp_embeddings[random.choice(np.argwhere(y_e_comp==c2))[0]]
                             
                            
                        #calculate and store pairwise distance
                        dist = metric(x1, x2)  #find euclidean distance between samples
                        pairs_dist[c2] = dist  #store distance for pair
                  
                    #print(x1)
                    #print(x2)
                    #print(pairs_dist)
                    votes[np.argmin(pairs_dist)] += 1  #vote for class with loweset distance to test sample
    
                
                if np.argmax(votes) == c:  #check for correct classification 
                    n_correct += 1  #increment counter if correct
                    class_correct += 1
                    
                cm[c,np.argmax(votes)] += 1  #update confusion matric
                count += 1  #update test counter
                class_count +=1
            
    return cm  #return confusion matrix of model predictions


'''


def Evaluate_Oneshot_Accuracy(model, x_train, y_train, x_test, y_test, j = 5, metric = euclidean_distance, eval_batch_size = 30000, rnd_seed = 382948234, log_interval=1000, verbose = True, device = 'cpu'):
    
    #initalise variables
    random.seed(rnd_seed)  #initalise seed for selecting random samples
    
    n_classes = len(np.unique(y_test))  #get number of unique classes
    excluded_classes = list(set(np.unique(y_test)) - set(np.unique(y_train)))  #excluded classes are those featured in test data but not trained on 
    n_per_class = int(eval_batch_size/n_classes)  #caluclate number of samples to evaluate per class
    
    i_excluded_comp = []
    y_excluded_comp = []
    
    if len(excluded_classes) > 0:
        for c in excluded_classes:
            exluded_indicies = list(np.where(y_test == c)[0])  #get indicies of excluded class
            
            
            comp_indices = random.sample(exluded_indicies,int(0.5*len(exluded_indicies)))  #get half of class for testing
            
            i_excluded_comp.extend(comp_indices)
            
            
            for i in range(len(i_excluded_comp)):
                y_excluded_comp.append(c)
            
            
        x_e_comp = np.array([x_test[x] for x in i_excluded_comp])
        y_e_comp = np.array(y_excluded_comp)
        
        x_test = np.delete(x_test,i_excluded_comp, axis = 0)
        y_test = np.delete(y_test,i_excluded_comp, axis = 0)
    
    
    n_correct = 0  #initalise correct prediction counter
    count = 0   #initalise 
    
    cm = np.zeros((n_classes,n_classes))  #initalise model confusion matric
    
    if log_interval < 0:
        raise ValueError('Invalid Log Rate')
    
    else:
        log_interval = int(log_interval)
        
    #convert data to tensors
    x_train_data = T.tensor(x_train, dtype=T.float32).to(device)
    x_test_data = T.tensor(x_test, dtype=T.float32).to(device)
    
    
    if len(excluded_classes) > 0:
        x_comp_data = T.tensor(x_e_comp, dtype=T.float32).to(device)
    
    
    model.eval()
    
    with T.no_grad():
        
        #get data embeddings
        train_embeddings = model.feed(x_train_data)  #get train embeddings
        test_embeddings = model.feed(x_test_data)  #get test embeddings
        
        #return train_embeddings, test_embeddings
        
        if len(excluded_classes) > 0:
            comp_embeddings = model.feed(x_comp_data)
        
        #perform model evaluation
        for c in range(n_classes):  #iterate over classes
            
            class_correct = 0
            class_count = 0
        
            #report current class
            if verbose:
                print('predicting class ' + str(c))
            
              
            #generate all predictions for current class
            for i in range(n_per_class):
              
                #periodically report evaluation progress
                if verbose and log_interval != 0:          
                    if i % log_interval == 0 and i > 0:
                        print('predicted ' + str(i) + ' samples') 
                        print('class accuracy = ' + str((class_correct*100)/class_count))
              
              
                #generate test prediction
                votes = np.zeros(j)   #initalise number of class votes for pedictions    
                #print(random.choice(np.argwhere(y_test==c))[0])
                x1 = test_embeddings[random.choice(np.argwhere(y_test==c))[0]]#get random instance of test class
                #print(x1)
                
                for pair in range(j):  #repeat prediction j times
                         
                    pairs_dist = np.zeros(n_classes)
                      
                    for c2 in range(n_classes):  #iterate over classes again to generate comparison pair for each class
          
                          
                        if not c2 in excluded_classes:
                            x2 = train_embeddings[random.choice(np.argwhere(y_train==c2))[0]] #get random instance of test class
                            #x2 = test_embeddings[random.choice(np.argwhere(y_test==c2))[0]]
                        else:
                            x2 = comp_embeddings[random.choice(np.argwhere(y_e_comp==c2))[0]]
                             
                            
                        #calculate and store pairwise distance
                        dist = metric(x1, x2)  #find euclidean distance between samples
                        pairs_dist[c2] = dist  #store distance for pair
                  
                    #print(x1)
                    #print(x2)
                    #print(pairs_dist)
                    votes[np.argmin(pairs_dist)] += 1  #vote for class with loweset distance to test sample
    
                
                if np.argmax(votes) == c:  #check for correct classification 
                    n_correct += 1  #increment counter if correct
                    class_correct += 1
                    
                cm[c,np.argmax(votes)] += 1  #update confusion matric
                count += 1  #update test counter
                class_count +=1
            
    return cm  #return confusion matrix of model predictions
