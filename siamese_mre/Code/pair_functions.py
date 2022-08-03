# -*- coding: utf-8 -*-
"""
Pair Generation Functions

Created on Tue Aug  2 21:20:52 2022

@author: jackw
"""

import numpy as np
from math import factorial
from itertools import combinations
import random



#find the number of possible combinations of n objects, given sample size r
def num_combinations(n,r):
    '''
    Function applying the formula:
        nCr = n!/r!(n-r)!
    
    To calculate number of combinations of a set
    
    Parameters
    ----------
    n : Int
        total number of objects in the set.
    r : Int
        number of choosing objects from the set.

    Returns
    -------
    Int
        The number of possible combinations of set.

    '''
    
    combination = factorial(n)/(factorial(r)*factorial(n-r)) #calculate number of combinations
    return int(combination)  #return number of combinations



#--------------------------- Get Pairs From Dataset ---------------------------

def get_pairs(x_data, y_data, batch_size = 30000, seed = 3902423, log_interval = 1000):

    random.seed(seed)  #set random seed for pair selection

    classes = np.unique(y_data)  #get list of classes in dataset
    n_classes = len(classes)  #find numbe of classes in dataset
    n_combinations = num_combinations(n_classes,2)
    
    #find number of pairs required for each type
    n_similar_pairs = int(batch_size/2)
    n_dissimilar_pairs = int(batch_size/2)
    
    
    n_similar_per_class = n_similar_pairs/n_classes
    n_per_type = n_dissimilar_pairs/n_combinations
    
    pairs = []
    pair_indicies = []
    labels = []
    
    #generate similar pairs 
    
    for c in classes:
        print('finding similar pairs for class ' + str(c))
        
        pair_count = 0 
        unique_pairs = num_combinations(np.count_nonzero(y_data==c),2) 
        
        while pair_count < n_similar_per_class:
            
            i1 = random.choice(np.argwhere(y_data==c))[0]
            i2  = random.choice(np.argwhere(y_data==c))[0]
            
            
            #get two random samples belonging to class
            x1 = x_data[i1]
            x2 = x_data[i2]
            
            pair = (x1,x2)  #form pair using random samples
            
            if ((not (i1,i2) in pair_indicies) and (not (i2,i1) in pair_indicies) and i1 != i2) or unique_pairs < n_similar_per_class:  #check same sample not chosen twice, allow resampling if unique pairs not possible
                pair_indicies.append((i1,i2))
                pairs.append(pair)  #store pair
                labels.append(0)
                pair_count += 1  #increment pair count
                
                if pair_count % log_interval == 0:
                    print('found ' + str(pair_count) + ' pairs')
            
        
    #generate dissimilar pairs
    print('finding dissimilar pairs')
    for pair_type in combinations(classes,2):
        print(pair_type)
        
        pair_count = 0
        
        #get classes to form pairs of
        c1 = pair_type[0]
        c2 = pair_type[1]
        
        unique_pairs = np.count_nonzero(y_data==c1) * np.count_nonzero(y_data==c2)
        
        while pair_count < n_per_type:
            
            i1 = random.choice(np.argwhere(y_data==c1))[0]
            i2  = random.choice(np.argwhere(y_data==c2))[0]
            
            
            #get two random samples belonging to class
            x1 = x_data[i1]
            x2 = x_data[i2]
            
            #get two random samples belonging to class
            #x1 = x_data[random.choice(np.argwhere(y_data==c1))[0]]
            #x2 = x_data[random.choice(np.argwhere(y_data==c2))[0]]
            
            pair = (x1,x2)  #form pair using random samples
    
            if (not (i1,i2) in pair_indicies)  or unique_pairs < n_per_type:  #check same sample not chosen twice, allow resampling if unique pairs not possible
                pair_indicies.append((i1,i2)) 
                pairs.append(pair)  #store pair
                labels.append(1)
                pair_count += 1  #increment pair count
                
                if pair_count % log_interval == 0:
                    print('found ' + str(pair_count) + ' pairs')
            
            
    left_samples = np.array([sample[0] for sample in pairs])
    right_samples  = np.array([sample[1] for sample in pairs])
    y = np.array(labels)
    
    
    return left_samples, right_samples, y
    