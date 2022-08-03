# -*- coding: utf-8 -*-
"""
Pytorch Functions

Created on Tue Aug  2 20:57:15 2022

@author: jackw
"""

#import required libraries
import torch as T
import time
import torch.nn.functional as F


#------------------------- Metric and Loss Functions --------------------------

#Find Euclidean Distance Between Tensors
def euclidean_distance(a,b):
    return F.pairwise_distance(a, b)


#contrastice loss function in pytorch
class ContrastiveLoss(T.nn.Module):
  def __init__(self, m=1.0, metric = euclidean_distance):
    super(ContrastiveLoss, self).__init__() 
    self.m = m  #define margin
    self.metric = metric  #define distance function

  def forward(self, x1, x2, target):
    #x1, x2 are input samples
    #target is 0 is samples similar, otherwise target is 1

    dist = self.metric(x1, x2)  #find euclidean distance between samples
    
    loss = T.mean((1-target) * T.pow(dist, 2) + (target) * T.pow(T.clamp(self.m - dist, min=0.0), 2))  #calculate contrastive loss

    return loss  #return loss


#------------------------ Neural Network Initalisation ------------------------

def parse_embedding_architecture(architecture, input_size = None):
    #initialise layer lists
    layers = T.nn.ModuleList([])
    
    #process architecture string
    architecture = ''.join(architecture.split())  #remove whitespace from string
    architecture = architecture.split(':')  #sperate string using layer delimiters 
    
    
    layer_types = []  #store types of layers in network
    
    for i, layer in enumerate(architecture):  #iterate over network layers

        layer_type = layer[:layer.find('(')]  #get type of layer to add to model
        layer_type = layer_type.lower()   #convert to upper case to avoid case sensitivity
        
        layer_val = layer[layer.find('(')+1:layer.find(')')]  #get value of layer to add to model
        
        
        #get integer value for layer
        try:
            layer_val = float(layer_val)  #cast string to float
            
            #cast to int if possible
            if layer_val%1 == 0:
                layer_val = int(layer_val)
            
        except ValueError:  #catch casting errors
            print('Invalid layer value given in architecture: cannot cast to int')
            
            
        #set input size on first iteration
        if i == 0:
            if input_size is not None:  #input size given as function argument
                linear_vals = [input_size]
                
            elif layer_type == 'in' or layer_type == 'input':  #input
                linear_vals = [layer_val]
                
            else:  #no input size given, raise error
                raise ValueError('No Input Diemensions Given')



        #find required layer type
        if layer_type == 'dense' or layer_type == 'de':  #add dense layer
            linear_vals.append(int(layer_val))
            layers.append(T.nn.Linear(linear_vals[-2], linear_vals[-1]))
            layer_types.append('de')
            

        elif layer_type == 'dr' or layer_type == 'dropout':  #add dropout layer
            layers.append(T.nn.Dropout(layer_val))
            layer_types.append('dr')
            
        else:  #invalid layer type
            raise ValueError('Invalid layer type specified in architecture')
    
    
    return layers, layer_types  #return architecture


def architecture_forward_pass(x, layers, layer_types):
    for i, l in enumerate(layers):
        if layer_types[i] == 'de':
            x = T.relu(l(x))
    
        elif layer_types[i] == 'dr':
            x = l(x)
        
    return x


class Siamese_Dataset(T.utils.data.Dataset):
  
  def __init__(self, x1, x2, y, device = 'cpu'):
     
    #set device type
    self.device = T.device(device)
     
    #convert data to tensors
    self.x1_data = T.tensor(x1, dtype=T.float32).to(self.device)
    self.x2_data = T.tensor(x2, dtype=T.float32).to(self.device)
    self.y_data = T.tensor(y, dtype=T.int64).to(self.device) 

    #find length of dataset
    self.n = len(self.y_data)  

  
  #get dataset length
  def __len__(self):
    return self.n


  #get data pair and similarity 
  def __getitem__(self, i):

    x1 = self.x1_data[i] 
    x2 = self.x2_data[i]
    y = self.y_data[i]
    
    return (x1, x2, y)




#create siamese network class
class SiameseNet(T.nn.Module):
  def __init__(self, input_size, architecture = 'De(98):Dr(0.1):De(79):Dr(0.1):De(59):Dr(0.1):De(39):Dr(0.1):De(20)'):
    super(SiameseNet, self).__init__()  # initalise 
    
    #get embedding network from architecture
    self.layers, self.layer_types = parse_embedding_architecture(architecture, input_size)
    

  #pass of neural network for single input samples
  def feed(self, z):  
    return architecture_forward_pass(z, self.layers, self.layer_types) #perform forward pass of neural network for sample
    

  #forward pass of network feeds 2 inputs and returns embedding space representaions
  def forward(self, x1, x2):

    emb1 = self.feed(x1)
    emb2 = self.feed(x2)
    
    return emb1, emb2 


#------------ Custom Training Loop -----------------------------

def Siamese_Train(model, optimiser, loss_fn, train_dl, val_dl, epochs, ep_log_interval = 1, device = 'cpu'):
    print('Training Model')

    #coolect training and validation metrics for each epoch
    history = {}
    history['loss'] = []
    history['val_loss'] = []    
    
    start_time = time.time()
    
    #start training loop
    for epoch in range(epochs):
        
        train_loss = 0  #initalise train loss
        
        model.train()  #set model to train
        
        #--------------- train and evaluate on training dataset ---------------
        
        for batch_num, batch in enumerate(train_dl):
           
            x1,x2,y = batch   #get training batch
            
            emb1, emb2 = model(x1, x2)  #predict euclidean distances for training batch
            
            optimiser.zero_grad()  #reset gradient values
            
            loss = loss_fn(emb1, emb2, y)  #compute training loss
            
            #apply backpropogation
            loss.backward()         # compute gradients 
            optimiser.step()            # update weights
            
            train_loss += loss.item() * x1.size(0)  # multiply sample loss by batch size for batch loss
            
        train_loss = train_loss / len(train_dl.dataset)  #find per sample loss
        
        
        #-------------------- evaluate on training dataset --------------------
        model.eval()  #set model to evaluat
        val_loss = 0  #init val loss
        
        for batch_num, batch in enumerate(val_dl):
        
            x1,x2,y = batch  #get validation batch
            
            emb1, emb2 = model(x1, x2)  #predict euclidean distances for validation batch
            
            loss = loss_fn(emb1, emb2, y)  #compute validation loss
            
            val_loss += loss.item() * x1.size(0)  # multiply sample loss by batch size for batch loss
            
        val_loss = val_loss / len(val_dl.dataset)
        
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if epoch % ep_log_interval == 0 and ep_log_interval != 0:
            train_time = time.time() - start_time
            print(f"epoch = {epoch}  |  train_loss = {train_loss:.6f}  |  val_loss = {val_loss:.6f} |  training_for: {train_time:.2f}" )
    
    
    #end model training
    end_time = time.time()
    total_time = end_time - start_time  #find time to trian
    #time_per_epoch = total_time / epochs  #find time per epoch
    
    print(f'\nTraining Complete in: {total_time:.2f} seconds')
    
    return history
