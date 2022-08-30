import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import os 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_files(file_path):
    with open(file_path,'rb') as file:
        m=pickle.load(file)
    return m


def stride(l, window_size=4, stride=2):
    d=[]
    seq_len=len(l)
    assert window_size<seq_len, "Get a window size smaller than sequence length"
    assert stride<seq_len, "Get a stride smaller than sequence length"
    
    for i in range(0,seq_len,stride):
        if (i+window_size>seq_len):
            break
        else:
            d.append(l[i:i+window_size])
    return d

def process(x,y,window_size,s):
    
    x=np.array(x)
    y=np.array(y)
            
    
    y=y.reshape((y.shape[0],1))
    
    #Create sequences
    sequences=stride(x, window_size=window_size, stride=s)
    transformed_sequences=[]
    transformed_sequences=np.array(sequences) #Array of sequences with converted IDS, shape (#samples, #window_size, #features per ID)
    #Label sequences 
    labels=stride(list(y), window_size=window_size, stride=s)
    sequences_labels= [int(np.any(i)) for i in labels]
    #Count # of Injections 
    NumberOfInjections=[sum(i) for i in labels]
    sequences_labels=np.array(sequences_labels)
    NumberOfInjections=np.expand_dims(NumberOfInjections, axis=1)
    NumberOfInjections=np.array(NumberOfInjections)
    
    sequences=transformed_sequences
    labels=sequences_labels
    
    return sequences,labels


def balanced_subsample(x,y,subsample_size=1.0):
    np.random.seed(2)

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

class SOMEIPDataset(Dataset):
    
    def __init__(self, path, window_size=16, s=16, balanced=True, subsample_size=1.0, train_state = 'train', ratio=0.2):
        
        
        self.path=path
        os.chdir(self.path)
        
        x_path=[]
        y_path=[]

        for file in os.listdir():
            if file.endswith('_y.pickle'):
                y_path.append(f"{file}")
            else:
                x_path.append(f"{file}")

        x_path.sort()
        y_path.sort()
        
        #Concatenate
        for i,(j,k) in enumerate(zip(x_path,y_path)):
    
            x=read_files(j)
            y=read_files(k)

            if i==0:
                
                sequences,labels = process(x,y,window_size=window_size,s=s)
            else: 

                sequences=np.concatenate((sequences, process(x,y,window_size=window_size,s=s)[0]), axis=0)
                labels=np.concatenate((labels, process(x,y,window_size=window_size,s=s)[1]), axis=0)
                
        #Minority_DownSampling
        if balanced==True:
            xs,ys=balanced_subsample(sequences,labels,subsample_size=subsample_size)
            self.xs=xs
            self.ys=ys
        else:
            self.xs=sequences
            self.ys=labels
        
        

        if train_state == 'train':
            data_train, data_valid, target_train, target_valid = train_test_split(self.xs, self.ys,
                                                                                  test_size=ratio, random_state=42) 
            self.data = data_train
            self.labels = target_train
            
            

        elif train_state == 'valid':
            data_train, data_valid, target_train, target_valid = train_test_split(self.xs, self.ys,
                                                                                  test_size=ratio, random_state=42) 
            self.data = data_valid
            self.labels = target_valid    
            
            

        elif train_state=='test':

            self.data = self.xs
            self.labels = self.ys
            
        print(len(self.data))
                
                      
            

    
    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return len(self.data)
    
    def getitem(self, index):

        x = torch.tensor(self.data[index], dtype=torch.float32)
                    
        y = torch.tensor(self.labels[index], dtype=torch.float32)

        return x, y



