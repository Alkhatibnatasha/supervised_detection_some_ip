from torch.utils.data import DataLoader
from lstm.model import LSTM
from lstm.dataset import SOMEIPDataset
from lstm.dataset.utils import save_parameters
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import tqdm

import os

class Trainer():
    def __init__(self, options):
        self.device = options["device"]
        self.model_dir = options["model_dir"]
        self.model_path = options["model_path"]
        self.path_train = options["path_train"] 
        self.path_valid = options["path_valid"]
        self.output_path = options["output_dir"]
        self.window_size = options["window_size"]
        self.stride = options["stride"]
        self.features = options["features"]
        self.num_workers = options["num_workers"]
        self.criterion = options["criterion"]
        self.batch_size = options["batch_size"]
        self.lr = options["lr"]
        self.adam_beta1 = options["adam_beta1"]
        self.adam_beta2 = options["adam_beta2"]
        self.adam_eps = options["adam_eps"]
        self.epochs = options["epochs"]
        self.n_epochs_stop = options["n_epochs_stop"]
        self.balanced=options["balanced"]
        self.subsample_size=options["subsample_size"]
        self.ratio=options["ratio"]
        self.cwd_orig=options["cwd_orig"]
        
        self.verbose=2000

        print("Save options parameters")
        save_parameters(options, self.model_dir + "parameters.txt")
        self.writer = SummaryWriter(self.model_dir+'/torch_logs')
        
    def train_epoch(self,model,device,dataloader,criterion,optimizer,verbose):
        
        model.train()
        train_loss=[]

        for i,(x,y) in enumerate(dataloader):

            x=x.to(device)
            
            y=y.reshape(y.shape[0],1).to(device)
            
            y_hat=model(x)
            
           

            loss=criterion(y_hat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Print batch loss
            if verbose > 0:
                if (i+1) % verbose == 0:
                    print('\t partial train loss (single batch): %f'%(loss.data))

            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)
    
    def machin(self):
        print("machin")
    
    def test_epoch(self,model,device,dataloader,criterion):
        #Set evaluation mode for convae
        valid_loss=[]
        model.eval()
        with torch.no_grad(): #No need to track the gradients
            # Define the lists to store the outputs for each batch
            for x,y in dataloader:
                #Move tensor to the proper device
                x=x.to(device)
                y=y.reshape(y.shape[0],1).to(device)
                y_hat=model(x)

                #Evaluate global loss
                val_loss=criterion(y_hat,y)
                
                valid_loss.append(val_loss.detach().cpu().numpy())

        return np.mean(valid_loss)
    

        
    def start_iteration(self,model,device,train_dataloader,valid_dataloader,criterion,optimizer,verbose):
        diz_loss={'train_loss':[],'val_loss':[]}

        best_valid_loss = np.inf
        num_steps_wo_improvement = 0 #patience counter

        for epoch in range(self.epochs):
            train_loss=self.train_epoch(model,device,train_dataloader,criterion,optimizer,2000)
            val_loss=self.test_epoch(model,device,valid_dataloader,criterion)
            print("\n EPOCH {}/{} \t train loss {} \t val loss {}".format(epoch+1,self.epochs,train_loss,val_loss))
            diz_loss["train_loss"].append(train_loss)
            diz_loss["val_loss"].append(val_loss)

            self.writer.add_scalar('training loss',
            train_loss,
            epoch + 1)


            self.writer.add_scalar('validation loss',
                val_loss,
                epoch + 1)

            if val_loss <= best_valid_loss:
                best_valid_loss=val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict()
                    }, self.model_path)
            else:
                num_steps_wo_improvement+=1

            if(num_steps_wo_improvement==self.n_epochs_stop):
                print("Early Stopping")
                break
                
                
        
    def train(self):

        print("\nLoading Train Dataset")
        
        train_dataset = SOMEIPDataset(self.path_train,window_size=self.window_size, s=self.stride, balanced=self.balanced, subsample_size=self.subsample_size,train_state = 'train',ratio=self.ratio)
        
        
        print("\nLoading valid Dataset")
        
        os.chdir(self.cwd_orig)
        

        valid_dataset = SOMEIPDataset(self.path_valid,window_size=self.window_size, s=self.stride, balanced=self.balanced, subsample_size=self.subsample_size,train_state = 'test')
        
        print("Creating Dataloader")
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size,shuffle = True, num_workers=self.num_workers)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size,shuffle = True,  num_workers=self.num_workers)
        
        self.machin()
        
        
        print("Building LSTM Classifier model")
        os.chdir(self.cwd_orig)
        self.model = LSTM(**self.features).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.adam_beta1, self.adam_beta2), eps=self.adam_eps)
        self.start_iteration(self.model,self.device,self.train_data_loader,self.valid_data_loader,
                      self.criterion,self.optimizer,self.verbose)
                      
                   
        
        
        
                             
            
        
        
