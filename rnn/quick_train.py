from torch.utils.data import DataLoader
from lstm.model import LSTMAutoencoder
from lstm.dataset import SOMEIPDataset
from lstm.dataset.utils import save_parameters
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import tqdm

# Standard Library
from statistics import mean

class Trainer():
    def __init__(self, options):
        self.device = options["device"]
        self.model_dir = options["model_dir"]
        self.model_path = options["model_path"]
        self.x_train_data = options["x_train_data"] 
        self.y_train_data = options["y_train_data"] 
        self.x_test_data = options["x_test_data"] 
        self.y_test_data = options["x_test_data"] 
        self.output_path = options["output_dir"]
        self.window_size = options["window_size"]
        self.stride = options["stride"]
        self.features = options["features"]
        self.num_workers = options["num_workers"]
        self.criterion = options["criterion2"]
        self.batch_size = options["batch_size"]
        self.lr = options["lr"]
        self.adam_beta1 = options["adam_beta1"]
        self.adam_beta2 = options["adam_beta2"]
        self.adam_eps = options["adam_eps"]
        self.epochs = options["epochs"]
        self.n_epochs_stop = options["n_epochs_stop"]
        
        self.verbose=2000

        print("Save options parameters")
        save_parameters(options, self.model_dir + "parameters.txt")
        self.writer = SummaryWriter(self.model_dir+'/torch_logs')
        
      
    def instantiate_model(model, train_set, encoding_dim, **kwargs):
            return model(train_set[-1].shape[-1], encoding_dim, **kwargs)
        
    def train_model(model, train_set, verbose, lr, epochs, denoise):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MSELoss(size_average=False)

    mean_losses = []
    for epoch in range(1, epochs + 1):
        model.train()

        # # Reduces learning rate every 50 epochs
        # if not epoch % 50:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr * (0.993 ** epoch)

        losses = []
        for x,_ in train_set:
            optimizer.zero_grad()

            # Forward pass
            x_prime = model(x)

            loss = criterion(x_prime, x)

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        mean_loss = mean(losses)
        mean_losses.append(mean_loss)

        if verbose:
            print(f"Epoch: {epoch}, Loss: {mean_loss}")

    return mean_losses

    def get_encodings(model, train_set):
        model.eval()
        encodings = [model.encoder(x) for x in train_set]
        return encodings
                
        
    def train(self):

        print("\nLoading Train Dataset")
        
        train_dataset = SOMEIPDataset(self.x_train_data,self.y_train_data,window_size=self.window_size, s=self.stride, train_state = 'train')
        
        print("\nLoading valid Dataset")

        valid_dataset = SOMEIPDataset(self.x_train_data,self.y_train_data,window_size=self.window_size,s= self.stride, train_state = 'valid')
        
        print("Creating Dataloader")
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size,shuffle = True, num_workers=self.num_workers)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size,shuffle = True,  num_workers=self.num_workers)
        
        
        
        print("Building LSTMAutoencoder model")
        self.lstm = LSTMAutoencoder(**self.features).to(self.device)
        self.optimizer = torch.optim.Adam(self.bilstm.parameters(), lr=self.lr, betas=(self.adam_beta1, self.adam_beta2), eps=self.adam_eps)
        
      
        losses = train_model(model, train_set, verbose, lr, epochs, denoise)
    encodings = get_encodings(model, train_set)

    return model.encoder, model.decoder, encodings, losses
       
    
    
        
                   
        
        
        
                             
            
        
        
