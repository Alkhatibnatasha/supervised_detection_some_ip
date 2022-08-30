import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from lstm.dataset import SOMEIPDataset
from lstm.model import LSTM
import os




class Time():
    def __init__(self, options):
        
        self.device = options["device"]
        self.model_dir = options["model_dir"]
        self.model_path = options["model_path"]
        self.model_folder_name="best_model"
        self.path_test = options["path_test"] 
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
        
        
    def helper(self):
        
        test_dataset = SOMEIPDataset(self.path_test,window_size=self.window_size, s=self.stride, balanced=self.balanced, subsample_size=self.subsample_size,train_state = 'test',ratio=self.ratio)
         
        self.test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size,shuffle = True,  num_workers=self.num_workers)
        
        os.chdir(self.cwd_orig)
        
        for x,_ in self.test_data_loader:
            print(type(x))
            print(x.shape)       
            x=x.to(self.device) 
           
            break 
            
        return torch.unsqueeze(x[0],0)

        

    def calculate(self):
        
        model = LSTM(**self.features).to(self.device)
        #checkpoint=torch.load(self.model_path,map_location=self.device)
        #model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        
        
        
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings=np.zeros((repetitions,1))
        dummy1=self.helper()
        print(dummy1.shape)
        dummy1=dummy1.to(self.device)
        

        #GPU-WARM-UP
        for _ in range(10):
            _ = model(dummy1)
        
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy1)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print("Mean is: " + str(mean_syn))
        print("Std is: " + str(std_syn))
