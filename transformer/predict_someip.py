from torch.utils.data import DataLoader
from transformer.model import TransformerClassifier
from transformer.dataset import SOMEIPDataset
from transformer.dataset.utils import save_parameters
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import tqdm
import json

class Predictor():
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

        
    
    

    def test(self,model,device,dataloader,th):

        model.eval()
        with torch.no_grad(): #No need to track the gradients
            # Define the lists to store the outputs for each batch
            output_predicted=[]
            ground_truth=[]

            for x,y in dataloader:
                
        
                x=x.to(device) 
                y=y.reshape(y.shape[0],1).to(device)
                y_hat,_=model(x)
                
                
                #loss
     
                predicted= ((y_hat > th)).type(torch.float32) 
  
                output_predicted.append(predicted)
                ground_truth.append(y.cpu())

            output_predicted=torch.cat(output_predicted)
            ground_truth=torch.cat(ground_truth)

        return output_predicted.data,ground_truth.data

   
        
        
    def predict(self):
        
        self.model = TransformerClassifier(**self.features).to(self.device)
        checkpoint=torch.load(self.model_path,map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('model_path:{}'.format(self.model_path))
        
        print("\nLoading test Dataset")
        
        test_dataset = SOMEIPDataset(self.path_test,window_size=self.window_size, s=self.stride, balanced=self.balanced, subsample_size=self.subsample_size,train_state = 'test',ratio=self.ratio)
        
        print("Creating Dataloader")
    
        self.test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size,shuffle = True,  num_workers=self.num_workers)
        
        os.chdir(self.cwd_orig)
      
        
        print("Check performance")
        model_params=dict()
        mode=np.arange(0.1,1,0.1)
        f1_scores=[]
        recalls=[]
        precisions=[]
        for i in mode: 
            print(i)
            pred,lab= self.test(self.model,self.device,self.test_data_loader,i)             
            pred = pred.cpu()
            lab = lab.cpu()
            acc = balanced_accuracy_score(lab, np.round(pred))
            conf_matrix = confusion_matrix(lab, np.round(pred))
            f1 = f1_score(lab, np.round(pred))
            f1_scores.append(f1)
            recall=recall_score(lab,np.round(pred))
            recalls.append(recall)
            precision=precision_score(lab,np.round(pred))  
            precisions.append(precision)
            model_params['acc'+str(i)] = acc
            model_params['f1'+str(i)] = f1
            model_params['recall'+str(i)]=recall
            model_params['precision'+str(i)]=precision
            model_params['conf_matrix'+str(i)] = str(conf_matrix)
                         
        model_params['model_name'] = self.model_folder_name
        config = json.dumps(model_params)
        max_f1=max(f1_scores)
        max_index=f1_scores.index(max_f1)
        corresp_recall=recalls[max_index]
        corresp_precision=precisions[max_index]
        print(f"F1: {max_f1}")
        print(f"Recall: {corresp_recall}")
        print(f"Precision: {corresp_precision}")


        f = open("{}/{}.json".format(self.model_dir,self.model_folder_name),"w")
        f.write(config)
        f.close()
            
            
            
        
        
        
        
        
        
        
        