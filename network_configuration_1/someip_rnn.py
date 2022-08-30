import sys
sys.path.append("../")
sys.path.append("../../")

import os

import argparse
import torch

from rnn import Trainer,Predictor
from rnn.dataset.utils import seed_everything
import torch.nn as nn 


cwd_orig=os.getcwd()

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
options['model'] = "rnn"
#checkproblem (../output/config1/)
options["output_dir"] = "../output/config1/"
options["model_dir"] = options["output_dir"] +options['model']+"/"
options["model_path"] = options["model_dir"]+"/best_model.pt"
options["path_train"] = options["output_dir"] + "data/train_missing_request"
options["path_valid"] = options["output_dir"] + "data/valid_missing_request"
options["path_test"] = options["output_dir"] + "data/test_missing_request"
options["cwd_orig"]=cwd_orig

#Sequence
options["window_size"] = 128
options["features"] = {"input_dim":58, "out_dim":1, "h_dims":[int(options["window_size"]/8),int(options["window_size"]/16)], "h_activ":nn.Tanh(),"out_activ":nn.Sigmoid()}
options["stride"] = 1

# model

options["epochs"] = 200
options["n_epochs_stop"] = 10
options["batch_size"] = 32


options["num_workers"] = 0
options["lr"] = 1e-4
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_eps"] = 1e-9
options["criterion"]=nn.BCELoss()

options["balanced"]=True
options["subsample_size"]=1.0
options["ratio"]=0.2

seed_everything(seed=1234)


if not os.path.exists(options['model_dir']):
    os.makedirs(options['model_dir'], exist_ok=True)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')
    
    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')

    args = parser.parse_args()
    print("arguments", args)
    
    if args.mode == 'train':
        print(options)
        Trainer(options).train()
        
    elif args.mode == 'predict':
        Predictor(options).predict()
