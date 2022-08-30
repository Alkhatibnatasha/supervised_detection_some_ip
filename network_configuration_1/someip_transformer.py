import sys
sys.path.append("../")
sys.path.append("../../")

import os

import argparse
import torch

from transformer import Trainer,Predictor,Time
from transformer.dataset.utils import seed_everything
import torch.nn as nn 


cwd_orig=os.getcwd()

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
options['model'] = "transformer"
options["output_dir"] = "../output/network_configuration_1/"
options["model_dir"] = options["output_dir"] +options['model']+"/"
options["model_path"] = options["model_dir"]+"/best_model.pt"
options["path_train"] = options["output_dir"] + "data/train"
options["path_valid"] = options["output_dir"] + "data/valid"
options["path_test"] = options["output_dir"] + "data/test"
options["cwd_orig"]=cwd_orig

#Sequence
options["window_size"] = 128
options["features"]={"num_layers" : 4,"d_model": 256,'d_input' :58,"num_heads" : 4, "dff" : 512,"maximum_position_encoding" : options["window_size"],"rate" : 0.1}


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


