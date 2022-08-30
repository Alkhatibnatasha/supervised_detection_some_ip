# Third Party
import torch
import torch.nn as nn


############
# COMPONENTS
############



class LSTM(nn.Module):
    def __init__(self, input_dim, out_dim=1, h_dims=[], h_activ=nn.Tanh(),
                 out_activ=nn.Sigmoid()):
        super(LSTM, self).__init__()
        
        layer_dims = [input_dim] + h_dims + [out_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        
        for index in range(self.num_layers-1):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True
            )
            self.layers.append(layer)

        self.h_activ, self.out_activ = h_activ, out_activ
        
        self.linear=nn.Linear(layer_dims[-2],out_dim)
        
    def forward(self, x):
        
 
        for index, layer in enumerate(self.layers):
          
            x, (h_n, c_n) = layer(x)
            x = self.h_activ(x)
            
        x=self.linear(x[:,-1])
        x=self.out_activ(x)
                

        return x
    
