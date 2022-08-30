import torch
import numpy as np
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#POSITIONAL ENCODING
def get_angles(pos, i, d_model):
    
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                              np.arange(d_model)[np.newaxis, :],
                              d_model)
  
  # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return torch.tensor(pos_encoding, dtype=torch.float32).to(device) #Torch Tensor float 32


def create_look_ahead_mask(size):
    
    mask = torch.triu(torch.ones((size,size), dtype=torch.float32), diagonal=1)
    return mask

#MHA
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_input= None):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        if d_input is None:
            d_input = d_model


        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_input, d_model, bias=False)
        self.wk = nn.Linear(d_input, d_model, bias=False)
        self.wv = nn.Linear(d_input, d_model, bias=False)

        self.dense = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, q, k, v, mask): 
            matmul_qk = torch.matmul(q, k.transpose(-2, -1))   # (..., seq_len_q, seq_len_k)

              # scale matmul_qk

            scaled_attention_logits = matmul_qk / np.sqrt(self.depth)

              # add the mask to the scaled tensor.
            if mask is not None:
            # (..., 1, 1, seq_len)  
               scaled_attention_logits += (mask * -1e9)  


              # softmax is normalized on the last axis (seq_len_k) so that the scores
              # add up to 1.
            attention_weights = nn.Softmax(dim=-1)(scaled_attention_logits)  # (..., seq_len_q, seq_len_k)

            output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

            return output, attention_weights
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """

        return x.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
    
    def forward(self, q, k, v, mask):

        batch_size = q.shape[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth) --- F
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)


        #Heads Concatination
        concat_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)


        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights




#PointWise Layer
class CNN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.k1convL1 = nn.Linear(d_model, hidden_dim)
        self.k1convL2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = self.k1convL2(x)
        return x
    
    
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate, d_input = None):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, d_input)
        self.cnn = CNN(d_model, dff)

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
    
    def forward(self, x, mask):
        
        # Multi-head attention 
        attn_output, attention_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        # Layer norm after adding the residual connection 
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        # Feed forward 
        cnn_output = self.cnn(out1)  # (batch_size, input_seq_len, d_model)
        cnn_output = self.dropout2(cnn_output)
        #Second layer norm after adding residual connection 
        out2 = self.layernorm2(out1 + cnn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attention_weights
    
    
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, d_input, num_heads, dff, maximum_position_encoding, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Linear(d_input, d_model)

        self.enc_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.enc_layers.append(EncoderLayer(d_model, num_heads, dff, rate))
            
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)
        
        
    def forward(self, x, mask):
        
        seq_len = x.shape[-2]
        attention_weights = {}
        
        x = self.embedding(x) # Transform to (batch_size, input_seq_length, d_model)
        
        x += self.pos_encoding[:, :seq_len, :]

        for i in range(self.num_layers):
            x, att = self.enc_layers[i](x, mask)
            attention_weights['layer{}_'.format(i+1)] = att

        return x, attention_weights  # (batch_size, input_seq_len, d_model)
    
    
class TransformerClassifier(nn.Module):
    def __init__(self, num_layers, d_model, d_input, num_heads, dff, maximum_position_encoding, rate=0.1):
        super().__init__()
        
        self.encoder = Encoder(num_layers, d_model, d_input, num_heads, dff,
                               maximum_position_encoding, rate=0.1)
        
        self.dense = nn.Linear(d_model, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x, mask = None):
        
        x, attention_weights = self.encoder(x, mask)
        
        
        x = torch.mean(x, dim=1)
        x = self.dense(x)
        x = self.activation(x)
        
        return x, attention_weights
    
    
    