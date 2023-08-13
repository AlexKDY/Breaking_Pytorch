import torch
import torch.nn as nn 
import sys 
import sys
import numpy as np
import matplotlib.pyplot as plt
from model import ImageClassifier
from utils import load_mnist
from utils import split_data
from utils import get_hidden_sizes

model_fn = "./model.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load(fn, device):
    d = torch.load(fn, map_location=device)
    
    return d['model'], d['config']

def test(model, x, y):
    model.eval()
    
    with torch.no_grad():
        y_hat = model(x)

        correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
        total_cnt = float(x.size(0))
        
        accuracy = correct_cnt / total_cnt
        print("Accuracy: %.4f" % accuracy)
        
model_dict, train_config = load(model_fn, device)

# Load MNIST test set.
x, y = load_mnist(is_train=False)
x, y = x.to(device), y.to(device)

input_size = int(x.shape[-1])
output_size = int(max(y)) + 1

model = ImageClassifier(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=get_hidden_sizes(input_size,
                                  output_size,
                                  train_config.n_layers),
    use_batch_norm=not train_config.use_dropout,
    dropout_p=train_config.dropout_p,
).to(device)

model.load_state_dict(model_dict)

test(model, x, y)
