import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RowCNN(nn.Module):
    def __init__(self, num_classes=8, split_size=100):
        super(RowCNN, self).__init__()
        self.window_sizes = [3, 4, 5]
        self.n_filters = 100
        self.num_classes = num_classes
        self.split_size = split_size
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.n_filters, [window_size, self.split_size], padding=(window_size - 1, 0))
            for window_size in self.window_sizes
        ])

        self.linear = nn.Linear(self.n_filters * len(self.window_sizes), self.num_classes)
        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xs = []
        # x = torch.unsqueeze(x, 1) # Might have to do it - [B, CH, R, C]
        # TODO : Very unsure, lot of changes to do
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, R_, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, R_]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2) 
        x = x.view(x.size(0), -1)  
        logits = self.linear(x)
        logits = self.sigmoid(logits)
        return logits

class BiRNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=100, num_layers=2, num_classes=8):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):    
        # Aren't the weights of the hidden units, but just placeholders to store the hidden activation for the current run
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, (h0, c0) = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.sigmoid(self.fc(out[:, -1, :]))
        return out

#class KgRNNCVAE(nn.Module):
#    def __init__(self, )