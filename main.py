import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn

import torch
from torch import nn
import torch_geometric
from torch_geometric import GCNConv
from torch_geometric import GATConv
from torch_geometric.data import Data

# Go from RNN to GRU

# Get the RNN have a longer sequence
# Maksure to handle vanishing gradient problems
# Can use gradient clipping incase
#Use Attention Mechanism

# Make sure you optimize like you are a maniac
# Use the best optimizers (ADAM and RMSProp)
# leverage the cyclic learning rates or reducelronplateau learning rates


class AdvancedGCRNN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_size, rnn_layers):
        super(AdvancedGCRNN, self).__init__()

        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)

        self.gat = GATConv(32, 32)

        self.rnn = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=rnn_layers, 
                           batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        
        x = self.gat(x, edge_index)
        x = torch.relu(x)
        
        x = x.unsqueeze(1)
        rnn_out, _ = self.rnn(x)
        
        rnn_out = self.dropout(rnn_out)
        rnn_out = self.batch_norm(rnn_out[:, -1, :])
        
        out = self.fc(rnn_out)
        return out
        

class Visualizer:
    def __init__(self, csv_file, graph_type, x_data, y_data, x_label, y_label, title):
        self.csv_file   = csv_file
        self.graph_type = graph_type
        self.x_label    = x_label
        self.y_label    = y_label
        self.title      = title
        
    def DataVisualizer(self):
        df = pd.read_csv(self.csv_file)
        
        if self.x_label in df.columns:
            x_data = df[self.x_label]
        else:
            x_data = None
            print(f"Warning: '{self.x_label}' column not found in CSV.")
        
        if self.y_label in df.columns:
            y_data = df[self.y_label]
        else:
            y_data = None
            print(f"Warning: '{self.y_label}' column not found in CSV.")
        
        if self.graph_type.strip().lower() == 'histogram':
            if y_data is not None:
                plt.hist(y_data, bins=30, alpha=0.75, color='blue')
                plt.title(self.title)
                plt.xlabel(self.x_label)
                plt.ylabel(self.y_label)
                plt.show()
            else:
                print("Error: Y-data not available for histogram.")
        
        elif self.graph_type.strip().lower() == 'plot':
            if y_data is not None:
                plt.plot(y_data, color='blue')
                plt.title(self.title)
                plt.xlabel(self.x_label)
                plt.ylabel(self.y_label)
                plt.show()
            else:
                print("Error: Y-data not available for plot line.")

        elif self.graph_type.strip().lower() == 'scatter':
            if y_data is not None:
                plt.scatter(y_data, color='red')
                plt.title(self.title)
                plt.xlabel(self.x_label)
                plt.ylabel(self.y_label)
                plt.show()
            else:
                print("Error: Y-data not available for scatter plot.")


        else:
            print(f"Graph type {self.graph_type} not supported.")
            
    def WebVisulizerData(self):
        pass

visualizer = Visualizer(
    csv_file="random_data.csv", 
    graph_type="plot", 
    x_data=None, 
    y_data="Score", 
    x_label="Value", 
    y_label="Score", 
    title="Histogram of Scores"
)
visualizer.DataVisualizer()
