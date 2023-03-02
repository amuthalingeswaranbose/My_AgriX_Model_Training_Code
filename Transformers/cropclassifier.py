# --------------- Model init ---------
# Imports
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Classes
# class CropClassifier(nn.Module):
class TransformerModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, num_heads=8, num_layers=4):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
                
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(self.hidden_dim, num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        

    def forward(self, batchX):
    
        x = self.embedding(batchX)
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1)
        out = self.fc(x[:, -1, :])
        return out
        
