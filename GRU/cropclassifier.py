# --------------- Model init ---------
# Imports
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Classes
class GRUModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=2):
        super(GRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=2, batch_first=True) # GRU number of layer 2
        #self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True) # GRU number of layer 1
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, batchX):
    
        out, _ = self.gru(batchX)
        return self.fc1(out[:, -1, :])
