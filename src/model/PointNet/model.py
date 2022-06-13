import torch.nn as nn
import torch.nn.functional as F
import torch
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv1d(128, 128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            #             nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Linear(128, 128)

        )   
        # self.layer = nn.Sequential(nn.Linear(256,128),nn.ReLU())
        # self.layer_norm = F.normalize(x, dim = 0)


    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        # logits = torch.cat((logits,x),1)
        # logits = self.layer(logits)
        # out = self.layer_norm(logits)
        out = F.normalize(logits, dim = 0)
        return out