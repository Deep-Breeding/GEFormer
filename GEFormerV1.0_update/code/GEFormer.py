import torch
import torch.nn as nn
from tools.gMLP import gMLPVision
from tools.CrossGated_MLP import CrossGatedMLP
from tools.TimeFeature_Block import TimeFeatureBlock

def exists(val):
    return val is not None
class GEFormer(nn.Module):
    def __init__(self, args, snp_len, env_days, trial=None):
        super(GEFormer, self).__init__()

        if exists(trial):
            dout = trial.suggest_float("dropout", args.dropout_1, args.dropout_2)
            dep = trial.suggest_int("depth", args.depth_1, args.depth_2)
            L1 = trial.suggest_int("neurons1", args.neurons1_1, args.neurons1_2)
            L2 = trial.suggest_int("neurons2", args.neurons2_1, args.neurons2_2)
        else:
            dout = args.dropout
            dep = args.depth
            L1 = args.neurons1
            L2 = args.neurons2

        self.gmlp = gMLPVision(image_size=(snp_len, 1),
                               patch_size=(snp_len, 1),
                               num_classes=126,
                               dim=126,
                               depth=dep,
                               snp_len=snp_len
                               )

        self.TimeFeatureBlock = TimeFeatureBlock(args, env_days)

        self.cgMLP = CrossGatedMLP(126)

        self.fc = nn.Sequential(
            nn.Linear(756, L1),
            nn.LeakyReLU(),
            nn.Dropout(dout),
            nn.Linear(L1, L2),
            nn.LeakyReLU(),
            nn.Dropout(dout),
            nn.Linear(L2, 1)
        )

    def forward(self, x, x1, x2):
        x3 = self.TimeFeatureBlock(x1, x2)

        x3 = x3.squeeze(1)
        x = x.transpose(0, 1)
        x = x.unsqueeze(1)
        x = x.unsqueeze(3)
        x = self.gmlp(x)

        x4 = torch.mul(x, x3)

        a = self.cgMLP(x, x3)
        b = self.cgMLP(x, x4)
        c = self.cgMLP(x3, x4)
        concatenated = torch.cat([a, b, c], dim=1)

        predict = self.fc(concatenated)
        return predict

