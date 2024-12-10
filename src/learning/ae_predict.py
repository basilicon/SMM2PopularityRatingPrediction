import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from progress.bar import Bar
import pandas as pd
import numpy as np
from subworld_autoencoder import SubworldAutoencoder, SubworldEncoder, MultiBatchDataset
from sklearn.metrics import roc_auc_score
from copy import deepcopy

# 0. Hyperparameters
lr = 1e-3
k_folds = 5
n_epochs = 10
seed = 42
loss_function = nn.MSELoss()

# 0.1. Filenames
base_encoder_fname = './models/subworld_autoencoder/sblvl-ae1.pth'
processed_levels_folder = "./data/processed_levels/"
labels_file = './data/tabular_only.csv'
temp_model_file = "./models/ae_full.pth"

# 1. Model declaration
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

class SubworldEncoderPrediction(nn.Module):
    def __init__(self, in_channels):
        super(SubworldEncoderPrediction, self).__init__()

        self.encoder = SubworldEncoder(in_channels)
        self.l1 = nn.Linear(64, 8)
        self.l2 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = self.encoder(x) # already has linearization
        x = torch.relu(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        return x

class LabeledMultiBatchDataset(MultiBatchDataset):
    def __init__(self, source_folder, label_csv_file):
        MultiBatchDataset.__init__(self, source_folder)

        df = pd.read_csv(label_csv_file)
        self.labels = torch.round(torch.from_numpy(df['likes-norm'].to_numpy()).reshape((-1, 1)).type(torch.FloatTensor)).to('cuda')

    def __getitem__(self, idx):
        return torch.from_numpy(MultiBatchDataset.__getitem__(self, idx)).unsqueeze(0), self.labels[idx // 2].unsqueeze(0)

# 2. Training


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        print("No GPU available. Exiting...")
        exit()

    torch.manual_seed(seed)
    np.random.seed(seed)

    network = SubworldEncoderPrediction(8)
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Total params: {pytorch_total_params}")
    network = None

    dataset = LabeledMultiBatchDataset(processed_levels_folder, labels_file)

    base_ae = torch.nn.DataParallel(SubworldAutoencoder(8))
    base_ae.load_state_dict(torch.load(base_encoder_fname, weights_only=True))
    base_encoder_sd = base_ae.module.encoder.state_dict()
    base_ae = None # free rest of the model

    # init model
    model = SubworldEncoderPrediction(8)
    model = nn.parallel.DataParallel(model)
    model.apply(reset_weights)
    model.module.encoder.load_state_dict(base_encoder_sd)
    model.to('cuda')

    # init optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(n_epochs):
        bar = Bar(f'Training epoch {epoch}', max=len(dataset), suffix='%(percent)d%% [%(elapsed_td)s / %(eta_td)s]')

        for data in dataset:
            inputs, targets = data

            # Hack undersampling
            if targets.item() == 1 and np.random.random() > 0.2:
                bar.next()
                continue

            inputs = inputs.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            bar.next()
        
        torch.save(model.state_dict(), temp_model_file)
        bar.finish()

    # Evaluation
    model.eval()
    c_matrix = [0, 0, 0, 0]
    auc = 0

    print("Starting evaluation.")
    with torch.no_grad():
        targets = np.zeros((len(dataset)))
        outputs = np.zeros((len(dataset)))

        i = 0
        for data in dataset:
            x, y = data
            x = x.to('cuda')
            out = model(x).to('cpu')

            outputs[i] = out
            targets[i] = y
            i += 1

        for i in range(targets.shape[0]):
            idx = 0 if targets[i] > 0.5 else 1
            idx += 0 if outputs[i] > 0.5 else 2
            c_matrix[idx] += 1
        
        auc = roc_auc_score(targets, outputs)
    
    print(f"Confusion matrix: {c_matrix}")
    print(f"ROC AUC score: {auc}")
