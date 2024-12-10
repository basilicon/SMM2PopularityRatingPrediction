import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from progress.bar import Bar
import numpy as np

# 0. Hyperparameters
k_folds = 5
n_epochs = 1000
batch_size = 256
lr = 3e-5
model_output_folder = None # "./models/mixed_models/"
seed = 42
loss_report_freq = 7500

# tabular, ae, mixed
model_type = "ae"

source_file = "./data/ae_levels.csv"
labels_file = './data/tabular_only.csv'
loss_function = nn.MSELoss()

# 1. Model declaration
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

class ScorePredictionNN(nn.Module):
    def __init__(self):
        super(ScorePredictionNN, self).__init__()
        self.layer1 = nn.Linear(32, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.output_layer(x))
        return x

class AEScorePredictionNN(nn.Module):
    def __init__(self):
        super(AEScorePredictionNN, self).__init__()
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.output_layer(x))
        return x

class MixedScorePredictionNN(nn.Module):
    def __init__(self):
        super(MixedScorePredictionNN, self).__init__()
        self.layer1 = nn.Linear(160, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.output_layer(x))
        return x

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        print("No GPU available. Exiting...")
        exit()

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = None
    match model_type:
        case "tabular":
            model = ScorePredictionNN
        case "ae":
            model = AEScorePredictionNN
        case "mixed":
            model = MixedScorePredictionNN
        case _:
            print("ERROR! Invalid type given. Exiting...")
            exit()

    network = model()
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Total params: {pytorch_total_params}")
    network = None

    # 2. Data preprocessing
    df = pd.read_csv(source_file)
    df2 = pd.read_csv(labels_file)

    X, Y = None, torch.round(torch.from_numpy(df2['likes-norm'].to_numpy()).reshape((-1, 1)).type(torch.FloatTensor))

    match model_type:
        case "tabular":
            X = torch.cat([
                F.one_hot(torch.from_numpy(df2['gamestyle'].to_numpy()).type(torch.LongTensor), num_classes=5),
                F.one_hot(torch.from_numpy(df2['theme'].to_numpy()).type(torch.LongTensor), num_classes=10),
                F.one_hot(torch.from_numpy(df2['tag1'].to_numpy()).type(torch.LongTensor), num_classes=16) + F.one_hot(torch.from_numpy(df2['tag2'].to_numpy()).type(torch.LongTensor), num_classes=16),
                torch.from_numpy(df2['timer'].to_numpy()).reshape((-1, 1))
            ], dim=1).type(torch.FloatTensor)
        case "ae":
            X = torch.cat([
                torch.from_numpy(np.genfromtxt(df['overworld'].apply(lambda x: x.strip("[]").replace("\n", " ")))),
                torch.from_numpy(np.genfromtxt(df['subworld'].apply(lambda x: x.strip("[]").replace("\n", " "))))
            ], dim=1).float()
        case "mixed":
            X = torch.cat([
                F.one_hot(torch.from_numpy(df2['gamestyle'].to_numpy()).type(torch.LongTensor), num_classes=5),
                F.one_hot(torch.from_numpy(df2['theme'].to_numpy()).type(torch.LongTensor), num_classes=10),
                F.one_hot(torch.from_numpy(df2['tag1'].to_numpy()).type(torch.LongTensor), num_classes=16) + F.one_hot(torch.from_numpy(df2['tag2'].to_numpy()).type(torch.LongTensor), num_classes=16),
                torch.from_numpy(df2['timer'].to_numpy()).reshape((-1, 1)),
                torch.from_numpy(np.genfromtxt(df['overworld'].apply(lambda x: x.strip("[]").replace("\n", " ")))),
                torch.from_numpy(np.genfromtxt(df['subworld'].apply(lambda x: x.strip("[]").replace("\n", " "))))
            ], dim=1).float()

    df, df2 = None, None

    dataset = TensorDataset(X, Y)

    # 3. K-fold cross validation
    kfold = KFold(n_splits=k_folds, shuffle=True)
    results = {}

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # random undersampling
        positive_ids = train_ids[Y[train_ids][:,0] > 0.5]
        negative_ids = train_ids[Y[train_ids][:,0] < 0.5]
        positive_ids = np.random.choice(positive_ids, len(negative_ids) * 2)
        train_ids = np.concat((positive_ids, negative_ids))

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        testloader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

        # Initialize network
        network = model()
        network = nn.DataParallel(network)
        network.apply(reset_weights)
        network.to('cuda')

        # Initialize optimizer
        optimizer = optim.Adam(network.parameters(), lr=lr)

        # Training loop
        bar = Bar(f'Training fold {fold}', max=n_epochs*trainloader.__len__(), suffix='%(percent)d%% [%(elapsed_td)s:%(eta_td)s]')
        for epoch in range(n_epochs):
            loss_report_counter = 0
            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                loss_report_counter += batch_size
                if loss_report_counter > loss_report_freq:
                    bar.message = f'Training fold {fold}, loss = {loss.item()}'
                    loss_report_counter -= loss_report_freq
                bar.next()
        bar.finish()

        print(f"Finished fold {fold}.")
        if model_output_folder != None:
            output_path = model_output_folder + f"ae-model-fold-{fold}.pth"
            print(f"Saving model to {output_path}")
            torch.save(network.module.state_dict(), output_path)

        # Evaluation
        network.eval()
        c_matrix = [0, 0, 0, 0]
        auc = 0

        with torch.no_grad():
            inputs, targets = X[test_ids], Y[test_ids]
            inputs = inputs.to('cuda')
            outputs = network(inputs).to('cpu')

            for i in range(targets.shape[0]):
                idx = 0 if targets[i] > 0.5 else 1
                idx += 0 if outputs[i] > 0.5 else 2
                c_matrix[idx] += 1
            
            auc = roc_auc_score(targets, outputs)

        print(f"Confusion matrix for this iteration: {c_matrix}")
        print(f"ROC AUC score for this iteration: {auc}")
        results[fold] = {
           "confusion": c_matrix,
           "auc": auc
        }

    print("Finished testing. Final results are: ")
    print(results)