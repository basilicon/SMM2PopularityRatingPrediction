import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from progress.bar import Bar
import numpy as np
import os
from math import ceil

# 0. Hyperparameters
lr = 1e-3
model_output_path = "./models/subworld_autoencoder/sblvl-ae2.pth"
source_folder = "./data/processed_levels/"
n_epochs = 10

# 1. Model declaration
class Inception(nn.Module):
    def __init__(self, in_channels, out_channels, l2_channel_reduction = 0.75, l3_channel_reduction = 0.5):
        super(Inception, self).__init__()

        l1_channels = out_channels // 4
        l3_4_channels = out_channels // 8
        l31_channels = ceil(l3_4_channels * l3_channel_reduction)

        l22_channels = out_channels - (l1_channels + l3_4_channels * 2)
        l21_channels = ceil(l22_channels * l2_channel_reduction)

        self.l1 = nn.Conv2d(in_channels, l1_channels, kernel_size=(1,1))

        self.l21 = nn.Conv2d(in_channels, l21_channels, kernel_size=(1,1))
        self.l22 = nn.Conv2d(l21_channels, l22_channels, kernel_size=(3,3), padding='same', padding_mode='replicate')

        self.l31 = nn.Conv2d(in_channels, l31_channels, kernel_size=(1,1))
        self.l32 = nn.Conv2d(l31_channels, l3_4_channels, kernel_size=(5,5), padding='same', padding_mode='replicate')

        self.l41 = nn.MaxPool2d((3,3), stride=(1,1), padding=(1,1))
        self.l42 = nn.Conv2d(in_channels, l3_4_channels, kernel_size=(1,1))
    
    def forward(self, x):
        l1 = self.l1(x)

        l2 = torch.relu(self.l21(x))
        l2 = self.l22(l2)

        l3 = torch.relu(self.l31(x))
        l3 = self.l32(l3)

        l4 = torch.relu(self.l41(x))
        l4 = self.l42(l4)

        x = torch.concat([l1, l2, l3, l4], dim=1)
        return x

class SubworldEncoder(nn.Module):
    def __init__(self, in_channels):
        super(SubworldEncoder, self).__init__()

        self.i1 = Inception(in_channels, 64, l3_channel_reduction=0.75)
        self.p2 = nn.AvgPool2d(kernel_size=(2,2), ceil_mode=True)
        self.i3 = Inception(64, 128)
        self.p4 = nn.AvgPool2d(kernel_size=(2,2))
        self.i5 = Inception(128, 256)
        self.f6 = nn.Flatten(2, 3)
        self.c7 = nn.Conv1d(256, 32, kernel_size=(1))
        self.p8 = nn.AdaptiveAvgPool2d(32)
        self.f9 = nn.Flatten(1, 2)
        self.l10 = nn.Linear(1024, 256)
        self.l11 = nn.Linear(256, 64)
    
    def forward(self, x):
        x = torch.relu(self.i1(x))
        x = self.p2(x)
        x = torch.relu(self.i3(x))
        x = self.p4(x)
        x = torch.relu(self.i5(x))
        x = self.f6(x)
        x = torch.sigmoid(self.c7(x))
        x = self.p8(x)
        x = self.f9(x)
        x = torch.relu(self.l10(x))
        x = torch.sigmoid(self.l11(x))

        return x
    
class SubworldDecoder(nn.Module):
    def __init__(self, out_channels):
        super(SubworldDecoder, self).__init__()

        self.l12 = nn.Linear(64, 256, 8)
        self.l13 = nn.Linear(256, 1024, 32)
        self.f14 = nn.Unflatten(1, (32, 32))
        # p15 is Upsample
        self.c16 = nn.Conv1d(32, 256, kernel_size=(1))
        # f17 is Reshape
        self.i18 = Inception(256, 128)
        # p19 is UpsamplingNearest2d
        self.i20 = Inception(128, 64)
        # p21 is UpsamplingNearest2d
        self.i22 = Inception(64, out_channels, l2_channel_reduction=1.5, l3_channel_reduction=2)
    
    def forward(self, x, subworld_shape):
        subworld_shape = torch.from_numpy(subworld_shape)
        p2_shape = torch.ceil(subworld_shape * 0.5)
        p4_shape = torch.floor(p2_shape * 0.5)
        p15_length = torch.prod(p4_shape)

        subworld_shape = subworld_shape.int()
        p2_shape = p2_shape.int()
        p4_shape = torch.concat([torch.tensor([1, 256]), p4_shape]).int()
        p15_length = p15_length.int()

        x = torch.relu(self.l12(x))
        x = torch.relu(self.l13(x))
        x = self.f14(x)
        x = F.interpolate(x, size=p15_length)
        x = torch.relu(self.c16(x))
        x = torch.reshape(x, tuple(p4_shape))
        x = torch.relu(self.i18(x))
        x = F.interpolate(x, size=list(p2_shape))
        x = torch.relu(self.i20(x))
        x = F.interpolate(x, size=list(subworld_shape))
        x = torch.sigmoid(self.i22(x))

        return x

class SubworldAutoencoder(nn.Module):
    def __init__(self, io_channels):
        super(SubworldAutoencoder, self).__init__()

        self.encoder = SubworldEncoder(io_channels)
        self.decoder = SubworldDecoder(io_channels)
    
    def forward(self, x):
        subworld_shape = np.asarray(x.shape[2:4], dtype=np.float32)

        x = self.encoder(x)
        x = self.decoder(x, subworld_shape)

        return x

# 2. Data preprocessing
class MultiBatchDataset(Dataset):
    def __init__(self, source_folder):
        self.source_folder = source_folder
        self.buffer = None
        self.buffer_idx = -1

        self.index_list = []
        self.fname_list = []
        self.size = 0

        bar = Bar(f'Loading dataset from {source_folder} ...', max=len(os.listdir(source_folder)), suffix='%(percent)d%% [%(elapsed_td)s:%(eta_td)s]')
        for file in os.listdir(source_folder):
            fpath = os.path.join(source_folder, file)
            if not os.path.isfile(fpath):
                continue
            
            df = pd.read_parquet(fpath)
            self.index_list.append(self.size)
            self.fname_list.append(fpath)
            self.size += df.shape[0]
            bar.next()
            # break
        bar.finish()

    def __len__(self):
        return self.size * 2

    def __getitem__(self, idx):
        analyze_subworld = idx % 2 == 1
        idx //= 2

        idx_idx = 0
        while idx_idx < len(self.index_list) and self.index_list[idx_idx] <= idx:
            idx_idx += 1
        idx_idx -= 1

        idx -= self.index_list[idx_idx]

        if self.buffer_idx == -1 or idx_idx != self.buffer_idx:
            self.buffer = pd.read_parquet(self.fname_list[idx_idx])
            self.buffer_idx = idx_idx

        x_shape = self.buffer.iat[idx, 3 if analyze_subworld else 8]
        x = np.asarray(self.buffer.iloc[idx, 4 if analyze_subworld else 9], dtype=np.float32)
        return x.reshape(x_shape).transpose((2, 0, 1))

if __name__ == '__main__':
    if False:
        model = SubworldAutoencoder(8)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(pytorch_total_params)
        exit()

    # 3. Training
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        print("No GPU available. Exiting...")
        exit()

    dataset = MultiBatchDataset(source_folder)
    print(f"Loaded dataset with {len(dataset)} instances.")

    model = SubworldAutoencoder(io_channels=8)
    model = nn.DataParallel(model)
    model.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    loader = DataLoader(dataset, shuffle=False)

    for epoch in range(n_epochs):
        bar = Bar(f'Training epoch {epoch}', max=len(dataset), suffix='%(percent)d%% [%(elapsed_td)s:%(eta_td)s]')
        for data in loader:
            data = data.to('cuda')
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, data)
            loss.backward()
            optimizer.step()

            bar.next()
        bar.finish()

    print(f"Finished training model. Saving model to {model_output_path}")
    torch.save(model.state_dict(), model_output_path)