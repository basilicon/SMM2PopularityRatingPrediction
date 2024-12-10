from subworld_autoencoder import SubworldAutoencoder, MultiBatchDataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from progress.bar import Bar

# hyperparameters
fname = './models/subworld_autoencoder/sblvl-ae1.pth'
source_folder = "./data/processed_levels/"
output_file = './data/ae2_levels.csv'
model = SubworldAutoencoder(8)

# main
if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        print("No GPU available. Exiting...")
        exit()

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(fname, weights_only=True))

    encoder = model.module.encoder
    encoder.eval()
    encoder.to('cuda')

    dataset = MultiBatchDataset(source_folder)
    loader = DataLoader(dataset, shuffle=False)

    arr = np.zeros((len(dataset), 64))

    bar = Bar(f'Mapping info to dataframe', max=len(dataset), suffix='%(percent)d%% [%(elapsed_td)s:%(eta_td)s]')
    i = 0
    overworld = True
    for item in loader:
        output = encoder(item.to('cuda')).to('cpu')
        arr[i] = output.detach().numpy()
        i += 1

        bar.next()
    bar.finish()
    
    arr = arr.reshape((len(dataset) // 2), 2, 64)
    df = pd.DataFrame({
        'overworld': list(arr[:, 0, :]),
        'subworld': list(arr[:, 1, :])
    })

    df.to_csv(output_file)

