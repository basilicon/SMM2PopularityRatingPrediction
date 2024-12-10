from level import Level
import numpy as np
import pandas as pd
from io import BytesIO
import zlib
from kaitaistruct import KaitaiStream
import os
import matplotlib.pyplot as plt
from progress.bar import Bar

# hyperparameters
input_folder = "./data/unprocessed_levels/"
output_folder = "./data/processed_levels/"
output_prefix = "processed_"

class EncodedSubworld:
    obj_indices = [
        [0,1,3,12,15,28,30,32,39,40,41,45,46,48,50,51,52,56,60,61,62,65,72,74,77,86,96,98,102,103,104,107,111,114,117,120,121,122,123,124,125,126],
        [2,58,78],
        [68],
        [13,24,43,47,54,83,110,118],
        [57],
        [4,5,6,26,63],
        [79,84,99,100],
        [8,10,18,19,36,112,132],
        [11,14,16,17,21,22,29,49,67,71,105,106,113],
        [27,55,90,92,95,97],
        [20,33,34,35,42,44,70,81,127,128,129,130,131],
        [7,9,23,53,80,85,87,88,93,94,119]
    ]

    obj_masks = np.asarray([
        [0,0,1,1,0,1,0,0],
        [0,0,1,1,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [1,0,1,0,0,0,0,0],
        [1,0,1,1,0,0,0,0],
        [1,0,0,1,0,0,0,0],
        [1,0,0,1,1,0,0,0],
        [0,1,0,1,1,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0,1],
        [1,0,0,0,0,0,0,0],
    ], dtype=np.float32)

    mask_table = {
        "ground": np.asarray([1,0,0,0,0,0,0,0], dtype=np.float32),
        "track": np.asarray([0,0,0,0,1,0,0,0], dtype=np.float32),
        "sound": np.asarray([0,0,0,0,0,0,1,0], dtype=np.float32),
        "icicle": np.asarray([0,1,1,1,0,0,0,0], dtype=np.float32)
    }

    def __init__(self, map: Level.Map) -> None:
        self.theme = int(map.theme)
        self.has_autoscroll = map.autoscroll_type != Level.Map.AutoscrollType.none
        self.orientation = int(map.orientation)

        self.map_tensor = np.zeros(shape=np.asarray(((map.boundary_right / 16, map.boundary_top / 16, 8)), dtype=np.int32))

        # objects
        for i in range(map.object_count):
            obj_mask_idx = -1
            obj = map.objects[i]
            for j in range(len(EncodedSubworld.obj_indices)):
                if int(obj.id) in EncodedSubworld.obj_indices[j]:
                    obj_mask_idx = j
                    break
            
            if obj_mask_idx == -1:
                continue

            pos = np.floor_divide((obj.x, obj.y), 160)
            if isValid(self.map_tensor.shape, pos):
                self.map_tensor[pos[0], pos[1]] += EncodedSubworld.obj_masks[obj_mask_idx]
        
        # sounds 
        for i in range(map.sound_effect_count):
            pos = (map.sounds[i].x, map.sounds[i].y)
            if isValid(self.map_tensor.shape, pos):
                self.map_tensor[pos[0], pos[1]] += EncodedSubworld.mask_table["sound"]
        
        # ground
        for i in range(map.ground_count):
            pos = (map.ground[i].x, map.ground[i].y)
            if isValid(self.map_tensor.shape, pos):
                self.map_tensor[pos[0], pos[1]] += EncodedSubworld.mask_table["ground"]
        
        # tracks
        for i in range(map.track_count):
            pos = (map.tracks[i].x, map.tracks[i].y)
            if isValid(self.map_tensor.shape, pos):
                self.map_tensor[pos[0], pos[1]] += EncodedSubworld.mask_table["track"]
        
        # icicles
        for i in range(map.ice_count):
            pos = (map.icicles[i].x, map.icicles[i].y)
            if isValid(self.map_tensor.shape, pos):
                self.map_tensor[pos[0], pos[1]] += EncodedSubworld.mask_table["icicle"]
        
        #plt.imshow(self.map_tensor.transpose(1, 0, 2)[:,:,:3], origin='lower')
        #plt.show()
        global global_bar
        global_bar.next()

def isValid(np_shape: tuple, index: tuple):
    if min(index) < 0:
        return False
    for ind,sh in zip(index,np_shape):
        if ind >= sh:
            return False
    return True

def process_levels() -> None:
    for file in os.listdir(input_folder):
        fpath = os.path.join(input_folder, file)

        if not os.path.isfile(fpath):
            continue
        
        df = pd.read_parquet(fpath)

        global global_bar
        global_bar = Bar(f'Reading {file}', max=df.shape[0] * 2, suffix='%(percent)d%% [%(elapsed_td)s:%(eta_td)s]')

        res = df.apply(get_subworld_encodings, axis='columns', result_type='expand')
        res.to_parquet(os.path.join(output_folder, output_prefix + file))

def get_subworld_encodings(row):
    level = Level(KaitaiStream(BytesIO(zlib.decompress(row["level_data"]))))
    overworld = EncodedSubworld(level.overworld)
    subworld = EncodedSubworld(level.subworld)

    return (overworld.theme, 
            overworld.has_autoscroll, 
            overworld.orientation, 
            overworld.map_tensor.shape, 
            overworld.map_tensor.flatten(), 
            subworld.theme, 
            subworld.has_autoscroll, 
            subworld.orientation, 
            subworld.map_tensor.shape,
            subworld.map_tensor.flatten()
            )

if __name__ == "__main__":
    process_levels()

    #df = pd.read_parquet("./data/unprocessed_levels/batch_20.parquet")
    #get_subworld_encodings(df.iloc[5])