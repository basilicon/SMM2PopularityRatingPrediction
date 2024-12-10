from datasets import load_dataset
from kaitaistruct import KaitaiStream
from io import BytesIO
from level import Level
import zlib

ds = load_dataset("TheGreatRambler/mm2_level", streaming=True, split="train")
level_data = next(iter(ds))["level_data"]
level = Level(KaitaiStream(BytesIO(zlib.decompress(level_data))))

# NOTE level.overworld.objects is a fixed size (limitation of Kaitai struct)
# must iterate by object_count or null objects will be included
for i in range(level.overworld.object_count):
    obj = level.overworld.objects[i]
    print("X: %d Y: %d ID: %s" % (obj.x, obj.y, obj.id))