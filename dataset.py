import torch
import numpy as np
import webdataset as wds
# tokenizer
from transformers import CLIPTokenizer
from stable_diffusion_ext.utils import image_to_tensor
from stable_diffusion_ext.dataset import DataPipelineTTI
import time

dataset = DataPipelineTTI(
    data_path = "/home/sergei/work/datasets/tti_test/00000.tar",
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
)


loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=3,
    num_workers=0
)

start = time.time()
for image,txt in loader:
    stop = time.time()
    print(f"{stop - start} s.")
    start = stop
    # print(image.shape)
    # print(txt.shape)
    # break
