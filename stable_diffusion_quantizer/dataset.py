import torch
import numpy as np
import webdataset as wds
# utils
from stable_diffusion_quantizer.utils import image_to_tensor


def image_preprocess(bgr2rgb=True, normalize=True, vrange=(0.0, 255.0), image_size=None, return_tensors="pt"):
    def _image_preprocess(image):
        return image_to_tensor(
            np.array(image),
            bgr2rgb=bgr2rgb,
            normalize=normalize,
            vrange=vrange,
            image_size=image_size,
            return_tensors=return_tensors
        )
    return _image_preprocess


def txt_preprocess(tokenizer, padding="max_length", truncation=True, return_tensors="pt"):
    def _txt_preprocess(txt):
        return tokenizer(
            txt,
            padding=padding,
            max_length=tokenizer.model_max_length,
            truncation=truncation,
            return_tensors=return_tensors
        ).input_ids[0]
    return _txt_preprocess


class DataPipelineTTI(wds.DataPipeline):
    def __init__(
            self,
            data_path,
            tokenizer,
            detshuffle=100,
            decode="pil",
            to_tuple="png;jpg txt",
            # image preprocess
            image_size=(512, 512),
            normalize=True,
            bgr2rgb=False,
            vrange=(0.0, 255.0),
            # text preprocess
            padding="max_length",
            truncation=True,
            return_tensors="pt"
    ):
        super().__init__(
            wds.SimpleShardList(data_path),
            wds.detshuffle(detshuffle),
            wds.tarfile_to_samples(),
            wds.decode(decode),
            wds.to_tuple(to_tuple),
            wds.map_tuple(
                image_preprocess(bgr2rgb, normalize, vrange, image_size, return_tensors),
                txt_preprocess(tokenizer, padding, truncation, return_tensors)
            )
        )
