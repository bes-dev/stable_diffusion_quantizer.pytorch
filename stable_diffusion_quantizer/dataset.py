import numpy as np
import webdataset as wds
# utils
from stable_diffusion_quantizer.utils import image_to_tensor
from torch.utils.data import Dataset


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


class TextDataset(Dataset):
    def __init__(self, path):
        self.data = open(path).readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].rstrip()


if __name__ == "__main__":
    from transformers import CLIPTokenizer
    from torch.utils.data import DataLoader
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    with open("/tmp/tmp.txt", "w") as f:
        for i in range(5):
            f.write(f"str{i}\n")
    dataset = TextDataset("/tmp/tmp.txt")
    loader = DataLoader(dataset, batch_size=3)
    for batch in loader:
        tokens = tokenizer(
            batch,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids
        print(tokens.size())
        break
