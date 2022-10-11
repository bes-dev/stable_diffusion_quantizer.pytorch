# Stable Diffusion Quantizer

# !!!Work In Progress!!!

PyTorch implementation of Stable Diffusion quantization pipeline


## Requirements

* Python 3.8+
* 1–8 high-end NVIDIA GPUs with at least 24 GB of memory.

## Training

```bash
pip install -r requirements.txt
python quantize.py --models-dir <path_to_diffusers_sd_checkpoint> --cfg <path_to_quantizer_cfg> --gpus <n_gpus>
```
