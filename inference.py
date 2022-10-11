import argparse
import os
import json
import torch
# scheduler
from diffusers import LMSDiscreteScheduler, PNDMScheduler
# model
from stable_diffusion_ext import StableDiffusionInferenceEngine
# utils
import cv2
import numpy as np
from stable_diffusion_ext.utils import from_duffusers_ckpt


def main(args):
    assert args.models_dir is not None
    cfg = from_duffusers_ckpt(args.models_dir)
    # scheduler
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        tensor_format="pt"
    )
    # pipeline
    pipeline = StableDiffusionInferenceEngine(scheduler, cfg)
    while True:
        prompt = input("> ")
        if not prompt:
            break
        img = pipeline(prompt)
        cv2.imshow("img", img)
        key = chr(cv2.waitKey() & 255)
        if key == "q":
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--device", type=str, default="cpu", help="device {cpu, cuda}")
    parser.add_argument("--models-dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
