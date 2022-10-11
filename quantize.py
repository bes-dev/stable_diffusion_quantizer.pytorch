import argparse
import os
import json
import torch
import pytorch_lightning as pl
# scheduler
from diffusers import LMSDiscreteScheduler, PNDMScheduler
# model
from stable_diffusion_ext import StableDiffusionQuantizer
# utils
import cv2
import numpy as np
from stable_diffusion_ext.utils import from_duffusers_ckpt


def main(args):
    # load config
    assert args.models_dir is not None and args.cfg is not None
    cfg = from_duffusers_ckpt(args.models_dir)
    cfg.update(json.load(open(args.cfg)))
    # create pipeline
    pipeline = StableDiffusionQuantizer(cfg)
    if args.ckpt is not None:
        pipeline.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    # checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.getcwd() if args.checkpoint_dir is None else args.checkpoint_dir,
        filename="{epoch}",
        save_top_k=True,
        save_last=True,
        verbose=True,
        monitor=cfg["trainer"]["monitor"],
        mode=cfg["trainer"]["monitor_mode"],
    )
    if args.gpus > 0:
        engine = {"devices": args.gpus, "accelerator": "gpu", "strategy": "ddp"}
    else:
        engine = {"accelerator": "cpu"}
    trainer = pl.Trainer(
        max_epochs=cfg["optimizer"]["epochs"],
        accumulate_grad_batches=args.grad_batches,
        val_check_interval=args.val_check_interval if args.val_check_interval > 0 else 1.0,
        gradient_clip_val=15,
        gradient_clip_algorithm="value",
        callbacks=[checkpoint_callback],
        **engine
    )
    trainer.fit(pipeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # hardware
    parser.add_argument("--gpus", type=int, default=0, help="Number of available GPUs")
    parser.add_argument('--distributed-backend', type=str, default="ddp", choices=('dp', 'ddp', 'ddp2'),
                        help='Supports three options dp, ddp, ddp2')
    # pipeline configure
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=None)
    # checkpoint
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="path to store checkpoints")
    # pipeline
    parser.add_argument("--val-check-interval", type=int, default=0, help="Validation check interval")
    parser.add_argument("--grad-batches", type=int, default=1, help="Number of batches to accumulate")

    args = parser.parse_args()
    main(args)
