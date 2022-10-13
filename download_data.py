import argparse
import os
import shutil
import torch
from img2dataset import download
from diffusers import StableDiffusionPipeline


def main(args):
    print("Download dataset...")
    subsets = args.subsets.split(',')
    for subset in subsets:
        print(f"download '{subset}' subset...")
        download(
            processes_count=args.processes_count,
            thread_count=args.thread_count,
            url_list=os.path.join(args.data_folder, subset, f"{subset}.parquet"),
            image_size=args.image_size,
            output_folder=os.path.join(args.data_folder, subset),
            output_format=args.output_format,
            input_format="parquet",
            url_col=args.url_col,
            caption_col=args.caption_col,
            enable_wandb=False,
            distributor="multiprocessing",
            resize_only_if_bigger=True,
            resize_mode="keep_ratio",
            skip_reencode=True,
            save_additional_columns=["similarity","hash","punsafe","pwatermark","aesthetic","LANGUAGE"]
        )

    if args.use_auth_token is not None:
        print("Download model...")
        args_ext = {}
        if args.fp16:
            args_ext["torch_type"] = torch.float16
            args_ext["revision"] = "fp16"
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.model_name,
            use_auth_token=args.use_auth_token,
            **args_ext
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--data-folder", type=str, default="data/")
    parser.add_argument("--subsets", type=str, default="train,val,init")
    # image size
    parser.add_argument("--image-size", type=int, default=512)
    # multiprocessing
    parser.add_argument("--processes-count", type=int, default=16)
    parser.add_argument("--thread-count", type=int, default=64)
    # data format
    parser.add_argument("--input-format", type=str, default="parquet")
    parser.add_argument("--output-format", type=str, default="webdataset")
    # col names
    parser.add_argument("--url-col", type=str, default="URL")
    parser.add_argument("--caption-col", type=str, default="TEXT")
    # huggingface-hub token & model params
    parser.add_argument("--use-auth-token", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    main(args)
