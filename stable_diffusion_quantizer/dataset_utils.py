import os
import re
import shutil
import json
from clint.textui import progress
import requests
from img2dataset import download
import pandas as pd
from addict import Dict


def split_parquet(path_src, path_dst, ratio=0.1):
    data = pd.read_parquet(path_src)
    new_size = int(len(data) * ratio)
    data_split = data[:new_size]
    data_split.to_parquet(path_dst)
    return path_dst


def get_parquet_dataset_path(path):
    expression = re.compile('[0-9]*.tar')
    names = [s for s in os.listdir(path) if expression.match(s)]
    names.sort()
    name = f"{{{names[0].replace('.tar', '')}..{names[-1].replace('.tar', '')}}}.tar"
    return os.path.join(path, name)


def download_file(url, output_name):
    print(url, output_name)
    r = requests.get(url, stream=True)
    with open(output_name, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
            if chunk:
                f.write(chunk)
                f.flush()
    return output_name


def download_webdataset(
        url,
        output_dir,
        split_ratio=1,
        processes_count=16,
        thread_count=32,
        image_size=512,
        input_format="parquet",
        output_format="webdataset",
        url_col="URL",
        caption_col="TEXT",
        enable_wandb=False,
        number_sample_per_shard=1000,
        distributor="multiprocessing",
        **kwargs
):
    try:
        cfg = Dict(json.load(open(os.path.join(output_dir, "cfg.json"))))
    except:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, "shards"))
        cfg = Dict({
            "ready": False,
            "src": None,
            "split": None,
            "split_ratio": split_ratio,
            "path": None
        })
    if cfg.src is None:
        cfg.src = download_file(url, os.path.join(output_dir, "files.parquet"))
    if not cfg.ready or split_ratio != cfg.split_ratio:
        if split_ratio < 1:
            cfg.split = split_parquet(cfg.src, os.path.join(output_dir, f"split_{split_ratio}.parquet"), split_ratio)
        else:
            cfg.split = cfg.src
        download(
            processes_count=processes_count,
            thread_count=thread_count,
            url_list=cfg.split,
            image_size=image_size,
            output_folder=os.path.join(output_dir, "shards"),
            output_format=output_format,
            input_format=input_format,
            url_col=url_col,
            caption_col=caption_col,
            enable_wandb=enable_wandb,
            number_sample_per_shard=number_sample_per_shard,
            distributor=distributor
        )
        cfg.path = get_parquet_dataset_path(os.path.join(output_dir, "shards"))
        cfg.ready = True
        with open(os.path.join(output_dir, "cfg.json"), "w") as f:
            json.dump(cfg, f, indent=4)
    return cfg
