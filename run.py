from argparse import ArgumentParser
import json
from addict import Dict
from diffusers import StableDiffusionPipeline
from stable_diffusion_quantizer.unet.quantizer import stable_diffusion_quantizer_unet


def main(args):
    cfg = Dict(json.load(open(args.cfg)))
    pipeline = StableDiffusionPipeline.from_pretrained(
        cfg.model.model_name,
        use_auth_token=args.token
    )
    pipeline = stable_diffusion_quantizer_unet(pipeline, cfg)
    pipeline.save_pretrained(args.output_name)


if __name__ == "__main__":
    parser = ArgumentParser("Stable Diffusion Quantizer")
    parser.add_argument("--cfg", type=str, default="configs/config_template.json")
    parser.add_argument("--token", type=str, help="hf_auth_token")
    parser.add_argument("--output-name", type=str, help="output name")
    args = parser.parse_args()
    main(args)
