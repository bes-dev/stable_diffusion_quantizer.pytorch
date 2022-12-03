from stable_diffusion_quantizer.modules.calibrator import Calibrator
from stable_diffusion_quantizer.modules.modules import QLinear, QConv2d
from stable_diffusion_quantizer.utils import *
from tqdm.auto import tqdm
from addict import Dict
from random_prompt import RandomPromptGenerator


def stable_diffusion_quantizer_unet_calibrate(
        pipeline,
        steps=20,
        batch_size=1,
        **kwargs
):
    prompt_generator = RandomPromptGenerator()
    for _ in tqdm(range(steps)):
        pipeline(prompt_generator.random_prompts(batch_size))