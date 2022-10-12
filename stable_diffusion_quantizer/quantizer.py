import inspect
import os
import numpy as np
# random
import random
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# dataset
from stable_diffusion_quantizer.dataset import DataPipelineTTI
# scheduler
from diffusers import DDPMScheduler, LMSDiscreteScheduler
# tokenizer
from transformers import CLIPTokenizer, CLIPModel
# models
from stable_diffusion_quantizer.models import UNet2DConditionModel, AutoencoderKL
# quantization
from stable_diffusion_quantizer.models import QConv2d, QLinear, Calibrator
# utils
from stable_diffusion_quantizer.utils import image_to_tensor, get_names_by_type, get_layer_by_name, set_layer_by_name
from tqdm.autonotebook import tqdm


class StableDiffusionQuantizer(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        # noise scheduler
        # TODO: LMSDiscreteScheduler works better for real use-cases
        self.scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        # condition model
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg["text_encoder"]["name"])
        self.text_encoder = CLIPModel.from_pretrained(cfg["text_encoder"]["name"])
        # unet
        self.unet = UNet2DConditionModel(**cfg["unet"]["cfg"])
        self.unet.load_state_dict(torch.load(cfg["unet"]["ckpt"], map_location="cpu"))
        # vae
        self.vae = AutoencoderKL(**cfg["vae"]["cfg"])
        self.vae.load_state_dict(torch.load(cfg["vae"]["ckpt"], map_location="cpu"))
        # dataset
        self.trainset = DataPipelineTTI(
            data_path = cfg["trainset"]["path"],
            tokenizer = self.tokenizer,
            **cfg["trainset"]["params"]
        )
        self.valset = DataPipelineTTI(
            data_path = cfg["valset"]["path"],
            tokenizer = self.tokenizer,
            **cfg["valset"]["params"]
        )
        # utils
        self.register_buffer("device_info", torch.tensor(0))
        # initialize quantization
        if cfg["trainer"]["initialize_quantization"]:
            print("Initialize quantization")
            self.initialize_quantization(
                self.unet,
                dataset=self.get_dataloader(
                    DataPipelineTTI(
                        data_path = cfg["initset"]["path"],
                        tokenizer = self.tokenizer,
                        **cfg["initset"]["params"]
                    ),
                    batch_size = cfg["initset"]["batch_size"],
                    num_workers = cfg["initset"]["num_workers"]
                )
            )
        self.distillation = False
        if cfg["trainer"]["distillation"]:
            self.distillation = True
            self.distillation_weight = cfg["trainer"]["distillation_weight"]
            self.unet_teacher = UNet2DConditionModel(**cfg["unet"]["cfg"])
            self.unet_teacher.load_state_dict(torch.load(cfg["unet"]["ckpt"], map_location="cpu"))

    def _log_loss(self, loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, exclude=["loss"]):
        for k, v in loss.items():
            if not k in exclude:
                self.log(k, v, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)

    @torch.no_grad()
    def make_unet_input(self, image, input_ids):
        # Convert images to latent space
        # latents = self.vae.encode(image).latent_dist.sample().detach()
        moments = self.vae.encode(image, return_moments=True)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        std = torch.exp(logvar * 0.5)
        latents = (mean + std * torch.randn(*mean.shape)) * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn(latents.shape).to(latents.device)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), device=self.device_info.device
        ).long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder.text_model(input_ids)[0]

        return noisy_latents, timesteps, encoder_hidden_states

    # generic step
    def step(self, batch, batch_nb):
        image, input_ids = batch
        # Predict the noise residual
        noisy_latents, timesteps, encoder_hidden_states = self.make_unet_input(image, input_ids)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states)["sample"]
        # compute loss
        loss = {}
        loss["mse"] = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
        loss["loss"] = loss["mse"]
        if self.distillation:
            with torch.no_grad():
                noise_pred_teacher = self.unet_teacher(noisy_latents, timesteps, encoder_hidden_states)["sample"]
            loss["mse_distillation"] = F.mse_loss(noise_pred, noise_pred_teacher, reduction="none").mean([1, 2, 3]).mean()
            loss["loss"] += self.distillation_weight * loss["mse_distillation"]
        return loss

    def get_dataloader(self, dataset, batch_size, num_workers):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

    # training step
    def training_step(self, batch, batch_nb):
        loss = self.step(batch, batch_nb)
        self._log_loss(loss)
        return loss

    def train_dataloader(self):
        return self.get_dataloader(self.trainset, self.cfg["trainset"]["batch_size"], self.cfg["trainset"]["num_workers"])

    # validation step
    def validation_step(self, batch, batch_nb):
        loss = self.step(batch, batch_nb)
        loss["loss_val"] = loss["loss"]
        self._log_loss(loss)
        return loss

    def val_dataloader(self):
        return self.get_dataloader(self.valset, self.cfg["valset"]["batch_size"], self.cfg["valset"]["num_workers"])

    def configure_optimizers(self):
        opts = []
        opts.append(
            torch.optim.AdamW(
                self.unet.parameters(),
                lr = self.cfg["optimizer"]["lr"],
                betas = (self.cfg["optimizer"]["adam_beta1"], self.cfg["optimizer"]["adam_beta2"]),
                weight_decay = self.cfg["optimizer"]["adam_weight_decay"],
                eps = self.cfg["optimizer"]["adam_epsilon"]
            )
        )
        return opts, []

    def initialize_quantization(self, m, dataset=None, conv=True, linear=True, bits=8, momentum=0.1):
        names = []
        if conv:
            names.extend(get_names_by_type(m, torch.nn.Conv2d))
        if linear:
            names.extend(get_names_by_type(m, torch.nn.Linear))
        # calibrator
        if dataset is not None:
            for name in names:
                src = get_layer_by_name(m, name)
                dst = Calibrator(src, momentum=momentum)
                set_layer_by_name(m, name, dst)
            for i, batch in enumerate(tqdm(dataset)):
                image, input_ids = batch
                image = image.to(self.device_info.device)
                input_ids = input_ids.to(self.device_info.device)
                self.unet(*self.make_unet_input(image, input_ids))
            for name in names:
                src = get_layer_by_name(m, name)
                if src.mtype() == nn.Conv2d:
                    dst = QConv2d(src.m, bits = bits)
                elif src.mtype() == nn.Linear:
                    dst = QLinear(src.m, bits = bits)
                else:
                    raise ValueError("Unknown module type")
                dst.activation_quantizer.lower.data.fill_(src.params["min"])
                dst.activation_quantizer.length.data.fill_(src.params["max"] - src.params["min"])
                dst.weight_quantizer.scale.data.fill_(src.m.weight.abs().max())
                set_layer_by_name(m, name, dst)
        else:
            for name in names:
                src = get_layer_by_name(m, name)
                if type(src) == nn.Conv2d:
                    dst = QConv2d(src, bits = bits)
                elif type(src) == nn.Linear:
                    dst = QLinear(src, bits = bits)
                else:
                    raise ValueError("Unknown module type")
                dst.weight_quantizer.scale.data.fill_(src.weight.abs().max())
                set_layer_by_name(m, name, dst)
        return names

    @torch.no_grad()
    def forward(
            self,
            prompt,
            init_image = None,
            mask = None,
            height = 512,
            width = 512,
            strength = 0.5,
            num_inference_steps = 32,
            guidance_scale = 7.5,
            eta = 0.0
    ):
        # extract condition
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device_info.device)
        # text_embeddings = self.text_encoder.get_text_features(tokens)
        text_embeddings = self.text_encoder.text_model(tokens)[0]

        # do classifier free guidance
        if guidance_scale > 1.0:
            tokens_uncond = self.tokenizer(
                "",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device_info.device)
            # uncond_embeddings = self.text_encoder.get_text_features(tokens_uncond)
            uncond_embeddings = self.text_encoder.text_model(tokens_uncond)[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # initialize latent latent
        if init_image is None:
            latents = torch.randn((self.unet.in_channels, height // 8, width // 8)).to(self.device_info.device)
            init_timestep = num_inference_steps
        else:
            init_latents = self._encode_image(init_image)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)
            timesteps = np.array([[self.scheduler.timesteps[-init_timestep]]]).astype(np.long)
            noise = np.random.randn(*self.latent_shape)
            latents = self.scheduler.add_noise(init_latents, noise, timesteps)[0]

        if init_image is not None and mask is not None:
            mask = self._preprocess_mask(mask)
        else:
            mask = None

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.stack([latents, latents]) if guidance_scale > 1.0 else latents[None]
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, text_embeddings)["sample"]

            # perform guidance
            if guidance_scale > 1.0:
                noise_pred = noise_pred[0] + guidance_scale * (noise_pred[1] - noise_pred[0])

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

            # masking for inapinting
            if mask is not None:
                init_latents_proper = self.scheduler.add_noise(init_latents, noise, t)
                latents = ((init_latents_proper * mask) + (latents * (1 - mask)))[0]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents[None])

        # convert tensor to opencv's image format
        image = image.detach().cpu().numpy()
        image = (image / 2 + 0.5).clip(0, 1)
        image = (image[0].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
        return image
