import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm


def stable_diffusion_quantizer_unet_qat(
        pipeline,
        dataset,
        unet_teacher=None,
        # grad mode params
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        # mixed precision
        mixed_precision="no",  # ["no", "fp16", "bf16"]
        # logging
        logging_dir="logs",
        # optimizer params
        use_8bit_adam=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_weight_decay=1e-2,
        adam_epsilon=1e-8,
        # dataset params
        batch_size=1,
        num_workers=1,
        shuffle=True,
        # training parameters
        max_train_steps=100,
        num_train_epochs=1,
        max_grad_norm = 1.0,
        # checkpoint
        checkpoint_step = -1,
        **kwargs
):
    # init accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # init model
    scheduler = pipeline.scheduler
    text_encoder = pipeline.text_encoder
    unet = pipeline.unet
    vae = pipeline.vae

    # gradient mode
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # prepare params
    params = unet.parameters()

    # init optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # prepare dataset
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # prepare models by accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, trainloader
    )

    # apply mixed precision
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # push vae && text encoder to mixed precision
    if unet_teacher is not None:
        unet_teacher.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("stable_diffusion_quantizer_unet", config={})    
    
    # start training
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(trainloader):
            with accelerator.accumulate(unet):
                image, input_ids = batch
                # Convert images to latent space
                latents = vae.encode(image.to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(input_ids.to(accelerator.device))[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                if unet_teacher is not None:
                    with torch.no_grad():
                        teacher_noise_pred = unet_teacher(noisy_latents, timesteps, encoder_hidden_states).sample
                    loss += F.mse_loss(noise_pred.float(), teacher_noise_pred.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break
            elif checkpoint_step > 0 and not global_step % checkpoint_step:
                print("make checkpoint...")
                torch.save(unet.state_dict(), "unet_last.ckpt")

        accelerator.wait_for_everyone()

    # Create the pipeline using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline.unet = accelerator.unwrap_model(unet)

    accelerator.end_training()
    return pipeline

