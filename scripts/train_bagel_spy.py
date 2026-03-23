"""
Bagel Spy-Civ Game Training Script (Vision-Zero dual task).

Each step:
  1. Generate game data (scene description pairs, spy assignment)
  2. N players each generate an image via Bagel (spy gets modified prompt)
  3. All players vote by looking at all images (Bagel understanding mode)
  4. Game rewards → Flow-GRPO advantages → per-SDE-step PPO-clip loss

Based on:
  - flow_grpo/scripts/train_bagel.py (model loading, FSDP, inferencer)
  - SPY-UMM/train_spy_umm.py (game loop structure)
"""

from collections import defaultdict
import contextlib
import os
import datetime
import time
import hashlib
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers.utils.torch_utils import is_compiled_module

# bagel
from flow_grpo.bagel.data.data_utils import add_special_tokens
from flow_grpo.bagel.data.transforms import ImageTransform
from flow_grpo.bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from flow_grpo.bagel.modeling.qwen2 import Qwen2Tokenizer
from flow_grpo.bagel.modeling.autoencoder import load_ae
from flow_grpo.bagel.inferencer import InterleaveInferencer

# spy game
from flow_grpo.spy_game_data import SpyGameDataGenerator
from flow_grpo.spy_game_reward import (
    build_voting_grid, tensor_images_to_pil, run_bagel_vote,
    compute_group_advantages,
)

import numpy as np
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model
from flow_grpo.fsdp_utils import save_fsdp_checkpoint
from huggingface_hub import snapshot_download

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
logger = get_logger(__name__)


def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    # For spy game: gradient_accumulation = G * N * sde_window_size
    spy_cfg = config.spy_game
    num_players = spy_cfg.num_players
    G = spy_cfg.group_size
    grad_accum = G * num_players * config.sample.sde_window_size

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=grad_accum,
    )
    accelerator.state.fsdp_plugin.activation_checkpointing = config.activation_checkpointing
    accelerator.state.fsdp_plugin.transformer_cls_names_to_wrap = ['Qwen2MoTDecoderLayer']

    if accelerator.is_main_process:
        wandb.init(project="flow_grpo_spy", name=config.run_name, config=config.to_dict())
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # ==================== MODEL LOADING ====================
    model_path = config.pretrained.model
    if not os.path.exists(model_path):
        model_local_dir = snapshot_download(repo_id=model_path)
    else:
        model_local_dir = model_path

    llm_config = Qwen2Config.from_json_file(os.path.join(model_local_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config for understanding mode (voting)
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_local_dir, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE
    vae_model, vae_config = load_ae(local_path=os.path.join(model_local_dir, "ae.safetensors"))

    # Bagel config: enable BOTH generation AND understanding
    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,  # Enable ViT for understanding/voting
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, bagel_config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_local_dir)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(512, 256, 8)
    vit_transform = ImageTransform(490, 112, 7)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_local_dir, "ema.safetensors"),
        device_map={"": f"cuda:{accelerator.local_process_index}"},
        offload_buffers=False,
        dtype=inference_dtype,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )
    model = model.eval()

    vae_model.requires_grad_(False)
    model.requires_grad_(False)

    inference_hyper = dict(
        cfg_img_scale=1.0,
        cfg_interval=[0, 1.0],
        timestep_shift=config.train.timestep_shift,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(config.resolution, config.resolution),
    )

    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=vae_transform, vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    vae_model.to(accelerator.device, dtype=inference_dtype)
    model.to(accelerator.device, dtype=inference_dtype)

    # ==================== TRAINABLE PARAMS ====================
    if config.use_lora:
        target_modules = [
            "self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen",
            "self_attn.v_proj_moe_gen", "self_attn.o_proj_moe_gen",
            "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
        ]
        lora_config = LoraConfig(r=64, lora_alpha=128, init_lora_weights="gaussian",
                                 target_modules=target_modules)
        model.language_model = get_peft_model(model.language_model, lora_config)
        for name, param in model.language_model.named_parameters():
            if 'lora' in name:
                param.data = param.data.to(dtype=inference_dtype)
    else:
        for name, param in model.language_model.named_parameters():
            if 'moe_gen' in name:
                param.requires_grad = True

    transformer = model.language_model
    transformer.config.use_cache = False
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    model.language_model = transformer

    # ==================== GAME SETUP ====================
    game_generator = SpyGameDataGenerator(
        num_players=num_players,
        num_objects_min=spy_cfg.get('num_objects_min', 3),
        num_objects_max=spy_cfg.get('num_objects_max', 6),
        num_to_modify=spy_cfg.get('num_to_modify', 2),
    )
    max_vote_tokens = spy_cfg.get('max_vote_tokens', 512)
    num_inner_epochs = config.train.num_inner_epochs

    # ==================== TRAINING LOOP ====================
    logger.info("***** Running Bagel Spy-Civ Flow-GRPO Training *****")
    logger.info(f"  Players={num_players}, G={G}, inner_epochs={num_inner_epochs}")

    global_step = 0
    spy_caught_count = 0
    total_games = 0

    for epoch in range(config.num_epochs):
        # ── Eval & Checkpoint ────────────────────────────────────
        if not config.debug and epoch > 0 and epoch % config.save_freq == 0:
            save_fsdp_checkpoint(config.save_dir, transformer, global_step,
                                 accelerator.process_index)

        # ── Sampling: G games ────────────────────────────────────
        transformer.eval()
        all_game_trajs = []      # [G][N] each is dict with latents, log_probs, etc.
        all_game_rewards = []    # [G] each is list of N rewards
        all_game_images = []     # [G][N] PIL images for logging

        for g in tqdm(range(G), desc=f"Epoch {epoch}: games",
                      disable=not accelerator.is_local_main_process):

            game_data = game_generator.generate_game(epoch * G + g, global_step)
            prompts = [
                game_generator.format_generation_prompt_simple(game_data, pid)
                for pid in range(1, num_players + 1)
            ]

            # ── Phase 1: Each player generates an image ──────────
            player_trajs = []
            player_images_tensor = []

            with autocast():
                for pid in range(num_players):
                    with torch.no_grad():
                        output_dict = inferencer(
                            text=prompts[pid],
                            noise_level=config.sample.noise_level,
                            grpo_config=config,
                            accelerator=accelerator,
                            num_timesteps=config.sample.num_steps,
                            cfg_text_scale=config.sample.guidance_scale,
                            **inference_hyper,
                        )
                    player_trajs.append({
                        'image': output_dict['image'],          # [3,H,W]
                        'all_latents': output_dict['all_latents'],
                        'all_log_probs': output_dict['all_log_probs'],
                        'timesteps': output_dict['timesteps'],
                    })
                    player_images_tensor.append(output_dict['image'])

            # Convert to PIL for voting grid
            pil_images = tensor_images_to_pil(torch.stack(player_images_tensor))

            # ── Phase 2: Voting (Bagel understanding mode) ───────
            grid_image = build_voting_grid(pil_images, cell_size=config.resolution)
            game_votes = []

            with torch.no_grad():
                for pid in range(1, num_players + 1):
                    vote_prompt = game_generator.format_voting_prompt(game_data, player_id=pid)
                    vote_text = run_bagel_vote(
                        inferencer, grid_image, vote_prompt,
                        max_tokens=max_vote_tokens,
                    )
                    vote_info = game_generator.extract_vote(vote_text)
                    game_votes.append(vote_info)

            # ── Phase 3: Compute rewards ─────────────────────────
            game_outcome = game_generator.calculate_game_rewards(game_data, game_votes)
            gen_rewards = game_generator.compute_generation_rewards(game_outcome)

            if game_outcome['spy_caught']:
                spy_caught_count += 1
            total_games += 1

            all_game_trajs.append(player_trajs)
            all_game_rewards.append(gen_rewards)
            all_game_images.append(pil_images)

        # ── Group-relative advantages ────────────────────────────
        flat_rewards = [r for rw in all_game_rewards for r in rw]
        advantages = compute_group_advantages(flat_rewards).to(accelerator.device)

        # ── Log images & rewards ─────────────────────────────────
        if accelerator.is_main_process:
            spy_rate = spy_caught_count / max(total_games, 1)
            wandb.log({
                "epoch": epoch,
                "mean_reward": np.mean(flat_rewards),
                "spy_detection_rate": spy_rate,
                "reward_std": np.std(flat_rewards),
            }, step=global_step)

            # Log sample images every 5 epochs
            if epoch % 5 == 0 and all_game_images:
                with tempfile.TemporaryDirectory() as tmpdir:
                    imgs_to_log = []
                    game_data_last = game_generator.generate_game(
                        epoch * G, global_step)
                    for pid, img in enumerate(all_game_images[0]):
                        path = os.path.join(tmpdir, f"p{pid+1}.jpg")
                        img.save(path)
                        is_spy = "SPY" if pid + 1 == game_data_last['spy_player'] else "CIV"
                        reward = all_game_rewards[0][pid]
                        imgs_to_log.append(wandb.Image(
                            path, caption=f"P{pid+1}({is_spy}) R={reward:.2f}"))
                    wandb.log({"game_images": imgs_to_log}, step=global_step)

        # ── Training: per-player per-SDE-step backward ───────────
        transformer.train()
        # Set internal training flags correctly for Bagel
        transformer.module.training = False
        transformer.module.model.training = False
        if config.use_lora:
            transformer.module.model.model.training = False
            for layer in transformer.module.model.model.layers:
                layer.module.training = False
                layer.module.self_attn.training = False
        else:
            for layer in transformer.module.model.layers:
                layer.module.training = False
                layer.module.self_attn.training = False

        info = defaultdict(list)

        for inner_epoch in range(num_inner_epochs):
            adv_idx = 0
            for g in range(G):
                game_data = game_generator.generate_game(epoch * G + g, global_step)
                prompts = [
                    game_generator.format_generation_prompt_simple(game_data, pid)
                    for pid in range(1, num_players + 1)
                ]

                for pid in range(num_players):
                    ptraj = all_game_trajs[g][pid]

                    # Build per-step sample dict
                    latents = torch.stack(ptraj['all_latents'])   # [num_steps+1, ...]
                    log_probs = torch.stack(ptraj['all_log_probs'])  # [num_steps]
                    timesteps = ptraj['timesteps']  # [num_steps]

                    cur_sample = {
                        'timesteps': timesteps,
                        'latents': latents[:-1],
                        'prev_latents': latents[1:],
                        'log_probs': log_probs,
                        'advantages': advantages[adv_idx].unsqueeze(0).expand(
                            timesteps.shape[0], -1) if advantages.dim() > 0
                            else advantages[adv_idx].unsqueeze(0).unsqueeze(0).expand(
                            timesteps.shape[0], 1),
                    }
                    # dtimesteps
                    cur_sample['dtimesteps'] = torch.cat([
                        timesteps[:-1] - timesteps[1:],
                        timesteps[-1:],
                    ])

                    with autocast():
                        output_dict = inferencer(
                            text=prompts[pid],
                            noise_level=config.sample.noise_level,
                            learn=True,
                            sample=cur_sample,
                            grpo_config=config,
                            accelerator=accelerator,
                            optimizer=optimizer,
                            transformer=transformer,
                            num_timesteps=config.sample.num_steps,
                            cfg_text_scale=config.sample.guidance_scale,
                            **inference_hyper,
                        )

                    info["clipfrac"].append(output_dict["clipfrac"])
                    info["policy_loss"].append(output_dict["policy_loss"])
                    info["kl_loss"].append(output_dict["kl_loss"])
                    info["loss"].append(output_dict["loss"])
                    adv_idx += 1

                    if accelerator.sync_gradients:
                        log_info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        log_info = accelerator.reduce(log_info, reduction="mean")
                        log_info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(log_info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
