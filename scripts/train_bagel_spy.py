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
from flow_grpo.spy_game_data import SpyGameDataGenerator, TextFileGameDataGenerator
from flow_grpo.spy_game_reward import (
    build_voting_grid, build_voting_grid_tensor, grid_tensor_to_pil,
    tensor_images_to_pil, run_bagel_vote, compute_group_advantages,
)

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model
from flow_grpo.fsdp_utils import save_fsdp_checkpoint
from huggingface_hub import snapshot_download


# ─── Image diversity metrics (GPU-native, single sync) ──────────────────────

@torch.no_grad()
def compute_image_diversity_metrics(player_images_tensor: list) -> dict:
    """Compute diversity metrics entirely on GPU, sync once at end."""
    if len(player_images_tensor) < 2:
        return {}

    images = torch.stack(player_images_tensor).float()  # [N, 3, H, W]
    N = images.shape[0]
    flat = images.reshape(N, -1)  # [N, D]

    # All pairwise: use matrix operations instead of loops
    # Pairwise MSE via ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    sq_norms = (flat ** 2).sum(dim=1)  # [N]
    dots = flat @ flat.T  # [N, N]
    pairwise_sq_diff = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * dots  # [N, N]
    D = flat.shape[1]
    # Extract upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(N, N, device=flat.device, dtype=torch.bool), diagonal=1)
    mean_pairwise_mse = (pairwise_sq_diff[mask] / D).mean()

    # Per-pixel std
    pixel_std = images.std(dim=0).mean()

    # Cosine similarity via normalized dot product matrix
    norms = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
    cos_matrix = norms @ norms.T  # [N, N]
    mean_cosine_sim = cos_matrix[mask].mean()

    # Histogram intersection: batch all pairs
    gray = images.mean(dim=1).reshape(N, -1)  # [N, H*W]
    # Vectorized histc: bin each image
    bins = 64
    hist_all = torch.zeros(N, bins, device=flat.device)
    for i in range(N):
        hist_all[i] = torch.histc(gray[i], bins=bins, min=0.0, max=1.0)
    hist_all = hist_all / (hist_all.sum(dim=1, keepdim=True) + 1e-8)
    # Pairwise intersection
    hist_intersections = []
    for i in range(N):
        for j in range(i + 1, N):
            hist_intersections.append(torch.minimum(hist_all[i], hist_all[j]).sum())
    mean_hist_sim = torch.stack(hist_intersections).mean()

    # Single GPU→CPU sync for all 4 metrics
    results = torch.stack([mean_pairwise_mse, pixel_std, mean_cosine_sim, mean_hist_sim])
    r = results.cpu().numpy()

    return {
        "img_pairwise_mse": float(r[0]),
        "img_pixel_std": float(r[1]),
        "img_cosine_sim": float(r[2]),
        "img_hist_sim": float(r[3]),
    }


@torch.no_grad()
def compute_spy_vs_civ_divergence(player_images_tensor: list,
                                   spy_idx: int) -> dict:
    """Compute spy-vs-civilian divergence on GPU, single sync."""
    if len(player_images_tensor) < 2:
        return {}

    images = torch.stack(player_images_tensor).float()  # [N, 3, H, W]
    N = images.shape[0]
    flat = images.reshape(N, -1)  # [N, D]
    D = flat.shape[1]

    spy_flat = flat[spy_idx]  # [D]
    civ_mask = torch.arange(N, device=flat.device) != spy_idx
    civ_flat = flat[civ_mask]  # [N-1, D]

    # Spy vs each civilian MSE
    spy_civ_mse = ((spy_flat.unsqueeze(0) - civ_flat) ** 2).mean(dim=1)  # [N-1]
    mean_spy_civ_mse = spy_civ_mse.mean()

    # Civilian vs civilian MSE
    Nc = civ_flat.shape[0]
    if Nc >= 2:
        civ_sq = (civ_flat ** 2).sum(dim=1)
        civ_dots = civ_flat @ civ_flat.T
        civ_pw = civ_sq.unsqueeze(1) + civ_sq.unsqueeze(0) - 2 * civ_dots
        civ_mask_tri = torch.triu(torch.ones(Nc, Nc, device=flat.device, dtype=torch.bool), diagonal=1)
        mean_civ_civ_mse = (civ_pw[civ_mask_tri] / D).mean()
    else:
        mean_civ_civ_mse = torch.tensor(0.0, device=flat.device)

    # Ratio
    mse_ratio = mean_spy_civ_mse / (mean_civ_civ_mse + 1e-8)

    # Cosine: spy vs civilian mean
    civ_mean = civ_flat.mean(dim=0)
    cos_sim = F.cosine_similarity(spy_flat.unsqueeze(0), civ_mean.unsqueeze(0))

    # Single sync
    results = torch.stack([mean_spy_civ_mse, mean_civ_civ_mse, mse_ratio, cos_sim.squeeze()])
    r = results.cpu().numpy()

    return {
        "spy_vs_civ_mse": float(r[0]),
        "civ_vs_civ_mse": float(r[1]),
        "spy_civ_mse_ratio": float(r[2]),
        "spy_vs_civ_cosine": float(r[3]),
    }


def compute_vote_statistics(all_game_votes: list, all_game_data: list) -> dict:
    """Compute detailed voting statistics across all games.

    Args:
        all_game_votes: List of [G] lists, each containing N vote dicts.
        all_game_data: List of [G] game_data dicts.

    Returns:
        Dict of voting statistics.
    """
    total_votes = 0
    valid_votes = 0
    correct_votes = 0
    na_votes = 0
    self_votes = 0  # players voting for themselves
    spy_self_votes = 0  # spy voting for themselves (very bad)

    for game_votes, game_data in zip(all_game_votes, all_game_data):
        spy = game_data["spy_player"]
        for pid_0, vote_info in enumerate(game_votes):
            pid = pid_0 + 1
            total_votes += 1
            if vote_info is None:
                continue
            voted = vote_info.get("voted_spy")
            if voted == "N/A":
                na_votes += 1
                valid_votes += 1
            elif isinstance(voted, int):
                valid_votes += 1
                if voted == spy:
                    correct_votes += 1
                if voted == pid:
                    self_votes += 1
                    if pid == spy:
                        spy_self_votes += 1

    return {
        "vote_valid_rate": valid_votes / max(total_votes, 1),
        "vote_accuracy": correct_votes / max(valid_votes, 1),
        "vote_na_rate": na_votes / max(total_votes, 1),
        "vote_self_rate": self_votes / max(total_votes, 1),
        "spy_self_vote_rate": spy_self_votes / max(len(all_game_data), 1),
    }

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

    # For spy game: gradient_accumulation matches train_bagel.py pattern
    # Each call to inferencer(learn=True) internally does sde_window_size backward+step calls.
    # We want one real optimizer update per: G games × N players × num_inner_epochs
    # (matching SPY-UMM: all player trajectories across all games accumulated before update)
    spy_cfg = config.spy_game
    num_players = spy_cfg.num_players
    G = spy_cfg.group_size
    num_inner_epochs = config.train.num_inner_epochs

    # grad_accum = total backward calls per real optimizer step
    # = G games × N players × sde_window_size (per-SDE-step backward inside generate_image_learn)
    grad_accum = config.train.gradient_accumulation_steps * num_players * config.sample.sde_window_size

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=grad_accum,
    )
    if hasattr(accelerator.state, 'fsdp_plugin') and accelerator.state.fsdp_plugin is not None:
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
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

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
    prompt_type = spy_cfg.get('prompt_type', 'clevr')
    if prompt_type == 'clevr':
        game_generator = SpyGameDataGenerator(
            num_players=num_players,
            num_objects_min=spy_cfg.get('num_objects_min', 3),
            num_objects_max=spy_cfg.get('num_objects_max', 6),
            num_to_modify=spy_cfg.get('num_to_modify', 2),
            max_props_per_obj=spy_cfg.get('max_props_per_obj', 1),
        )
    else:
        # Text-file based: ocr, geneval, pickscore
        game_generator = TextFileGameDataGenerator(
            dataset_path=config.dataset,
            split='train',
            prompt_type=prompt_type,
            num_players=num_players,
        )
    max_vote_tokens = spy_cfg.get('max_vote_tokens', 512)
    num_inner_epochs = config.train.num_inner_epochs
    use_role_advantage = spy_cfg.get('use_role_advantage', False)
    logger.info(f"  Game data: prompt_type={prompt_type}")
    logger.info(f"  Role advantage: {'ENABLED' if use_role_advantage else 'DISABLED'}")

    # ==================== TRAINING LOOP ====================
    logger.info("***** Running Bagel Spy-Civ Flow-GRPO Training *****")
    logger.info(f"  Players={num_players}, G={G}, inner_epochs={num_inner_epochs}")
    logger.info(f"  SDE window={config.sample.sde_window_size}")
    logger.info(f"  Gradient accumulation steps (Accelerator)={grad_accum}")
    logger.info(f"    = {config.train.gradient_accumulation_steps}(config) "
                f"x {num_players}(players) x {config.sample.sde_window_size}(sde)")
    logger.info(f"  Backward calls per real optimizer step: {grad_accum}")
    logger.info(f"  Total trajectories per epoch: {G * num_players} "
                f"({G} games x {num_players} players)")

    global_step = 0
    spy_caught_count = 0
    total_games = 0
    batch_time_start = time.time()

    for epoch in range(config.num_epochs):
        # ── Eval & Checkpoint ────────────────────────────────────
        if not config.debug and epoch > 0 and epoch % config.save_freq == 0:
            save_fsdp_checkpoint(config.save_dir, transformer, global_step,
                                 accelerator.process_index)

        # ── Sampling: G games ────────────────────────────────────
        transformer.eval()
        all_game_trajs = []       # [G][N] each is dict with latents, log_probs, etc.
        all_game_rewards = []     # [G] each is list of N rewards
        all_game_images_t = []    # [G][N] tensor images (stay on GPU)
        all_game_votes = []       # [G][N] vote dicts for vote stats
        all_game_data_list = []   # [G] game_data dicts for analysis
        all_game_outcomes = []    # [G] game outcome dicts
        epoch_spy_rewards = []
        epoch_civ_rewards = []
        t_gen_total, t_vote_total = 0.0, 0.0

        for g in tqdm(range(G), desc=f"Epoch {epoch}: games",
                      disable=not accelerator.is_local_main_process):

            game_data = game_generator.generate_game(epoch * G + g, global_step)
            prompts = [
                game_generator.format_generation_prompt_simple(game_data, pid)
                for pid in range(1, num_players + 1)
            ]

            # ── Phase 1: Each player generates an image ──────────
            # Use unwrapped model for sampling (DDP wrapping breaks .model access)
            wrapped_lm_gen = model.language_model
            model.language_model = accelerator.unwrap_model(transformer)
            t_gen_start = time.time()
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

            model.language_model = wrapped_lm_gen  # restore for training later
            t_gen_total += time.time() - t_gen_start

            # Build voting grid on GPU (no CPU transfer), convert to PIL once
            t_vote_start = time.time()
            grid_tensor = build_voting_grid_tensor(player_images_tensor,
                                                    cell_size=config.resolution)
            grid_pil = grid_tensor_to_pil(grid_tensor, num_players=num_players,
                                           cell_size=config.resolution)

            # ── Phase 2: Voting (Bagel understanding mode) ───────
            # Temporarily use unwrapped model for voting (DDP/FSDP wrapping
            # breaks generate_text() which accesses .model.embed_tokens directly)
            wrapped_lm = model.language_model
            model.language_model = accelerator.unwrap_model(transformer)
            game_votes = []
            with torch.no_grad():
                for pid in range(1, num_players + 1):
                    vote_prompt = game_generator.format_voting_prompt(game_data, player_id=pid)
                    vote_text = run_bagel_vote(
                        inferencer, grid_pil, vote_prompt,
                        max_tokens=max_vote_tokens,
                    )
                    vote_info = game_generator.extract_vote(vote_text)
                    game_votes.append(vote_info)
            model.language_model = wrapped_lm  # restore wrapped model

            t_vote_total += time.time() - t_vote_start

            # ── Phase 3: Compute rewards ─────────────────────────
            game_outcome = game_generator.calculate_game_rewards(game_data, game_votes)
            gen_rewards = game_generator.compute_generation_rewards(game_outcome)

            if game_outcome['spy_caught']:
                spy_caught_count += 1
            total_games += 1

            # Track spy vs civilian rewards separately
            spy_pid = game_data['spy_player']
            epoch_spy_rewards.append(gen_rewards[spy_pid - 1])
            for i, r in enumerate(gen_rewards):
                if i != spy_pid - 1:
                    epoch_civ_rewards.append(r)

            # Update EMA baselines
            civ_rs = [gen_rewards[i] for i in range(num_players) if i != spy_pid - 1]
            game_generator.update_baselines(gen_rewards[spy_pid - 1],
                                            np.mean(civ_rs) if civ_rs else 0.0)

            all_game_trajs.append(player_trajs)
            all_game_rewards.append(gen_rewards)
            all_game_images_t.append(player_images_tensor)  # keep tensors on GPU
            all_game_votes.append(game_votes)
            all_game_data_list.append(game_data)
            all_game_outcomes.append(game_outcome)

        # ── Group-relative advantages ────────────────────────────
        if use_role_advantage:
            # Vision-Zero style: subtract per-role EMA baselines before normalization
            adjusted_rewards = []
            for g in range(G):
                spy_pid = all_game_data_list[g]['spy_player']
                adj = game_generator.apply_role_advantage(all_game_rewards[g], spy_pid)
                adjusted_rewards.append(adj)
            flat_rewards_raw = [r for rw in all_game_rewards for r in rw]
            flat_rewards = [r for rw in adjusted_rewards for r in rw]
        else:
            flat_rewards_raw = [r for rw in all_game_rewards for r in rw]
            flat_rewards = flat_rewards_raw
        advantages = compute_group_advantages(flat_rewards).to(accelerator.device)

        # ── Comprehensive Logging ────────────────────────────────
        if accelerator.is_main_process:
            batch_time = time.time() - batch_time_start
            batch_time_start = time.time()

            spy_rate = spy_caught_count / max(total_games, 1)

            # === Reward metrics ===
            reward_metrics = {
                "epoch": epoch,
                "mean_reward": np.mean(flat_rewards_raw),
                "reward_std": np.std(flat_rewards_raw),
                "reward_min": np.min(flat_rewards_raw),
                "reward_max": np.max(flat_rewards_raw),
                "spy_mean_reward": np.mean(epoch_spy_rewards) if epoch_spy_rewards else 0,
                "civ_mean_reward": np.mean(epoch_civ_rewards) if epoch_civ_rewards else 0,
                "advantage_mean": advantages.mean().item(),
                "advantage_std": advantages.std().item(),
                "advantage_abs_max": advantages.abs().max().item(),
            }

            # === Game metrics ===
            game_metrics = {
                "spy_detection_rate": spy_rate,
                "spy_detection_rate_epoch": sum(1 for o in all_game_outcomes if o['spy_caught']) / G,
                "ema_baseline_spy": game_generator.b_spy,
                "ema_baseline_civ": game_generator.b_civ,
                "ema_baseline_gap": game_generator.b_civ - game_generator.b_spy,
                "use_role_advantage": float(use_role_advantage),
            }
            if use_role_advantage:
                game_metrics["mean_adjusted_reward"] = np.mean(flat_rewards)
                game_metrics["adjusted_reward_std"] = np.std(flat_rewards)

            # === Vote statistics ===
            vote_stats = compute_vote_statistics(all_game_votes, all_game_data_list)

            # === Image diversity metrics (averaged over all G games) ===
            diversity_metrics_accum = defaultdict(list)
            spy_div_metrics_accum = defaultdict(list)
            for g_idx in range(G):
                div_m = compute_image_diversity_metrics(all_game_images_t[g_idx])
                for k, v in div_m.items():
                    diversity_metrics_accum[k].append(v)

                spy_idx = all_game_data_list[g_idx]['spy_player'] - 1
                spy_div_m = compute_spy_vs_civ_divergence(
                    all_game_images_t[g_idx], spy_idx)
                for k, v in spy_div_m.items():
                    spy_div_metrics_accum[k].append(v)

            diversity_metrics = {k: np.mean(v) for k, v in diversity_metrics_accum.items()}
            spy_div_metrics = {k: np.mean(v) for k, v in spy_div_metrics_accum.items()}

            # === Timing (per-phase breakdown) ===
            timing_metrics = {
                "batch_time": batch_time,
                "games_per_sec": G / max(batch_time, 0.01),
                "time_generation": t_gen_total,
                "time_voting": t_vote_total,
                "time_other": batch_time - t_gen_total - t_vote_total,
                "pct_generation": t_gen_total / max(batch_time, 0.01) * 100,
                "pct_voting": t_vote_total / max(batch_time, 0.01) * 100,
            }

            # Merge all and log
            all_metrics = {
                **reward_metrics, **game_metrics, **vote_stats,
                **diversity_metrics, **spy_div_metrics, **timing_metrics,
            }
            wandb.log(all_metrics, step=global_step)

            # Log to console
            logger.info(
                f"Epoch {epoch} | "
                f"R={np.mean(flat_rewards):.3f}±{np.std(flat_rewards):.3f} "
                f"SpyR={np.mean(epoch_spy_rewards):.3f} CivR={np.mean(epoch_civ_rewards):.3f} | "
                f"SpyRate={spy_rate:.1%} VoteAcc={vote_stats.get('vote_accuracy', 0):.1%} | "
                f"CosSim={diversity_metrics.get('img_cosine_sim', 0):.3f} "
                f"SpyCivRatio={spy_div_metrics.get('spy_civ_mse_ratio', 0):.2f} | "
                f"{batch_time:.1f}s"
            )

            # Log sample images every 5 epochs (only convert to PIL here)
            if epoch % 5 == 0 and all_game_images_t:
                pil_for_log = tensor_images_to_pil(torch.stack(all_game_images_t[0]))
                with tempfile.TemporaryDirectory() as tmpdir:
                    imgs_to_log = []
                    gd = all_game_data_list[0]
                    for pid, img in enumerate(pil_for_log):
                        path = os.path.join(tmpdir, f"p{pid+1}.jpg")
                        img.save(path)
                        is_spy = "SPY" if pid + 1 == gd['spy_player'] else "CIV"
                        reward = all_game_rewards[0][pid]
                        vote = all_game_votes[0][pid]
                        vote_str = str(vote.get('voted_spy', '?')) if vote else 'INVALID'
                        imgs_to_log.append(wandb.Image(
                            path,
                            caption=f"P{pid+1}({is_spy}) R={reward:.2f} Vote={vote_str}"))
                    grid_path = os.path.join(tmpdir, "grid.jpg")
                    build_voting_grid(pil_for_log, cell_size=config.resolution).save(grid_path)
                    imgs_to_log.append(wandb.Image(grid_path, caption="Voting Grid"))
                    wandb.log({"game_images": imgs_to_log}, step=global_step)

        # ── Training: per-player per-SDE-step backward ───────────
        transformer.train()
        # Set internal training flags correctly for Bagel
        # Use unwrap_model to handle both single-GPU and FSDP cases
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.training = False
        unwrapped.model.training = False
        if config.use_lora:
            unwrapped.model.model.training = False
            for layer in unwrapped.model.model.layers:
                layer.training = False
                if hasattr(layer, 'self_attn'):
                    layer.self_attn.training = False
        else:
            for layer in unwrapped.model.layers:
                layer.training = False
                if hasattr(layer, 'self_attn'):
                    layer.self_attn.training = False

        info = defaultdict(list)

        # Pre-compute prompts for training (reuse from sampling, avoid recomputation)
        all_game_prompts = []
        for g in range(G):
            gd = all_game_data_list[g]
            all_game_prompts.append([
                game_generator.format_generation_prompt_simple(gd, pid)
                for pid in range(1, num_players + 1)
            ])

        for inner_epoch in range(num_inner_epochs):
            adv_idx = 0
            for g in range(G):
                prompts = all_game_prompts[g]

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
                    info["clipfrac_gt_one"].append(output_dict.get("clipfrac_gt_one", output_dict["clipfrac"]))
                    info["clipfrac_lt_one"].append(output_dict.get("clipfrac_lt_one", output_dict["clipfrac"]))
                    info["policy_loss"].append(output_dict["policy_loss"])
                    info["kl_loss"].append(output_dict["kl_loss"])
                    info["loss"].append(output_dict["loss"])
                    adv_idx += 1

                    if accelerator.sync_gradients:
                        log_info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        log_info = accelerator.reduce(log_info, reduction="mean")
                        log_info["abs_policy_loss"] = abs(log_info["policy_loss"].item())
                        log_info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(log_info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
