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

Architecture: b7a190a (cross-GPU player parallelism) + acceleration patches:
  - Remove accelerate dispatch hooks (10x voting speedup)
  - Deferred gradient sync (DDP ~12s → <1s)
  - Async logging (non-blocking)
  - DDP-compatible checkpoint save
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
    build_voting_grid, tensor_images_to_pil,
    run_bagel_vote_multi_image_repeated, compute_group_advantages,
)

import numpy as np
import torch
import torch.nn.functional as F
import wandb

import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor


# ─── Async logging helper ────────────────────────────────────────────────────
_log_executor = ThreadPoolExecutor(max_workers=1)

def _async_log_metrics(all_game_images_t, all_game_data_list, all_game_rewards,
                       all_game_votes, all_game_outcomes, flat_rewards_raw,
                       flat_rewards, advantages, epoch_spy_rewards, epoch_civ_rewards,
                       spy_caught_count, total_games, game_generator,
                       use_role_advantage, G, num_players, t_gen_total, t_vote_total,
                       batch_time, epoch, global_step, config):
    """Run all logging in background thread to avoid blocking GPU."""
    spy_rate = spy_caught_count / max(total_games, 1)

    reward_metrics = {
        "epoch": epoch,
        "mean_reward": np.mean(flat_rewards_raw),
        "reward_std": np.std(flat_rewards_raw),
        "reward_min": np.min(flat_rewards_raw),
        "reward_max": np.max(flat_rewards_raw),
        "spy_mean_reward": np.mean(epoch_spy_rewards) if epoch_spy_rewards else 0,
        "civ_mean_reward": np.mean(epoch_civ_rewards) if epoch_civ_rewards else 0,
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_abs_max": float(advantages.abs().max()),
    }

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

    vote_stats = compute_vote_statistics(all_game_votes, all_game_data_list)

    # Image diversity metrics (compute on CPU copies to avoid blocking GPU)
    diversity_metrics = {}
    spy_div_metrics = {}
    try:
        diversity_accum = defaultdict(list)
        spy_div_accum = defaultdict(list)
        for g_idx in range(G):
            imgs = all_game_images_t[g_idx]
            d = compute_image_diversity_metrics(imgs)
            for k, v in d.items():
                diversity_accum[k].append(v)
            spy_idx = all_game_data_list[g_idx]['spy_player'] - 1
            s = compute_spy_vs_civ_divergence(imgs, spy_idx)
            for k, v in s.items():
                spy_div_accum[k].append(v)
        diversity_metrics = {k: np.mean(v) for k, v in diversity_accum.items()}
        spy_div_metrics = {k: np.mean(v) for k, v in spy_div_accum.items()}
    except Exception:
        pass

    timing_metrics = {
        "batch_time": batch_time,
        "games_per_sec": G / max(batch_time, 0.01),
        "time_generation": t_gen_total,
        "time_voting": t_vote_total,
        "time_other": batch_time - t_gen_total - t_vote_total,
        "pct_generation": t_gen_total / max(batch_time, 0.01) * 100,
        "pct_voting": t_vote_total / max(batch_time, 0.01) * 100,
    }

    all_metrics = {
        **reward_metrics, **game_metrics, **vote_stats,
        **diversity_metrics, **spy_div_metrics, **timing_metrics,
    }
    wandb.log(all_metrics, step=global_step)

    logger.info(
        f"Epoch {epoch} | "
        f"R={np.mean(flat_rewards_raw):.3f}±{np.std(flat_rewards_raw):.3f} "
        f"SpyR={np.mean(epoch_spy_rewards) if epoch_spy_rewards else 0:.3f} "
        f"CivR={np.mean(epoch_civ_rewards) if epoch_civ_rewards else 0:.3f} | "
        f"SpyRate={spy_rate:.1%} VoteAcc={vote_stats.get('vote_accuracy', 0):.1%} | "
        f"CosSim={diversity_metrics.get('img_cosine_sim', 0):.3f} "
        f"SpyCivRatio={spy_div_metrics.get('spy_civ_mse_ratio', 0):.2f} | "
        f"{batch_time:.1f}s"
    )

    if epoch % 5 == 0 and all_game_images_t:
        try:
            pil_for_log = tensor_images_to_pil(torch.stack(all_game_images_t[0]))
            with tempfile.TemporaryDirectory() as tmpdir:
                imgs_to_log = []
                gd = all_game_data_list[0]
                # Count God votes per player for caption
                god_vote_summary = {}
                for v in all_game_votes[0]:
                    if v and isinstance(v.get('voted_spy'), int):
                        pid_v = v['voted_spy']
                        god_vote_summary[pid_v] = god_vote_summary.get(pid_v, 0) + 1
                for pid, img in enumerate(pil_for_log):
                    path = os.path.join(tmpdir, f"p{pid+1}.jpg")
                    img.save(path)
                    is_spy = "SPY" if pid + 1 == gd['spy_player'] else "CIV"
                    reward = all_game_rewards[0][pid]
                    god_votes_received = god_vote_summary.get(pid + 1, 0)
                    imgs_to_log.append(wandb.Image(
                        path, caption=f"P{pid+1}({is_spy}) R={reward:.2f} GodVotes={god_votes_received}"))
                grid_path = os.path.join(tmpdir, "grid.jpg")
                build_voting_grid(pil_for_log, cell_size=config.resolution).save(grid_path)
                imgs_to_log.append(wandb.Image(grid_path, caption="Voting Grid"))
                wandb.log({"game_images": imgs_to_log}, step=global_step)
        except Exception:
            pass


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
    sq_norms = (flat ** 2).sum(dim=1)  # [N]
    dots = flat @ flat.T  # [N, N]
    pairwise_sq_diff = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * dots  # [N, N]
    D = flat.shape[1]
    mask = torch.triu(torch.ones(N, N, device=flat.device, dtype=torch.bool), diagonal=1)
    mean_pairwise_mse = (pairwise_sq_diff[mask] / D).mean()

    pixel_std = images.std(dim=0).mean()

    norms = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
    cos_matrix = norms @ norms.T  # [N, N]
    mean_cosine_sim = cos_matrix[mask].mean()

    gray = images.mean(dim=1).reshape(N, -1)  # [N, H*W]
    bins = 64
    hist_all = torch.zeros(N, bins, device=flat.device)
    for i in range(N):
        hist_all[i] = torch.histc(gray[i], bins=bins, min=0.0, max=1.0)
    hist_all = hist_all / (hist_all.sum(dim=1, keepdim=True) + 1e-8)
    hist_intersections = []
    for i in range(N):
        for j in range(i + 1, N):
            hist_intersections.append(torch.minimum(hist_all[i], hist_all[j]).sum())
    mean_hist_sim = torch.stack(hist_intersections).mean()

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

    spy_civ_mse = ((spy_flat.unsqueeze(0) - civ_flat) ** 2).mean(dim=1)  # [N-1]
    mean_spy_civ_mse = spy_civ_mse.mean()

    Nc = civ_flat.shape[0]
    if Nc >= 2:
        civ_sq = (civ_flat ** 2).sum(dim=1)
        civ_dots = civ_flat @ civ_flat.T
        civ_pw = civ_sq.unsqueeze(1) + civ_sq.unsqueeze(0) - 2 * civ_dots
        civ_mask_tri = torch.triu(torch.ones(Nc, Nc, device=flat.device, dtype=torch.bool), diagonal=1)
        mean_civ_civ_mse = (civ_pw[civ_mask_tri] / D).mean()
    else:
        mean_civ_civ_mse = torch.tensor(0.0, device=flat.device)

    mse_ratio = mean_spy_civ_mse / (mean_civ_civ_mse + 1e-8)
    cos_sim = F.cosine_similarity(spy_flat.unsqueeze(0), civ_flat.mean(dim=0).unsqueeze(0))

    results = torch.stack([mean_spy_civ_mse, mean_civ_civ_mse, mse_ratio, cos_sim.squeeze()])
    r = results.cpu().numpy()

    return {
        "spy_vs_civ_mse": float(r[0]),
        "civ_vs_civ_mse": float(r[1]),
        "spy_civ_mse_ratio": float(r[2]),
        "spy_vs_civ_cosine": float(r[3]),
    }


def compute_vote_statistics(all_game_votes: list, all_game_data: list) -> dict:
    """Compute God-judge voting statistics across all games.

    all_game_votes: list of lists, each inner list contains K God vote dicts
                    with {'voted_spy': int}.
    """
    total_votes = 0
    valid_votes = 0
    correct_votes = 0
    na_votes = 0

    for game_votes, game_data in zip(all_game_votes, all_game_data):
        spy = game_data["spy_player"]
        for vote_info in game_votes:
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

    return {
        "vote_valid_rate": valid_votes / max(total_votes, 1),
        "vote_accuracy": correct_votes / max(valid_votes, 1),
        "vote_na_rate": na_votes / max(total_votes, 1),
    }

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

    spy_cfg = config.spy_game
    num_players = spy_cfg.num_players
    G = spy_cfg.group_size
    num_inner_epochs = config.train.num_inner_epochs

    # NOTE: gradient_accumulation_steps is set here for Accelerator init but is
    # overridden to 999999 during training (see [ACCEL PATCH 2] below).
    # Actual gradient sync is done manually via no_sync + all_reduce.
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=1,  # placeholder; overridden by manual sync
    )
    if hasattr(accelerator.state, 'fsdp_plugin') and accelerator.state.fsdp_plugin is not None:
        accelerator.state.fsdp_plugin.activation_checkpointing = config.activation_checkpointing
        accelerator.state.fsdp_plugin.transformer_cls_names_to_wrap = ['Qwen2MoTDecoderLayer']

    n_gpus = accelerator.num_processes
    import math as _math

    if accelerator.is_main_process:
        wandb_resume_id = getattr(config, 'wandb_resume_id', None)
        if wandb_resume_id:
            wandb.init(project="flow_grpo_spy", id=wandb_resume_id, resume="must",
                       name=config.run_name, config=config.to_dict())
        else:
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

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_local_dir, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_local_dir, "ae.safetensors"))

    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
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

    # Load via load_checkpoint_and_dispatch (handles meta tensors correctly),
    # then remove hooks to avoid ~10x overhead on autoregressive generation.
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_local_dir, "ema.safetensors"),
        device_map={"": f"cuda:{accelerator.local_process_index}"},
        offload_buffers=False,
        dtype=inference_dtype,
        force_hooks=True,
        offload_folder="/adialab/usr/shadabk/MedUMM/.offload"
    )
    # [ACCEL PATCH 1] Remove accelerate dispatch hooks — they add ~10x overhead
    # to autoregressive generation (1041 AlignDevicesHook calls per forward).
    # Model is fully on one GPU so hooks are unnecessary.
    from accelerate.hooks import remove_hook_from_module
    for module in model.modules():
        remove_hook_from_module(module)
    model = model.eval()
    torch.cuda.empty_cache()

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
    logger.info(f"  God-judge voting: K={spy_cfg.get('god_vote_K', 8)} votes per game")
    logger.info(f"  Role advantage: {'ENABLED' if use_role_advantage else 'DISABLED'}")

    # ==================== TRAINING LOOP ====================
    logger.info("***** Running Bagel Spy-Civ Flow-GRPO Training *****")
    logger.info(f"  Players={num_players}, G={G}, inner_epochs={num_inner_epochs}")
    logger.info(f"  SDE window={config.sample.sde_window_size}")
    games_per_rank = _math.ceil(G / n_gpus)
    logger.info(f"  GPUs={n_gpus}, games_per_rank={games_per_rank} (sequential gen per game)")
    logger.info(f"  Gradient sync: manual no_sync + all_reduce AVG")
    logger.info(f"  Backward calls per rank per inner epoch: "
                f"{games_per_rank}(games/rank) x {num_players}(players) x {config.sample.sde_window_size}(sde) "
                f"= {games_per_rank * num_players * config.sample.sde_window_size}")
    logger.info(f"  Total trajectories per epoch: {G * num_players} "
                f"({G} games x {num_players} players)")

    global_step = 0
    start_epoch = 0
    spy_caught_count = 0
    total_games = 0

    # ── Resume from checkpoint ──────────────────────────────────
    resume_from = getattr(config, 'resume_from', None)
    if resume_from and os.path.isdir(resume_from):
        ckpt_file = os.path.join(resume_from, "model.safetensors")
        if os.path.exists(ckpt_file):
            from safetensors.torch import load_file as _load_file
            ckpt_state = _load_file(ckpt_file, device=str(accelerator.device))
            unwrapped = accelerator.unwrap_model(transformer)
            msg = unwrapped.load_state_dict(ckpt_state, strict=False)
            logger.info(f"Resumed from {resume_from}")
            logger.info(f"  Loaded {len(ckpt_state)} tensors, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
            del ckpt_state
            torch.cuda.empty_cache()
            # Recover global_step from checkpoint dir name (e.g. checkpoint-360)
            ckpt_name = os.path.basename(resume_from.rstrip('/'))
            if '-' in ckpt_name:
                start_epoch = int(ckpt_name.split('-')[-1])
                global_step = start_epoch
                logger.info(f"  Resuming from epoch/step {start_epoch}")
        else:
            logger.warning(f"Resume checkpoint not found: {ckpt_file}, training from scratch")

    batch_time_start = time.time()

    for epoch in range(start_epoch, config.num_epochs):
        # Free GPU memory from previous epoch
        torch.cuda.empty_cache()

        # ── Eval & Checkpoint ────────────────────────────────────
        # [ACCEL PATCH 4] DDP-compatible checkpoint save (replaces save_fsdp_checkpoint)
        if not config.debug and epoch > 0 and epoch % config.save_freq == 0:
            if accelerator.is_main_process:
                from safetensors.torch import save_file as _save_file
                save_path = os.path.join(config.save_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                unwrapped = accelerator.unwrap_model(transformer)
                state_dict = {k: v.cpu() for k, v in unwrapped.state_dict().items()
                              if 'moe_gen' in k}
                _save_file(state_dict, os.path.join(save_path, "model.safetensors"))
                logger.info(f"Checkpoint saved: {save_path} ({len(state_dict)} tensors)")
                del state_dict
            accelerator.wait_for_everyone()

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

        # Use unwrapped model for all inference (DDP wrapping breaks .model access)
        wrapped_lm_save = model.language_model
        model.language_model = accelerator.unwrap_model(transformer)

        rank = accelerator.process_index
        n_gpus = accelerator.num_processes

        # ================================================================
        # PHASE 1: Sequential generation — each player sees previous images
        # Parallelism: each GPU handles a subset of GAMES (not players).
        # Within each game, players generate sequentially: 1 → 2 → 3 → 4.
        # ================================================================
        t_gen_start = time.time()
        all_my_trajs = []   # [G] dict mapping pid -> traj
        all_my_images = []  # [G] dict mapping pid -> image tensor

        # Pre-generate all game data
        for g in range(G):
            all_game_data_list.append(game_generator.generate_game(epoch * G + g, global_step))

        # Each GPU handles its subset of games
        my_game_ids_gen = [g for g in range(G) if g % n_gpus == rank]

        # Initialize per-game storage for all G games
        for g in range(G):
            all_my_trajs.append({})
            all_my_images.append({})

        with autocast():
            for g in my_game_ids_gen:
                game_data = all_game_data_list[g]
                game_pil_images = []  # Accumulate PIL images for previous players

                for pid in range(num_players):
                    player_id = pid + 1  # 1-indexed

                    # Build interleaved input: [prev_label, prev_img, ..., prompt]
                    input_list = game_generator.build_generation_input_list(
                        game_data, player_id, game_pil_images)

                    with torch.no_grad():
                        output_dict = inferencer.interleave_inference(
                            input_list,
                            noise_level=config.sample.noise_level,
                            grpo_config=config,
                            accelerator=accelerator,
                            num_timesteps=config.sample.num_steps,
                            cfg_text_scale=config.sample.guidance_scale,
                            **inference_hyper,
                        )
                    # output_dict is a single item (the generated image dict)
                    result = output_dict[0] if isinstance(output_dict, list) else output_dict

                    all_my_trajs[g][pid] = {
                        'image': result['image'],
                        'all_latents': result['all_latents'],
                        'all_log_probs': result['all_log_probs'],
                        'timesteps': result['timesteps'],
                    }
                    all_my_images[g][pid] = result['image']

                    # Convert to PIL and accumulate for next player's context
                    pil_img = tensor_images_to_pil(result['image'].unsqueeze(0).float())[0]
                    game_pil_images.append(pil_img)

        # SINGLE all_reduce for ALL G games' images at once: [G, N, 3, H, W]
        # Find image shape from any generated image
        img_shape = None
        for g in my_game_ids_gen:
            img_shape = list(all_my_images[g][0].shape)
            break
        # Broadcast img_shape from rank that has it
        if img_shape is None:
            img_shape = [3, config.resolution, config.resolution]  # fallback
        all_images_packed = torch.zeros(G, num_players, *img_shape,
                                        device=accelerator.device, dtype=torch.bfloat16)
        for g in my_game_ids_gen:
            for pid in range(num_players):
                all_images_packed[g, pid] = all_my_images[g][pid]
        torch.distributed.all_reduce(all_images_packed, op=torch.distributed.ReduceOp.SUM)

        t_gen_total = time.time() - t_gen_start

        # ================================================================
        # BATCHED PHASE 2: God-judge voting (Vision-Zero style)
        # One unbiased God-judge votes K times per game. No player voting.
        # Each GPU handles a subset of games. Single all_reduce at end.
        # ================================================================
        t_vote_start = time.time()
        god_vote_K = spy_cfg.get('god_vote_K', 8)

        # Each GPU handles a subset of games for voting
        my_game_ids = [g for g in range(G) if g % n_gpus == rank]

        # Collect vote_counts per game: {player_id: num_god_votes_received}
        # Pack as [G, num_players] tensor (vote counts, not individual votes)
        all_vote_count_tensors = torch.zeros(G, num_players,
                                              device=accelerator.device, dtype=torch.long)

        with torch.no_grad():
            for g in my_game_ids:
                game_data = all_game_data_list[g]
                player_images_g = [all_images_packed[g, pid] for pid in range(num_players)]

                # Convert individual player image tensors to PIL for VLM voting
                player_pils = tensor_images_to_pil(
                    torch.stack(player_images_g).float())

                # God-judge prompt (player_id=None → unbiased perspective)
                god_prompt = game_generator.format_voting_prompt(
                    game_data, player_id=None,
                    god_sees_description=spy_cfg.get('god_sees_description', False))

                # K repeated votes with cached ViT encoding (encode images once)
                vote_texts = run_bagel_vote_multi_image_repeated(
                    inferencer, player_pils, god_prompt,
                    num_generations=god_vote_K,
                    max_tokens=max_vote_tokens,
                )

                # Count votes per player
                for vtext in vote_texts:
                    vote_info = game_generator.extract_vote(vtext)
                    if vote_info and isinstance(vote_info.get('voted_spy'), int):
                        voted_pid = vote_info['voted_spy']
                        if 1 <= voted_pid <= num_players:
                            all_vote_count_tensors[g, voted_pid - 1] += 1

        # SINGLE all_reduce: SUM vote counts across GPUs
        # (each game is handled by exactly one GPU, others have 0)
        torch.distributed.all_reduce(all_vote_count_tensors, op=torch.distributed.ReduceOp.SUM)

        t_vote_total = time.time() - t_vote_start

        # ================================================================
        # Phase 3: Compute rewards for all G games (CPU, fast)
        # ================================================================
        for g in range(G):
            game_data = all_game_data_list[g]
            player_images_tensor = [all_images_packed[g, pid] for pid in range(num_players)]

            # Build vote_counts dict from God-judge vote count tensor
            vote_counts = {pid + 1: all_vote_count_tensors[g, pid].item()
                           for pid in range(num_players)}

            # Build game_outcome compatible with compute_generation_rewards
            spy_pid = game_data['spy_player']
            spy_caught = vote_counts[spy_pid] == max(vote_counts.values())
            game_outcome = {
                "player_rewards": {pid + 1: 0.0 for pid in range(num_players)},
                "spy_caught": spy_caught,
                "vote_counts": vote_counts,
                "spy_player": spy_pid,
            }

            # Also build game_votes list for vote statistics logging
            # (expand vote_counts into a flat list of K individual votes for stats)
            game_votes = []
            for pid in range(num_players):
                cnt = vote_counts[pid + 1]
                for _ in range(cnt):
                    game_votes.append({'voted_spy': pid + 1})

            gen_rewards = game_generator.compute_generation_rewards(
                game_outcome,
                beta=spy_cfg.reward_beta,
                lambda_param=spy_cfg.reward_lambda,
            )

            if game_outcome['spy_caught']:
                spy_caught_count += 1
            total_games += 1

            spy_pid = game_data['spy_player']
            epoch_spy_rewards.append(gen_rewards[spy_pid - 1])
            for i, r in enumerate(gen_rewards):
                if i != spy_pid - 1:
                    epoch_civ_rewards.append(r)

            # Build player_trajs for training
            player_trajs = [None] * num_players
            for pid in range(num_players):
                if pid in all_my_trajs[g]:
                    player_trajs[pid] = all_my_trajs[g][pid]
                else:
                    player_trajs[pid] = {'image': player_images_tensor[pid]}

            all_game_trajs.append(player_trajs)
            all_game_rewards.append(gen_rewards)
            all_game_images_t.append(player_images_tensor)
            all_game_votes.append(game_votes)
            all_game_outcomes.append(game_outcome)

        # ── Group-relative advantages ────────────────────────────
        # First apply role advantage using PREVIOUS epoch's baseline (before update)
        if use_role_advantage:
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

        # NOW update baselines with current epoch's data (for next epoch)
        for g in range(G):
            spy_pid = all_game_data_list[g]['spy_player']
            spy_r = all_game_rewards[g][spy_pid - 1]
            civ_rs = [all_game_rewards[g][i] for i in range(num_players) if i != spy_pid - 1]
            game_generator.update_baselines(spy_r, sum(civ_rs) / len(civ_rs) if civ_rs else 0.0)

        # Vision-Zero: role-adjusted rewards → group-level mean/std normalization
        advantages = compute_group_advantages(flat_rewards).to(accelerator.device)

        # ── [ACCEL PATCH 3] Async Logging (non-blocking, runs in background thread) ──
        if accelerator.is_main_process:
            batch_time = time.time() - batch_time_start
            batch_time_start = time.time()

            # Copy tensors to CPU for async logging
            images_for_log = [[t.cpu() for t in game_imgs] for game_imgs in all_game_images_t]
            adv_cpu = advantages.cpu()

            _log_executor.submit(
                _async_log_metrics,
                images_for_log, list(all_game_data_list), list(all_game_rewards),
                list(all_game_votes), list(all_game_outcomes), list(flat_rewards_raw),
                list(flat_rewards), adv_cpu, list(epoch_spy_rewards), list(epoch_civ_rewards),
                spy_caught_count, total_games, game_generator,
                use_role_advantage, G, num_players, t_gen_total, t_vote_total,
                batch_time, epoch, global_step, config,
            )

        # ── Training: per-player per-SDE-step backward ───────────
        transformer.train()
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

        # Pre-compute PIL images of previous players for each game (for training context)
        # all_images_packed is [G, N, 3, H, W] — convert to PIL per game per player
        all_game_prev_pils = []  # [G][N] list of lists: prev_pils[g][pid] = list of PIL for players 0..pid-1
        for g in range(G):
            game_prev_pils = []
            accumulated_pils = []
            for pid in range(num_players):
                game_prev_pils.append(list(accumulated_pils))  # copy current list
                pil_img = tensor_images_to_pil(all_images_packed[g, pid].unsqueeze(0).float())[0]
                accumulated_pils.append(pil_img)
            all_game_prev_pils.append(game_prev_pils)

        # Restore wrapped model for training backward
        model.language_model = wrapped_lm_save

        rank = accelerator.process_index
        n_gpus = accelerator.num_processes

        # ── [ACCEL PATCH 2] Training with deferred gradient sync ──
        saved_grad_accum = accelerator.gradient_accumulation_steps
        accelerator.gradient_accumulation_steps = 999999

        no_sync_ctx = transformer.no_sync if hasattr(transformer, 'no_sync') else contextlib.nullcontext
        info = defaultdict(list)

        # Training: per-game parallelism (same as generation phase)
        my_game_ids_train = set(g for g in range(G) if g % n_gpus == rank)

        for inner_epoch in range(num_inner_epochs):
            adv_idx = 0

            with no_sync_ctx():
                for g in range(G):
                    for pid in range(num_players):
                        has_traj = (g in my_game_ids_train) and ('all_latents' in all_game_trajs[g][pid])

                        if has_traj:
                            ptraj = all_game_trajs[g][pid]
                            latents = torch.stack(ptraj['all_latents'])
                            log_probs = torch.stack(ptraj['all_log_probs'])
                            timesteps = ptraj['timesteps']

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
                            cur_sample['dtimesteps'] = torch.cat([
                                timesteps[:-1] - timesteps[1:],
                                timesteps[-1:],
                            ])

                            # Build interleaved input with previous players' images
                            prev_pils = all_game_prev_pils[g][pid]
                            input_list = game_generator.build_generation_input_list(
                                all_game_data_list[g], pid + 1, prev_pils)

                            with autocast():
                                output_dict = inferencer.interleave_inference(
                                    input_list,
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
                            # interleave_inference returns a list
                            result = output_dict[0] if isinstance(output_dict, list) else output_dict

                            info["clipfrac"].append(result["clipfrac"])
                            info["clipfrac_gt_one"].append(result.get("clipfrac_gt_one", result["clipfrac"]))
                            info["clipfrac_lt_one"].append(result.get("clipfrac_lt_one", result["clipfrac"]))
                            info["policy_loss"].append(result["policy_loss"])
                            info["kl_loss"].append(result["kl_loss"])
                            info["loss"].append(result["loss"])

                        adv_idx += 1

            # ── ONE gradient sync + optimizer step per inner epoch ──
            if n_gpus > 1:
                for param in transformer.parameters():
                    if param.grad is not None:
                        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)

            torch.nn.utils.clip_grad_norm_(
                [p for p in transformer.parameters() if p.grad is not None],
                config.train.max_grad_norm or 1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

            if info["loss"] and accelerator.is_main_process:
                log_info = {k: torch.mean(torch.stack(v)).item()
                            for k, v in info.items()}
                log_info["abs_policy_loss"] = abs(log_info["policy_loss"])
                log_info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                wandb.log(log_info, step=global_step)
            global_step += 1
            info = defaultdict(list)

        # Restore original grad_accum
        accelerator.gradient_accumulation_steps = saved_grad_accum

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
