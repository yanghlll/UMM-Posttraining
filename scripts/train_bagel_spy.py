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

Architecture: FSDP SHARD_GRAD_OP (ZeRO-2: gradient + optimizer states sharded) + patches:
  - SHARD_GRAD_OP keeps full params on each GPU (no all-gather during forward),
    so asymmetric inference (spy sees prev images, early EOS exit) works correctly.
  - Gradients reduce-scattered only at accumulation boundary (once per epoch).
  - Remove accelerate dispatch hooks (10x voting speedup)
  - Async logging (non-blocking)
  - FSDP checkpoint save (full state dict, rank 0 only)
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
    sample_vote_with_logprobs, batch_sample_vote_with_logprobs,
    compute_decision_rewards,
    compute_text_grpo_loss, build_vote_kv_cache,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import wandb

import tempfile
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
        "reward_std": np.std(flat_rewards_raw),
        "spy_mean_reward": np.mean(epoch_spy_rewards) if epoch_spy_rewards else 0,
        "civ_mean_reward": np.mean(epoch_civ_rewards) if epoch_civ_rewards else 0,
        "advantage_std": float(advantages.std()),
    }

    game_metrics = {
        "spy_detection_rate": spy_rate,
        "spy_detection_rate_epoch": sum(1 for o in all_game_outcomes if o['spy_caught']) / G,
        "ema_baseline_spy": game_generator.b_spy,
        "ema_baseline_civ": game_generator.b_civ,
        "ema_baseline_gap": game_generator.b_civ - game_generator.b_spy,
    }
    if use_role_advantage:
        game_metrics["mean_adjusted_reward"] = np.mean(flat_rewards)

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

    all_metrics = {
        **reward_metrics, **game_metrics, **vote_stats,
        **diversity_metrics, **spy_div_metrics,
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


def _log_grad_diagnostics(transformer, accelerator, phase, global_step, logger):
    """Log gradient and parameter update diagnostics for both training phases.

    Checks:
      - generation phase: moe_gen params have grads, und params do NOT
      - decision phase:   und params have grads, moe_gen params do NOT (zeroed)
      - parameters actually changed (non-zero grad norm)
    """
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(transformer)
    gen_grad_norm = 0.0
    gen_grad_count = 0
    und_grad_norm = 0.0
    und_grad_count = 0
    gen_has_nonzero = False
    und_has_nonzero = False
    for n, p in unwrapped.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is not None:
            gn = p.grad.float().norm().item()
            if 'moe_gen' in n:
                gen_grad_norm += gn ** 2
                gen_grad_count += 1
                if gn > 1e-10:
                    gen_has_nonzero = True
            else:
                und_grad_norm += gn ** 2
                und_grad_count += 1
                if gn > 1e-10:
                    und_has_nonzero = True
    gen_grad_norm = gen_grad_norm ** 0.5
    und_grad_norm = und_grad_norm ** 0.5

    if phase == 'generation':
        ok = gen_has_nonzero and not und_has_nonzero
        logger.info(
            f"[Grad Check step={global_step}] phase={phase} | "
            f"moe_gen_grad_norm={gen_grad_norm:.6f} ({gen_grad_count} params) "
            f"und_grad_norm={und_grad_norm:.6f} ({und_grad_count} params) | "
            f"{'OK' if ok else 'WARN: und grads should be 0'}")
    else:
        ok = und_has_nonzero and not gen_has_nonzero
        logger.info(
            f"[Grad Check step={global_step}] phase={phase} | "
            f"und_grad_norm={und_grad_norm:.6f} ({und_grad_count} params) "
            f"moe_gen_grad_norm={gen_grad_norm:.6f} ({gen_grad_count} params) | "
            f"{'OK' if ok else 'WARN: moe_gen grads should be 0'}")


def _log_param_change(transformer, accelerator, phase, global_step, logger,
                      prev_checksums):
    """Check if parameters actually changed after optimizer step.

    Uses relative change threshold to distinguish real updates from float noise.
    """
    if not accelerator.is_main_process:
        return prev_checksums
    unwrapped = accelerator.unwrap_model(transformer)
    checksums = {}
    changed_gen = 0
    changed_und = 0
    total_gen = 0
    total_und = 0
    max_gen_delta = 0.0
    max_und_delta = 0.0
    for n, p in unwrapped.named_parameters():
        if not p.requires_grad:
            continue
        # Use mean of absolute values for more stable checksum
        cs = p.data.float().abs().mean().item()
        rel_thresh = max(cs * 1e-5, 1e-8)  # relative threshold
        if 'moe_gen' in n:
            total_gen += 1
            if n in prev_checksums:
                delta = abs(cs - prev_checksums[n])
                max_gen_delta = max(max_gen_delta, delta)
                if delta > rel_thresh:
                    changed_gen += 1
        else:
            total_und += 1
            if n in prev_checksums:
                delta = abs(cs - prev_checksums[n])
                max_und_delta = max(max_und_delta, delta)
                if delta > rel_thresh:
                    changed_und += 1
        checksums[n] = cs

    if prev_checksums:
        if phase == 'generation':
            ok = changed_gen > 0 and changed_und == 0
        else:
            ok = changed_und > 0 and changed_gen == 0
        logger.info(
            f"[Param Check step={global_step}] phase={phase} | "
            f"moe_gen changed={changed_gen}/{total_gen} (max_delta={max_gen_delta:.2e}) "
            f"und changed={changed_und}/{total_und} (max_delta={max_und_delta:.2e}) | "
            f"{'OK' if ok else 'WARN'}")
    return checksums


def _set_fsdp_training_mode(transformer, accelerator, use_lora):
    """Set FSDP wrapper to train mode but inner modules to inference mode.

    Matches original flow_grpo pattern (train_bagel.py L808-819):
      transformer.train()          → FSDP handles gradients
      inner.training = False       → Bagel dispatches to forward_inference

    With FSDP auto_wrap, each layer may be independently wrapped,
    so we set training=False on both the wrapper and inner module.
    """
    transformer.train()
    inner = accelerator.unwrap_model(transformer)
    inner.training = False
    inner.model.training = False
    layers = inner.model.model.layers if use_lora else inner.model.layers
    if use_lora:
        inner.model.model.training = False
    for layer in layers:
        layer.training = False
        if hasattr(layer, 'module'):
            layer.module.training = False
            if hasattr(layer.module, 'self_attn'):
                layer.module.self_attn.training = False
        elif hasattr(layer, 'self_attn'):
            layer.self_attn.training = False


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

    # gradient_accumulation_steps is set to 1 here; overridden per-epoch in
    # training loop to match (games_per_rank * num_players * sde_window).
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=1,
    )
    if hasattr(accelerator.state, 'fsdp_plugin') and accelerator.state.fsdp_plugin is not None:
        accelerator.state.fsdp_plugin.activation_checkpointing = config.activation_checkpointing
        accelerator.state.fsdp_plugin.transformer_cls_names_to_wrap = ['Qwen2MoTDecoderLayer']

    n_gpus = accelerator.num_processes
    import math as _math
    import logging as _logging

    # File logging: write all ranks' logs to a shared log file
    log_dir = getattr(config, 'logdir', '/adialab/usr/shadabk/MedUMM/flow_grpo/logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"spy_train_{config.run_name}.log")
    file_handler = _logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(_logging.Formatter(
        f'[%(asctime)s][rank{accelerator.process_index}] %(message)s',
        datefmt='%H:%M:%S'))
    _logging.getLogger(__name__).addHandler(file_handler)
    _logging.getLogger(__name__).setLevel(_logging.INFO)

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

    # Create frozen reference model for KL penalty (if beta > 0)
    # Prepared via accelerator.prepare() so FSDP wraps it with SHARD_GRAD_OP
    # (same strategy as main transformer — safe for asymmetric forward calls).
    if config.train.beta > 0:
        language_model_ref = Qwen2ForCausalLM(llm_config)
        language_model_ref.load_state_dict(model.language_model.state_dict())
        language_model_ref.to(device=f"cuda:{accelerator.local_process_index}", dtype=inference_dtype)
        language_model_ref.eval()
        language_model_ref.requires_grad_(False)
        model.language_model_ref = language_model_ref
        logger.info("  KL reference model: CREATED (frozen copy of language_model)")

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
        # Enable und params for decision phase training
        decision_training = spy_cfg.get('decision_training', False)
        if decision_training:
            # Enable all und params (non-moe_gen) for decision phase training
            for name, param in model.language_model.named_parameters():
                if 'moe_gen' not in name and 'embed_tokens' not in name and 'lm_head' not in name:
                    param.requires_grad = True

    transformer = model.language_model
    transformer.config.use_cache = False

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Dual optimizers: gen (diffusion GRPO) + und (VLM GRPO)
    gen_params = [p for n, p in transformer.named_parameters()
                  if 'moe_gen' in n and p.requires_grad]
    gen_optimizer = torch.optim.AdamW(
        gen_params,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    decision_training = spy_cfg.get('decision_training', False)
    und_scheduler = None
    if decision_training:
        und_params = [p for n, p in transformer.named_parameters()
                      if 'moe_gen' not in n and p.requires_grad]
        und_optimizer = torch.optim.AdamW(
            und_params,
            lr=spy_cfg.get('decision_lr', 1e-5),
            betas=(0.9, 0.999),
            weight_decay=spy_cfg.get('decision_weight_decay', 0.01),
            eps=1e-8,
        )
        # Warmup + cosine scheduler (Vision-Zero: warmup_ratio=0.1, cosine)
        total_decision_steps = config.num_epochs  # 1 step per epoch
        warmup_steps = int(total_decision_steps * spy_cfg.get('decision_warmup_ratio', 0.1))
        if spy_cfg.get('decision_lr_scheduler', 'cosine') == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            warmup_scheduler = LinearLR(und_optimizer, start_factor=0.01, total_iters=max(warmup_steps, 1))
            cosine_scheduler = CosineAnnealingLR(und_optimizer, T_max=max(total_decision_steps - warmup_steps, 1))
            und_scheduler = SequentialLR(und_optimizer, [warmup_scheduler, cosine_scheduler],
                                         milestones=[warmup_steps])
        logger.info(f"  Decision training: ENABLED (und params: {len(und_params)}, "
                     f"lr={spy_cfg.get('decision_lr', 1e-5)}, warmup={warmup_steps}, "
                     f"scheduler={spy_cfg.get('decision_lr_scheduler', 'cosine')})")
    else:
        und_optimizer = None

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # FSDP: transformer + optimizers via accelerator.prepare() (SHARD_GRAD_OP).
    # Ref model NOT prepared — kept as plain model on each GPU (no FSDP).
    # This avoids FSDP sharding embed_tokens which breaks direct .model.embed_tokens() calls.
    if decision_training:
        transformer, gen_optimizer, und_optimizer = accelerator.prepare(
            transformer, gen_optimizer, und_optimizer)
    else:
        transformer, gen_optimizer = accelerator.prepare(transformer, gen_optimizer)
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
    max_vote_tokens = spy_cfg.get('max_vote_tokens', 1024)
    num_inner_epochs = config.train.num_inner_epochs
    use_role_advantage = spy_cfg.get('use_role_advantage', True)
    logger.info(f"  Game data: prompt_type={prompt_type}")
    logger.info(f"  God-judge voting: K={spy_cfg.get('god_vote_K', 8)} votes per game")
    logger.info(f"  Role advantage: {'ENABLED' if use_role_advantage else 'DISABLED'}")

    # ==================== TRAINING LOOP ====================
    logger.info("***** Running Bagel Spy-Civ Flow-GRPO Training *****")
    logger.info(f"  Players={num_players}, G={G}, inner_epochs={num_inner_epochs}")
    logger.info(f"  SDE window={config.sample.sde_window_size}")
    games_per_rank = _math.ceil(G / n_gpus)
    logger.info(f"  GPUs={n_gpus}, games_per_rank={games_per_rank} (sequential gen per game)")
    logger.info(f"  Gradient sync: FSDP + Accelerator accumulate() (both gen & decision)")
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
    
    # Initial training mode: FSDP wrapper train + inner modules inference dispatch
    _set_fsdp_training_mode(transformer, accelerator, config.use_lora)

    batch_time_start = time.time()

    for epoch in range(start_epoch, config.num_epochs):
        # Free GPU memory periodically (not every epoch to avoid overhead)
        if epoch % 5 == 0:
            torch.cuda.empty_cache()

        # ── Eval & Checkpoint ────────────────────────────────────
        # FSDP checkpoint save
        if not config.debug and epoch > 0 and epoch % config.save_freq == 0:
            from flow_grpo.fsdp_utils import save_fsdp_checkpoint
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

        # FSDP: keep model.language_model as the FSDP-wrapped transformer
        # (set at accelerator.prepare). Do NOT unwrap — FSDP manages parameter sharding.

        rank = accelerator.process_index
        n_gpus = accelerator.num_processes

        # ── PHASE 1 + 2 setup ──────────────────────────────────────
        t_gen_start = time.time()
        all_my_trajs = []   # [G] dict mapping pid -> traj
        all_my_images = []  # [G] dict mapping pid -> image tensor

        # Pre-generate all game data
        for g in range(G):
            all_game_data_list.append(game_generator.generate_game(epoch * G + g, global_step))


        # Distribute games across GPUs (round-robin)
        my_game_ids_gen = [g for g in range(G) if g % n_gpus == rank]

        # Initialize per-game storage for all G games
        for g in range(G):
            all_my_trajs.append({})
            all_my_images.append({})

        god_vote_K = spy_cfg.get('god_vote_K', 8)
        first_game_vote_texts = None
        first_game_god_prompt = None
        all_vote_count_tensors = torch.zeros(G, num_players,
                                              device=accelerator.device, dtype=torch.long)
        all_game_pils = {}
        all_game_god_prompts = {}
        all_vote_samples = {}
        all_game_vote_infos = {}  # g -> list of K vote parse results (including None)

        # ================================================================
        # PHASE 1: Image generation — each GPU generates its assigned games
        # SHARD_GRAD_OP: full params on each GPU, no cross-rank sync during forward.
        # Players generate sequentially within each game (spy sees prev images).
        #
        # Data flow (matches flow_grpo pattern):
        #   game_data → build_generation_input_list → interleave_inference
        #   → pre-stack latents/log_probs (avoid repeated stack in backward)
        #   → convert to PIL once (reused for voting + backward prev_images)
        # ================================================================
        _t_gen_s = time.time()
        all_game_prev_pils_local = {}  # g -> [pid -> list of prev player PILs]
        with autocast():
            for g in my_game_ids_gen:
                game_data = all_game_data_list[g]
                spy_player = game_data['spy_player']
                game_pil_images = []   # accumulated PILs for spy context
                game_prev_pils = []    # per-player prev_images for backward

                for pid in range(num_players):
                    player_id = pid + 1
                    # Spy sees previous players' images; civilians generate independently
                    if player_id == spy_player:
                        prev_images = list(game_pil_images)
                    else:
                        prev_images = []
                    game_prev_pils.append(list(prev_images))

                    input_list = game_generator.build_generation_input_list(
                        game_data, player_id, prev_images)

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
                    result = output_dict[0] if isinstance(output_dict, list) else output_dict

                    # Pre-stack latents/log_probs during sampling (flow_grpo pattern)
                    # so backward doesn't need to re-stack every inner epoch
                    all_my_trajs[g][pid] = {
                        'image': result['image'],
                        'latents': torch.stack(result['all_latents']),
                        'log_probs': torch.stack(result['all_log_probs']),
                        'timesteps': result['timesteps'],
                    }
                    all_my_images[g][pid] = result['image']

                    # Convert to PIL once — reused for voting, logging, and backward
                    pil_img = tensor_images_to_pil(result['image'].unsqueeze(0).float())[0]
                    game_pil_images.append(pil_img)

                all_game_pils[g] = game_pil_images
                all_game_prev_pils_local[g] = game_prev_pils
                all_game_god_prompts[g] = game_generator.format_voting_prompt(
                    game_data, player_id=None,
                    god_sees_description=spy_cfg.get('god_sees_description', False))

            _t_gen_only = time.time() - _t_gen_s

            # ── PHASE 2: Batched voting — all games' votes in one packed forward ──
            _t_vote_s = time.time()
            batch_pils = [all_game_pils[g] for g in my_game_ids_gen]
            batch_prompts = [all_game_god_prompts[g] for g in my_game_ids_gen]

            with torch.no_grad():
                if decision_training:
                    batch_vote_results = batch_sample_vote_with_logprobs(
                        inferencer, batch_pils, batch_prompts,
                        num_generations=god_vote_K,
                        max_tokens=max_vote_tokens,
                        temperature=spy_cfg.get('decision_temperature', 0.7),
                        return_base_kv=True)
                    for idx, g in enumerate(my_game_ids_gen):
                        vote_samples, base_cached_kv = batch_vote_results[idx]
                        all_vote_samples[g] = vote_samples
                        all_vote_samples[f'{g}_kv'] = base_cached_kv
                else:
                    batch_vote_results = batch_sample_vote_with_logprobs(
                        inferencer, batch_pils, batch_prompts,
                        num_generations=god_vote_K,
                        max_tokens=max_vote_tokens,
                        temperature=spy_cfg.get('decision_temperature', 0.9),
                        return_base_kv=False)

        # Parse votes for all games
        for idx, g in enumerate(my_game_ids_gen):
            if decision_training:
                vote_samples_g = all_vote_samples[g]
                vote_texts = [vs['text'] for vs in vote_samples_g]
            else:
                vote_texts = [vs['text'] for vs in batch_vote_results[idx]]

            game_vote_infos = []
            for vtext in vote_texts:
                vote_info = game_generator.extract_vote(vtext)
                game_vote_infos.append(vote_info)
                if vote_info and isinstance(vote_info.get('voted_spy'), int):
                    voted_pid = vote_info['voted_spy']
                    if 1 <= voted_pid <= num_players:
                        all_vote_count_tensors[g, voted_pid - 1] += 1
            all_game_vote_infos[g] = game_vote_infos

            if g == my_game_ids_gen[0] and rank == 0:
                first_game_vote_texts = vote_texts
                first_game_god_prompt = all_game_god_prompts[g]

        _t_vote_only = time.time() - _t_vote_s
        t_gen_total = _t_gen_only
        t_vote_total = _t_vote_only
        
        # Determine training phase early (needed for all_reduce decision)
        if decision_training:
            cycle_length = spy_cfg.get('phase_cycle_length', 10)
            total_cycle = cycle_length * 2
            cycle_pos = global_step % total_cycle
            training_phase = 'decision' if cycle_pos < cycle_length else 'generation'
        else:
            training_phase = 'generation'

        # all_reduce vote_counts (always needed for reward)
        torch.distributed.all_reduce(all_vote_count_tensors, op=torch.distributed.ReduceOp.SUM)

        # all_reduce images — only needed for generation training backward
        # (decision phase uses local PIL images cached in all_game_pils)
        img_shape = None
        for g in my_game_ids_gen:
            img_shape = list(all_my_images[g][0].shape)
            break
        if img_shape is None:
            img_shape = [3, config.resolution, config.resolution]
        all_images_packed = torch.zeros(G, num_players, *img_shape,
                                        device=accelerator.device, dtype=torch.bfloat16)
        for g in my_game_ids_gen:
            for pid in range(num_players):
                all_images_packed[g, pid] = all_my_images[g][pid]
        if training_phase == 'generation':
            # Only sync full image tensors when generation backward needs them
            torch.distributed.all_reduce(all_images_packed, op=torch.distributed.ReduceOp.SUM)

        t_gen_vote_combined = time.time() - t_gen_start  # combined gen+vote+allreduce time

        # ================================================================
        # Phase 3: Rewards → Advantages (Vision-Zero strategic reward)
        #
        # Flow:  vote_counts → spy_caught → zero-sum rewards → role-adjust → normalize
        #
        # vs original flow_grpo:
        #   flow_grpo: external reward_fn(images, prompts) → gather → normalize
        #   spy game:  vote_counts → compute_generation_rewards → role-adjust → normalize
        #
        # Key difference: rewards come from the game outcome (votes), not external metrics.
        # Zero-sum guarantees: sum(rewards) = 0 per game.
        # Role advantage: subtract per-role EMA baseline before normalization
        # to prevent spy/civ reward imbalance from dominating the advantage signal.
        # ================================================================
        flat_rewards_raw = []
        flat_rewards_adj = []

        for g in range(G):
            game_data = all_game_data_list[g]
            spy_pid = game_data['spy_player']

            # Vote counts → spy detection (majority vote, unique winner)
            vote_counts = {pid + 1: all_vote_count_tensors[g, pid].item()
                           for pid in range(num_players)}
            max_votes = max(vote_counts.values())
            spy_caught = (vote_counts[spy_pid] == max_votes and
                          sum(1 for v in vote_counts.values() if v == max_votes) == 1)

            # Zero-sum generation rewards (Vision-Zero: Ψ-based formula)
            game_outcome = {
                "spy_caught": spy_caught,
                "vote_counts": vote_counts,
                "spy_player": spy_pid,
                "player_rewards": {pid + 1: 0.0 for pid in range(num_players)},
            }
            gen_rewards = game_generator.compute_generation_rewards(
                game_outcome, beta=spy_cfg.reward_beta, lambda_param=spy_cfg.reward_lambda)

            # Role advantage: subtract per-role EMA baseline (before normalization)
            if use_role_advantage:
                adj_rewards = game_generator.apply_role_advantage(gen_rewards, spy_pid)
            else:
                adj_rewards = gen_rewards

            # Update EMA baselines with current data (for next epoch)
            civ_rs = [gen_rewards[i] for i in range(num_players) if i != spy_pid - 1]
            game_generator.update_baselines(
                gen_rewards[spy_pid - 1],
                sum(civ_rs) / len(civ_rs) if civ_rs else 0.0)

            # Accumulate flat reward lists (G*N entries, same order as advantages index)
            flat_rewards_raw.extend(gen_rewards)
            flat_rewards_adj.extend(adj_rewards)

            # Track stats
            if spy_caught:
                spy_caught_count += 1
            total_games += 1
            epoch_spy_rewards.append(gen_rewards[spy_pid - 1])
            for i, r in enumerate(gen_rewards):
                if i != spy_pid - 1:
                    epoch_civ_rewards.append(r)

            # Build per-game training data
            player_images_tensor = [all_images_packed[g, pid] for pid in range(num_players)]
            player_trajs = [None] * num_players
            for pid in range(num_players):
                if pid in all_my_trajs[g]:
                    player_trajs[pid] = all_my_trajs[g][pid]
                else:
                    player_trajs[pid] = {'image': player_images_tensor[pid]}

            all_game_trajs.append(player_trajs)
            all_game_rewards.append(gen_rewards)
            all_game_images_t.append(player_images_tensor)
            all_game_votes.append(all_game_vote_infos.get(g, []))
            all_game_outcomes.append(game_outcome)

        # Group-level mean/std normalization → advantages (matches flow_grpo's advantage computation)
        # flow_grpo: advantages = (rewards - mean) / (std + eps)  [global across all samples]
        # spy game:  same formula but on role-adjusted rewards [across G*N trajectories]
        advantages = compute_group_advantages(flat_rewards_adj).to(accelerator.device)

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
                list(flat_rewards_adj), adv_cpu, list(epoch_spy_rewards), list(epoch_civ_rewards),
                spy_caught_count, total_games, game_generator,
                use_role_advantage, G, num_players, t_gen_total, t_vote_total,
                batch_time, epoch, global_step, config,
            )

            # Log full vote texts for first game (every 10 epochs)
            if epoch % 5 == 0 and first_game_vote_texts is not None:
                try:
                    gd = all_game_data_list[0]
                    spy_pid = gd['spy_player']
                    # Vote counts for this game
                    vc = {pid+1: all_vote_count_tensors[0, pid].item() for pid in range(num_players)}

                    vote_log = f"<h3>Epoch {epoch} | Game 0 | Spy=Player {spy_pid}</h3>\n"
                    vote_log += f"<b>Original prompt (CIV):</b> {gd['original_description']}<br>\n"
                    vote_log += f"<b>Modified prompt (SPY):</b> {gd['modified_description']}<br>\n"
                    vote_log += f"<b>Change:</b> {gd['diff_metadata'].get('change', '?')}<br>\n"
                    max_vc = max(vc.values()) if vc else 0
                    spy_caught_log = (vc.get(spy_pid, 0) == max_vc and
                                      sum(1 for v in vc.values() if v == max_vc) == 1)
                    vote_log += f"<b>Vote counts:</b> {vc} | Spy caught: {spy_caught_log}<br>\n"
                    vote_log += f"<b>God prompt:</b><br><pre>{first_game_god_prompt}</pre><hr>\n"

                    import re as _re_log
                    n_valid = 0; n_correct = 0; n_na = 0; n_invalid = 0; n_has_think = 0
                    for vi, vt in enumerate(first_game_vote_texts):
                        vote_info = game_generator.extract_vote(vt)
                        voted = vote_info.get('voted_spy', '?') if vote_info else 'PARSE_FAIL'
                        # Stats
                        if vote_info is None:
                            n_invalid += 1
                        elif voted == 'N/A':
                            n_na += 1; n_valid += 1
                        elif isinstance(voted, int):
                            n_valid += 1
                            if voted == spy_pid:
                                n_correct += 1
                        think_m = _re_log.search(r'<think>(.*?)</think>', vt, _re_log.DOTALL)
                        if think_m and len(think_m.group(1).strip()) > 10:
                            n_has_think += 1

                        correct = "✓" if voted == spy_pid else ("N/A" if voted == "N/A" else "✗")
                        vote_log += f"<h4>Vote {vi+1}/{len(first_game_vote_texts)}: "
                        vote_log += f"voted=Player {voted} {correct}</h4>\n"
                        vote_log += f"<pre>{vt}</pre><hr>\n"

                    vote_log += (f"<b>Summary:</b> valid={n_valid}, correct={n_correct}, "
                                 f"N/A={n_na}, invalid={n_invalid}, has_think={n_has_think}<br>\n")

                    wandb.log({"vote_texts": wandb.Html(vote_log)}, step=global_step)
                except Exception:
                    pass

        # training_phase already determined above (before all_reduce)

        # FSDP: optimizer states are sharded across GPUs, no manual offload needed

        # Switch back to train mode before backward (matching original flow_grpo)
        _set_fsdp_training_mode(transformer, accelerator, config.use_lora)

        # Reuse PIL images computed during Phase 1 for generation backward.
        # all_game_prev_pils_local[g][pid] has the same prev_images used during sampling,
        # ensuring backward recomputes the same KV cache as inference.
        all_game_prev_pils = []
        if training_phase == 'generation':
            for g in range(G):
                if g in all_game_prev_pils_local:
                    all_game_prev_pils.append(all_game_prev_pils_local[g])
                else:
                    # Game not on this rank — placeholder (won't be used in backward)
                    all_game_prev_pils.append([[] for _ in range(num_players)])

        # FSDP: model.language_model stays as FSDP-wrapped transformer throughout

        rank = accelerator.process_index
        n_gpus = accelerator.num_processes

        # Set gradient_accumulation_steps per phase so Accelerator's accumulate()
        # mechanism triggers optimizer.step() only once per epoch.
        games_per_rank = len([g for g in range(G) if g % n_gpus == rank])
        if training_phase == 'generation':
            # Generation: 1 backward per (game, player, sde_step)
            accelerator.gradient_accumulation_steps = games_per_rank * num_players * config.sample.sde_window_size
        else:
            # Decision: 1 backward per (game, vote_sample_k)
            accelerator.gradient_accumulation_steps = games_per_rank * god_vote_K

        info = defaultdict(list)
        my_game_ids_train = set(g for g in range(G) if g % n_gpus == rank)
        _t_backward_start = time.time()

        # Snapshot params before backward for change detection (first 10 epochs only)
        _param_checksums = {}
        if epoch < 10:
            _param_checksums = _log_param_change(
                transformer, accelerator, training_phase, global_step, logger, {})

        for inner_epoch in range(num_inner_epochs):

            if training_phase == 'generation':
                # ── Generation GRPO backward (moe_gen params) ──
                # FSDP: all ranks must call forward the same number of times
                # (FSDP all-gathers params on each forward). Iterate only local
                # games so all ranks do games_per_rank × num_players forwards.
                my_game_ids_train_sorted = sorted(my_game_ids_train)
                for g in my_game_ids_train_sorted:
                    for pid in range(num_players):
                        adv_idx = g * num_players + pid
                        ptraj = all_game_trajs[g][pid]
                        if 'latents' not in ptraj:
                            continue

                        # Pre-stacked during Phase 1 sampling (no repeated stack)
                        latents = ptraj['latents']
                        log_probs = ptraj['log_probs']
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
                                optimizer=gen_optimizer,
                                transformer=transformer,
                                num_timesteps=config.sample.num_steps,
                                cfg_text_scale=config.sample.guidance_scale,
                                **inference_hyper,
                            )
                        result = output_dict[0] if isinstance(output_dict, list) else output_dict

                        info["clipfrac"].append(result["clipfrac"])
                        info["clipfrac_gt_one"].append(result.get("clipfrac_gt_one", result["clipfrac"]))
                        info["clipfrac_lt_one"].append(result.get("clipfrac_lt_one", result["clipfrac"]))
                        info["policy_loss"].append(result["policy_loss"])
                        info["kl_loss"].append(result["kl_loss"])
                        info["loss"].append(result["loss"])

                # Generation backward produces grads on ALL trainable params (moe_gen + und),
                # but gen_optimizer only steps moe_gen. Clear stale und grads directly
                # (can't use und_optimizer.zero_grad() — accelerator wraps it as no-op
                # during accumulation).
                for _n, _p in accelerator.unwrap_model(transformer).named_parameters():
                    if 'moe_gen' not in _n and _p.grad is not None:
                        _p.grad.zero_()

            elif training_phase == 'decision' and und_optimizer is not None:
                # ── Decision GRPO backward (und params) ──
                # NOTE: FSDP wraps params as non-leaf variables, so we cannot
                # toggle requires_grad directly. Instead, we zero out moe_gen
                # gradients after each backward to prevent gen params from updating.

                # FSDP: iterate only local games (same count per rank)
                for g in sorted(my_game_ids_train):
                    if g not in all_vote_samples:
                        continue
                    game_data = all_game_data_list[g]
                    spy_pid = game_data['spy_player']

                    # Reuse cached PIL images and god prompt from Phase 2
                    player_pils = all_game_pils[g]
                    god_prompt = all_game_god_prompts[g]
                    vote_samples = all_vote_samples[g]

                    # Compute decision rewards (per-game group normalization)
                    dec_rewards = compute_decision_rewards(
                        vote_samples, spy_pid, num_players,
                        game_generator.extract_vote,
                        lambda_fmt=spy_cfg.get('decision_lambda_fmt', 0.3),
                        beta_acc=spy_cfg.get('decision_beta_acc', 1.2))
                    dec_advantages = compute_group_advantages(dec_rewards)
                    info["decision_reward_mean"].append(torch.tensor(np.mean(dec_rewards)))
                    info["decision_reward_std"].append(torch.tensor(np.std(dec_rewards)))
                    info["decision_advantage_abs_max"].append(torch.tensor(max(abs(a) for a in dec_advantages.tolist())))

                    # Reuse KV cache from Phase 2 sampling (avoids redundant VIT+text encoding)
                    cached_kv = all_vote_samples.get(f'{g}_kv')
                    if cached_kv is None:
                        cached_kv = build_vote_kv_cache(
                            model, inferencer.tokenizer, inferencer.new_token_ids,
                            player_pils, god_prompt)

                    # Text GRPO backward for each vote sample (reusing cached KV)
                    for k, vs in enumerate(vote_samples):
                        with accelerator.accumulate(transformer):
                            loss, loss_info = compute_text_grpo_loss(
                                model, inferencer.tokenizer, inferencer.new_token_ids,
                                inferencer, player_pils, god_prompt, vs,
                                advantage=dec_advantages[k].item(),
                                clip_range=spy_cfg.get('decision_clip_range', 0.2),
                                kl_beta=spy_cfg.get('decision_kl_beta', 0.04),
                                cached_kv=cached_kv)
                            accelerator.backward(loss)
                            # Zero out moe_gen gradients (FSDP-safe: no requires_grad toggle)
                            for n, p in accelerator.unwrap_model(transformer).named_parameters():
                                if 'moe_gen' in n and p.grad is not None:
                                    p.grad.zero_()
                            if accelerator.sync_gradients:
                                torch.nn.utils.clip_grad_norm_(
                                    transformer.parameters(),
                                    config.train.max_grad_norm or 1.0)
                            und_optimizer.step()
                            und_optimizer.zero_grad()
                        info["decision_loss"].append(loss.detach())
                        info["decision_clipfrac"].append(
                            torch.tensor(loss_info['text_clipfrac']))
                        if loss_info.get('text_kl_mean', 0) > 0:
                            info["decision_kl"].append(
                                torch.tensor(loss_info['text_kl_mean']))
                        info["decision_token_len"].append(
                            torch.tensor(float(loss_info['text_seq_len'])))
                        del loss

                # FSDP handles gradient sync automatically via accumulate()
                if und_scheduler is not None:
                    und_scheduler.step()
                gen_optimizer.zero_grad()  # Clear any stray gen grads
                # Free cached KV from Phase 2
                for g in list(all_vote_samples.keys()):
                    if isinstance(g, str) and g.endswith('_kv'):
                        del all_vote_samples[g]

            # ── Gradient & parameter update diagnostics (first 10 epochs) ──
            if epoch < 10:
                _log_grad_diagnostics(transformer, accelerator, training_phase,
                                      global_step, logger)
                _param_checksums = _log_param_change(
                    transformer, accelerator, training_phase, global_step, logger,
                    _param_checksums)

            # Free sampling/voting caches at end of inner epoch to reduce memory pressure
            del all_vote_samples, all_game_prev_pils_local
            all_vote_samples = {}
            all_game_prev_pils_local = {}

            _t_backward_total = time.time() - _t_backward_start

            if info and accelerator.is_main_process:
                log_info = {}
                for k, v in info.items():
                    if v:
                        log_info[k] = torch.mean(torch.stack(v)).item()
                if "policy_loss" in log_info:
                    log_info["abs_policy_loss"] = abs(log_info["policy_loss"])
                log_info.update({"epoch": epoch, "inner_epoch": inner_epoch,
                                 "training_phase": 0 if training_phase == 'generation' else 1,
                                 })
                wandb.log(log_info, step=global_step)

            # Print phase timing to stdout for monitoring
            if accelerator.is_main_process:
                logger.info(
                    f"[Step {global_step}] phase={training_phase} | "
                    f"gen={_t_gen_only:.1f}s vote={_t_vote_only:.1f}s "
                    f"backward={_t_backward_total:.1f}s "
                    f"total={_t_gen_only + _t_vote_only + _t_backward_total:.1f}s"
                )
            global_step += 1
            info = defaultdict(list)

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
