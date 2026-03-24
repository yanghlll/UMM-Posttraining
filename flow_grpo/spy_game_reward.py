"""
Spy Game Reward Utilities for Flow-GRPO Bagel Training.

Provides voting grid construction and Bagel understanding-mode voting,
adapted from SPY-UMM/models/showo2_spy_wrapper.py:judge_vote().
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from typing import List


# ─── Tensor-native voting grid (avoids GPU→CPU→PIL→CPU round-trip) ───────────

def build_voting_grid_tensor(image_tensors: List[torch.Tensor],
                              cell_size: int = 256) -> torch.Tensor:
    """Build a voting grid directly from GPU tensors. No PIL needed.

    Args:
        image_tensors: List of [3,H,W] tensors on GPU, values in [0,1].
        cell_size: Target size per cell.

    Returns:
        Single [3, grid_H, grid_W] tensor on same device.
    """
    N = len(image_tensors)
    cols = min(N, 2)
    rows = (N + cols - 1) // cols
    device = image_tensors[0].device

    # Resize all images to cell_size in one batch (GPU-accelerated)
    stacked = torch.stack(image_tensors)  # [N, 3, H, W]
    resized = F.interpolate(stacked, size=(cell_size, cell_size),
                            mode='bilinear', align_corners=False)  # [N, 3, cs, cs]

    # Build grid on GPU
    grid = torch.full((3, rows * cell_size, cols * cell_size), 0.78,
                       device=device, dtype=resized.dtype)  # gray bg
    for idx in range(N):
        r, c = divmod(idx, cols)
        y, x = r * cell_size, c * cell_size
        grid[:, y:y + cell_size, x:x + cell_size] = resized[idx]

    return grid


def grid_tensor_to_pil(grid_tensor: torch.Tensor,
                       num_players: int = 0,
                       cell_size: int = 256) -> Image.Image:
    """Convert grid tensor to PIL and add player labels.

    Args:
        grid_tensor: [3, H, W] tensor from build_voting_grid_tensor.
        num_players: If > 0, draw "Player N" labels on each cell.
        cell_size: Cell size used in grid construction (for label placement).
    """
    img = (grid_tensor * 255).round().clamp(0, 255).to(torch.uint8).cpu()
    pil = Image.fromarray(img.permute(1, 2, 0).numpy())

    if num_players > 0:
        draw = ImageDraw.Draw(pil)
        cols = min(num_players, 2)
        for idx in range(num_players):
            r, c = divmod(idx, cols)
            x = c * cell_size + 5
            y = r * cell_size + 2
            draw.text((x, y), f"Player {idx + 1}", fill=(255, 0, 0))

    return pil


# ─── Legacy PIL-based grid (for logging only) ────────────────────────────────

def build_voting_grid(pil_images: List[Image.Image], cell_size: int = 256) -> Image.Image:
    """Build a labeled grid from PIL images. Used for wandb logging."""
    N = len(pil_images)
    cols = min(N, 2)
    rows = (N + cols - 1) // cols
    grid_w = cols * cell_size
    grid_h = rows * (cell_size + 20)
    grid = Image.new('RGB', (grid_w, grid_h), (200, 200, 200))
    draw = ImageDraw.Draw(grid)
    for idx, img in enumerate(pil_images):
        r, c = divmod(idx, cols)
        x = c * cell_size
        y = r * (cell_size + 20)
        draw.text((x + 5, y + 2), f"Player {idx + 1}", fill=(0, 0, 0))
        grid.paste(img.resize((cell_size, cell_size)), (x, y + 20))
    return grid


def tensor_images_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """Convert batch [N,3,H,W] in [0,1] to PIL. Only for logging."""
    if images.dim() == 3:
        images = images.unsqueeze(0)
    imgs = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    imgs = imgs.transpose(0, 2, 3, 1)
    return [Image.fromarray(img) for img in imgs]


# ─── Voting ──────────────────────────────────────────────────────────────────

def run_bagel_vote(inferencer, grid_image: Image.Image, vote_prompt: str,
                   max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Run Bagel understanding mode to vote on a grid image (legacy, single grid)."""
    output = inferencer.interleave_inference(
        input_lists=[grid_image, vote_prompt],
        understanding_output=True,
        do_sample=True,
        text_temperature=temperature,
        max_think_token_n=max_tokens,
    )
    if isinstance(output, list) and len(output) > 0:
        return output[0]
    return str(output)


def run_bagel_vote_multi_image(inferencer, player_images: List[Image.Image],
                                vote_prompt: str,
                                max_tokens: int = 512,
                                temperature: float = 0.7) -> str:
    """Run Bagel understanding mode with separate images for each player.

    Instead of a single grid image, passes each player's image separately
    with text labels, so the model clearly knows which image is which player.

    Args:
        inferencer: Bagel InterleaveInferencer.
        player_images: List of N PIL images, one per player (0-indexed).
        vote_prompt: Voting prompt text (appended after all images).
        max_tokens: Max tokens for vote response.
        temperature: Sampling temperature.
    """
    # Build interleaved input: [label, image, label, image, ..., vote_prompt]
    input_lists = []
    for i, img in enumerate(player_images):
        input_lists.append(f"Player {i+1}'s generated image:")
        input_lists.append(img)
    input_lists.append(vote_prompt)

    output = inferencer.interleave_inference(
        input_lists=input_lists,
        understanding_output=True,
        do_sample=True,
        text_temperature=temperature,
        max_think_token_n=max_tokens,
    )
    if isinstance(output, list) and len(output) > 0:
        return output[0]
    return str(output)


def run_bagel_votes_cached(inferencer, grid_image: Image.Image,
                           vote_prompts: List[str],
                           max_tokens: int = 512,
                           temperature: float = 0.7) -> List[str]:
    """Run multiple votes on the same grid image, caching ViT encoding.
    Legacy serial version - kept for fallback.
    """
    from copy import deepcopy
    from flow_grpo.bagel.data.data_utils import pil_img2rgb

    results = []
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        base_context = inferencer.init_gen_context()
        transformed_img = inferencer.vae_transform.resize_transform(pil_img2rgb(grid_image))
        base_context = inferencer.update_context_image(
            transformed_img, base_context, vae=False)
        for prompt in vote_prompts:
            vote_context = deepcopy(base_context)
            vote_context = inferencer.update_context_text(prompt, vote_context)
            gen_text = inferencer.gen_text(
                vote_context, do_sample=True,
                temperature=temperature, max_length=max_tokens)
            results.append(gen_text)
    return results


def run_bagel_votes_batch(model, tokenizer, new_token_ids, vit_transform,
                          grid_images: List[Image.Image],
                          vote_prompts: List[str],
                          max_tokens: int = 512,
                          temperature: float = 0.7) -> List[str]:
    """Batch vote: encode N grid images + prompts into packed sequence,
    generate all N vote texts in one autoregressive pass.

    This gives batch=N for the text generation, increasing GPU utilization
    from ~160W (batch=1) to much higher.

    Args:
        model: Unwrapped Bagel model.
        tokenizer: Qwen2Tokenizer.
        new_token_ids: Special token IDs dict.
        vit_transform: ImageTransform for ViT.
        grid_images: List of N PIL images (can be same or different).
        vote_prompts: List of N text prompts.
        max_tokens: Max tokens per vote.
        temperature: Sampling temperature.

    Returns:
        List of N vote text strings.
    """
    from flow_grpo.bagel.data.data_utils import pil_img2rgb
    from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache

    N = len(vote_prompts)
    device = next(model.parameters()).device

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        # Step 1: Build packed KV cache with N images + N prompts
        past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        kv_lens = [0] * N
        ropes = [0] * N

        # Encode N images with ViT (packed)
        images_transformed = [vit_transform.resize_transform(pil_img2rgb(img))
                              for img in grid_images]
        generation_input, kv_lens, ropes = model.prepare_vit_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=images_transformed,
            transforms=vit_transform,
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if isinstance(v, torch.Tensor):
                generation_input[k] = v.to(device)
        past_key_values = model.forward_cache_update_vit(past_key_values, **generation_input)

        # Encode N text prompts (packed)
        generation_input, kv_lens, ropes = model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=vote_prompts,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if isinstance(v, torch.Tensor):
                generation_input[k] = v.to(device)
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)

        # Step 2: Prepare start tokens for batch text generation
        gen_input = model.prepare_start_tokens(kv_lens, ropes, new_token_ids)
        for k, v in gen_input.items():
            if isinstance(v, torch.Tensor):
                gen_input[k] = v.to(device)

        # Step 3: Batch autoregressive generation (batch=N)
        output_tokens = model.generate_text(
            past_key_values=past_key_values,
            max_length=max_tokens,
            do_sample=True,
            temperature=temperature,
            end_token_id=new_token_ids['eos_token_id'],
            **gen_input,
        )

    # Step 4: Decode each sample's tokens
    results = []
    for i in range(N):
        text = tokenizer.decode(output_tokens[:, i])
        text = text.split('<|im_end|>')[0]
        if '<|im_start|>' in text:
            text = text.split('<|im_start|>')[1]
        results.append(text)

    return results


# ─── Batch image generation (packed sequence, like gen_images_mp.py) ──────────

def batch_generate_images(model, vae_model, tokenizer, new_token_ids,
                          prompts, grpo_config, resolution=512,
                          cfg_text_scale=4.0, cfg_img_scale=1.0,
                          cfg_interval=(0, 1.0), cfg_renorm_min=0.0,
                          cfg_renorm_type="global", timestep_shift=3.0,
                          num_timesteps=15, noise_level=1.3,
                          vae_transform=None, process_index=0):
    """Generate N images in one packed batch forward pass.

    Follows the official Bagel eval pattern (gen_images_mp.py).
    Returns per-sample images, latents, log_probs, and timesteps.

    Args:
        model: Bagel model (unwrapped).
        prompts: List of N prompt strings.
        ... (other args same as inferencer)

    Returns:
        List of N dicts, each with 'image', 'all_latents', 'all_log_probs', 'timesteps'.
    """
    from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache

    N = len(prompts)
    device = next(model.parameters()).device

    def _to_device(d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(device)
        return d

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        # ── Step 1: Encode all N prompts into packed KV cache ──
        past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        kv_lens = [0] * N
        ropes = [0] * N

        generation_input, kv_lens, ropes = model.prepare_prompts(
            curr_kvlens=kv_lens, curr_rope=ropes,
            prompts=prompts, tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        _to_device(generation_input)
        past_key_values = model.forward_cache_update_text(
            past_key_values, **generation_input)

        # ── Step 2: Prepare N VAE latents ──
        generation_input = model.prepare_vae_latent(
            curr_kvlens=kv_lens, curr_rope=ropes,
            image_sizes=[(resolution, resolution)] * N,
            new_token_ids=new_token_ids,
        )
        _to_device(generation_input)

        # ── Step 3: Prepare text-CFG (unconditional) KV cache ──
        cfg_text_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        cfg_kv_lens = [0] * N
        cfg_ropes = [0] * N
        generation_input_cfg_text = model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_kv_lens, curr_rope=cfg_ropes,
            image_sizes=[(resolution, resolution)] * N,
        )
        _to_device(generation_input_cfg_text)

        # ── Step 4: Prepare img-CFG KV cache ──
        cfg_img_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        cfg_img_kv_lens = [0] * N
        cfg_img_ropes = [0] * N
        generation_input_cfg_img = model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_kv_lens, curr_rope=cfg_img_ropes,
            image_sizes=[(resolution, resolution)] * N,
        )
        _to_device(generation_input_cfg_img)

        # ── Step 5: Generate images (packed batch, all N at once) ──
        unpacked_latents, all_latents, all_log_probs, timesteps = model.generate_image(
                past_key_values=past_key_values,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_img_past_key_values=cfg_img_past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=list(cfg_interval),
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                timestep_shift=timestep_shift,
                noise_level=noise_level,
                sample_sde_window_size=grpo_config.sample.sde_window_size,
                sample_sde_window_range=grpo_config.sample.sde_window_range,
                process_index=process_index,
                device=device,
                **generation_input,
                cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
                cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
                cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
                cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
                cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
                cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
                cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
                cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            )

    # ── Step 6: Decode and split per-sample results ──
    results = []
    for i in range(N):
        # Decode image from latent
        latent = unpacked_latents[i]
        h, w = resolution // model.latent_downsample, resolution // model.latent_downsample
        latent = latent.reshape(1, h, w, model.latent_patch_size, model.latent_patch_size, model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, model.latent_channel, h * model.latent_patch_size, w * model.latent_patch_size)
        # Cast to VAE dtype (latent may be float32 from SDE step)
        vae_dtype = next(vae_model.parameters()).dtype
        image = vae_model.decode(latent.to(vae_dtype))
        image = (image.float() * 0.5 + 0.5).clamp(0, 1)[0]

        # Split per-sample latents and log_probs from packed all_latents/all_log_probs
        # all_log_probs[t] is [N] tensor (per-sample means from generate_image)
        per_sample_latents = [lat.split([(unpacked_latents[j].shape[0]) for j in range(N)])[i]
                              for lat in all_latents]
        per_sample_log_probs = [lp[i] for lp in all_log_probs]

        results.append({
            'image': image,
            'all_latents': per_sample_latents,
            'all_log_probs': per_sample_log_probs,
            'timesteps': timesteps,
        })

    return results


# ─── Advantages ──────────────────────────────────────────────────────────────

def compute_group_advantages(flat_rewards: List[float], eps: float = 1e-4) -> torch.Tensor:
    """Compute group-relative advantages from flat reward list."""
    rewards = torch.tensor(flat_rewards, dtype=torch.float32)
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)
