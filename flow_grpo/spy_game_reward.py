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


# ─── Voting (uses model.chat() pattern from original Bagel eval) ─────────────
#
# IMPORTANT: The original Bagel eval uses model.chat() which passes raw PIL
# images directly to prepare_vit_images(transforms=vlm_transform).
# prepare_vit_images internally calls transforms(image) to do resize+normalize.
#
# The VLM image transform must use the correct params from Bagel's eval config:
#   max_image_size=980, min_image_size=378, image_stride=14, max_pixels=2_007_040
# NOT the vit_transform(490,112,7) or vae_transform(512,256,8) from training.
#
# Previous code had TWO bugs:
# 1. Wrong transform params (vit_transform with gen params, not VLM eval params)
# 2. Double transform (vae_transform.resize_transform then vit_transform inside
#    prepare_vit_images — image was resized and potentially normalized twice)


def build_vlm_image_transform():
    """Build the correct VLM image transform matching Bagel's eval config."""
    from flow_grpo.bagel.data.transforms import ImageTransform
    return ImageTransform(
        max_image_size=980,
        min_image_size=378,
        image_stride=14,
        max_pixels=2_007_040,
    )


def run_bagel_vote_multi_image(inferencer, player_images: List[Image.Image],
                                vote_prompt: str,
                                max_tokens: int = 512,
                                temperature: float = 0.7) -> str:
    """Single vote using model.chat() pattern (correct image transform).

    Follows original Bagel eval: raw PIL images → model.chat() →
    prepare_vit_images(transforms=vlm_transform) → generate_text.
    """
    from flow_grpo.bagel.data.data_utils import pil_img2rgb

    vlm_transform = build_vlm_image_transform()

    # Build prompt with player labels (interleaved text + images)
    # model.chat() takes flat list of images and a single text prompt,
    # so we prepend player labels into the prompt text.
    images_rgb = [pil_img2rgb(img) for img in player_images]

    # Construct full prompt: "Player 1's image: [img] Player 2's image: [img] ... <vote_prompt>"
    # Since model.chat() takes images separately and one text prompt,
    # we build the text prompt to reference images by order.
    full_prompt = ""
    for i in range(len(player_images)):
        full_prompt += f"Player {i+1}'s generated image:\n"
    full_prompt += "\n" + vote_prompt

    model = inferencer.model
    tokenizer = inferencer.tokenizer
    new_token_ids = inferencer.new_token_ids

    with torch.no_grad():
        output = model.chat(
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            image_transform=vlm_transform,
            images=images_rgb,
            prompt=full_prompt,
            max_length=max_tokens,
            do_sample=True,
            temperature=temperature,
        )
    return output


def run_bagel_vote_multi_image_repeated(inferencer, player_images: List[Image.Image],
                                         vote_prompt: str,
                                         num_generations: int = 8,
                                         max_tokens: int = 512,
                                         temperature: float = 0.7) -> List[str]:
    """K repeated votes with interleaved text-image input and cached prefill.

    ViT encoding done once → KV cache built once → deepcopy K times for generation.
    Think prompt is merged into vote_prompt (Bagel eval style, avoids extra KV segment).
    """
    from copy import deepcopy
    from flow_grpo.bagel.data.data_utils import pil_img2rgb
    from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache

    vlm_transform = build_vlm_image_transform()
    model = inferencer.model
    tokenizer = inferencer.tokenizer
    new_token_ids = inferencer.new_token_ids
    device = next(model.parameters()).device

    images_rgb = [pil_img2rgb(img) for img in player_images]

    # Think instruction merged into vote prompt (like Bagel eval COT_MC_INSTRUCTION_V2)
    full_prompt = vote_prompt

    results = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # Interleaved: [label] [image] for each player
        for i, image in enumerate(images_rgb):
            gi, newlens, new_rope = model.prepare_prompts(
                curr_kvlens=newlens, curr_rope=new_rope,
                prompts=[f"Player {i+1}'s generated image:"], tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )
            for k, v in gi.items():
                if torch.is_tensor(v): gi[k] = v.to(device)
            past_key_values = model.forward_cache_update_text(past_key_values, **gi)

            gi, newlens, new_rope = model.prepare_vit_images(
                curr_kvlens=newlens, curr_rope=new_rope,
                images=[image], transforms=vlm_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in gi.items():
                if torch.is_tensor(v): gi[k] = v.to(device)
            past_key_values = model.forward_cache_update_vit(past_key_values, **gi)

        # Think + vote prompt (single text segment)
        gi, newlens, new_rope = model.prepare_prompts(
            curr_kvlens=newlens, curr_rope=new_rope,
            prompts=["\n" + full_prompt], tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        for k, v in gi.items():
            if torch.is_tensor(v): gi[k] = v.to(device)
        past_key_values = model.forward_cache_update_text(past_key_values, **gi)

        # ── Batched generation: replicate packed KV cache K times ──
        # NaiveCache KV format: [seq_len, heads, dim] (packed, no batch dim)
        # K copies: [K*seq_len, heads, dim] via repeat (concatenation in packed space)
        # prepare_start_tokens naturally builds correct indexes for K sequences
        batch_kv = NaiveCache(model.config.llm_config.num_hidden_layers)
        for layer_idx in range(model.config.llm_config.num_hidden_layers):
            orig_k = past_key_values.key_cache[layer_idx]  # [S, heads, dim]
            orig_v = past_key_values.value_cache[layer_idx]
            batch_kv.key_cache[layer_idx] = orig_k.repeat(num_generations, 1, 1)
            batch_kv.value_cache[layer_idx] = orig_v.repeat(num_generations, 1, 1)

        batch_newlens = newlens * num_generations   # [S, S, ..., S]
        batch_new_rope = new_rope * num_generations
        gen_input = model.prepare_start_tokens(batch_newlens, batch_new_rope, new_token_ids)
        for k, v in gen_input.items():
            if torch.is_tensor(v): gen_input[k] = v.to(device)

        unpacked_latent = model.generate_text(
            past_key_values=batch_kv,
            max_length=max_tokens,
            do_sample=True,
            temperature=temperature,
            end_token_id=new_token_ids['eos_token_id'],
            **gen_input,
        )
        del batch_kv

    # unpacked_latent: [max_steps+1, K]
    for b in range(num_generations):
        output = tokenizer.decode(unpacked_latent[:, b])
        output = output.split('<|im_end|>')[0]
        if '<|im_start|>' in output:
            output = output.split('<|im_start|>')[1]
        results.append(output)

    return results


def run_bagel_votes_multi_prompt(inferencer, player_images: List[Image.Image],
                                  vote_prompts: List[str],
                                  max_tokens: int = 512,
                                  temperature: float = 0.7) -> List[str]:
    """Vote with N separate player images and different prompts per voter.

    Encodes all N player images into ViT KV cache ONCE, then for each
    voting prompt: deepcopy image KV cache → append text → generate.

    This is the correct approach: each image gets its own ViT tokens in the
    KV cache (with START_OF_IMAGE / END_OF_IMAGE markers), and the text
    prompt references them by order ("Player 1's generated image:", etc.).

    Args:
        inferencer: Bagel inferencer with model, tokenizer, new_token_ids.
        player_images: List of N PIL images, one per player.
        vote_prompts: List of M voting prompts (one per voter).
        max_tokens: Max generation tokens per vote.
        temperature: Sampling temperature.

    Returns:
        List of M vote text outputs.
    """
    from copy import deepcopy
    from flow_grpo.bagel.data.data_utils import pil_img2rgb
    from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache

    vlm_transform = build_vlm_image_transform()
    model = inferencer.model
    tokenizer = inferencer.tokenizer
    new_token_ids = inferencer.new_token_ids
    device = next(model.parameters()).device

    images_rgb = [pil_img2rgb(img) for img in player_images]

    # Build image label prefix (same for all prompts)
    image_prefix = ""
    for i in range(len(player_images)):
        image_prefix += f"Player {i+1}'s generated image:\n"

    results = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        # Step 1: Encode all N player images into KV cache (expensive, done ONCE)
        img_kv = NaiveCache(model.config.llm_config.num_hidden_layers)
        img_lens = [0]
        img_rope = [0]

        for image in images_rgb:
            generation_input, img_lens, img_rope = model.prepare_vit_images(
                curr_kvlens=img_lens, curr_rope=img_rope,
                images=[image], transforms=vlm_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            img_kv = model.forward_cache_update_vit(
                img_kv, **generation_input)

        # Step 2: For each voter prompt, deepcopy image KV → append text → generate
        for vote_prompt in vote_prompts:
            full_prompt = image_prefix + "\n" + vote_prompt

            kv_copy = deepcopy(img_kv)
            text_lens = list(img_lens)
            text_rope = list(img_rope)

            generation_input, text_lens, text_rope = model.prepare_prompts(
                curr_kvlens=text_lens, curr_rope=text_rope,
                prompts=[full_prompt], tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            kv_copy = model.forward_cache_update_text(
                kv_copy, **generation_input)

            gen_input = model.prepare_start_tokens(text_lens, text_rope, new_token_ids)
            for k, v in gen_input.items():
                if torch.is_tensor(v):
                    gen_input[k] = v.to(device)
            unpacked_latent = model.generate_text(
                past_key_values=kv_copy,
                max_length=max_tokens,
                do_sample=True,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **gen_input,
            )
            output = tokenizer.decode(unpacked_latent[:, 0])
            output = output.split('<|im_end|>')[0]
            if '<|im_start|>' in output:
                output = output.split('<|im_start|>')[1]
            results.append(output)
            del kv_copy

    return results


# Legacy functions kept for backward compatibility

def run_bagel_vote(inferencer, grid_image: Image.Image, vote_prompt: str,
                   max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Single vote on a grid image (legacy)."""
    return run_bagel_vote_multi_image(inferencer, [grid_image], vote_prompt,
                                       max_tokens, temperature)


def run_bagel_votes_cached(inferencer, grid_image: Image.Image,
                           vote_prompts: List[str],
                           max_tokens: int = 512,
                           temperature: float = 0.7) -> List[str]:
    """Multiple votes with different prompts on same image (legacy)."""
    results = []
    for prompt in vote_prompts:
        results.append(run_bagel_vote_multi_image(
            inferencer, [grid_image], prompt, max_tokens, temperature))
    return results


def run_bagel_votes_batch(model, tokenizer, new_token_ids, vit_transform,
                          grid_images: List[Image.Image],
                          vote_prompts: List[str],
                          max_tokens: int = 512,
                          temperature: float = 0.7) -> List[str]:
    """Batch vote (legacy — redirects to correct implementation)."""
    # This was buggy (double transform). Use single-vote loop instead.
    from flow_grpo.bagel.data.data_utils import pil_img2rgb
    from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache

    vlm_transform = build_vlm_image_transform()
    N = len(vote_prompts)
    device = next(model.parameters()).device

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        kv_lens = [0] * N
        ropes = [0] * N

        images_rgb = [pil_img2rgb(img) for img in grid_images]
        generation_input, kv_lens, ropes = model.prepare_vit_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=images_rgb,
            transforms=vlm_transform,
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if isinstance(v, torch.Tensor):
                generation_input[k] = v.to(device)
        past_key_values = model.forward_cache_update_vit(past_key_values, **generation_input)

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

        gen_input = model.prepare_start_tokens(kv_lens, ropes, new_token_ids)
        for k, v in gen_input.items():
            if isinstance(v, torch.Tensor):
                gen_input[k] = v.to(device)

        output_tokens = model.generate_text(
            past_key_values=past_key_values,
            max_length=max_tokens,
            do_sample=True,
            temperature=temperature,
            end_token_id=new_token_ids['eos_token_id'],
            **gen_input,
        )

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


# ─── Decision Phase GRPO (Vision-Zero style) ────────────────────────────────

def sample_vote_with_logprobs(inferencer, player_images: List[Image.Image],
                               vote_prompt: str, num_generations: int = 8,
                               max_tokens: int = 512, temperature: float = 0.7,
                               return_base_kv: bool = False):
    """Sample K votes and return token ids + log_probs for GRPO training.

    Args:
        return_base_kv: If True, also return the base KV cache (before K-repeat)
            for reuse in decision backward, avoiding redundant VIT+text encoding.

    Returns:
        List of K dicts, each with:
          'text': str, 'token_ids': Tensor [seq_len], 'log_probs': Tensor [seq_len],
          'prompt_len': int (KV cache length before generation)
        If return_base_kv=True, returns (results, cached_kv_tuple) where
        cached_kv_tuple = (past_key_values, newlens, new_rope, device, lm, extra_inputs).
    """
    from copy import deepcopy
    from flow_grpo.bagel.data.data_utils import pil_img2rgb
    from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache

    vlm_transform = build_vlm_image_transform()
    model = inferencer.model
    tokenizer = inferencer.tokenizer
    new_token_ids = inferencer.new_token_ids
    device = next(model.parameters()).device

    images_rgb = [pil_img2rgb(img) for img in player_images]

    # Think instruction merged into vote prompt (Bagel eval style)
    full_prompt = vote_prompt

    results = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # Interleaved: [label] [image] for each player
        for i, image in enumerate(images_rgb):
            gi, newlens, new_rope = model.prepare_prompts(
                curr_kvlens=newlens, curr_rope=new_rope,
                prompts=[f"Player {i+1}'s generated image:"], tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )
            for k, v in gi.items():
                if torch.is_tensor(v): gi[k] = v.to(device)
            past_key_values = model.forward_cache_update_text(past_key_values, **gi)

            gi, newlens, new_rope = model.prepare_vit_images(
                curr_kvlens=newlens, curr_rope=new_rope,
                images=[image], transforms=vlm_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in gi.items():
                if torch.is_tensor(v): gi[k] = v.to(device)
            past_key_values = model.forward_cache_update_vit(past_key_values, **gi)

        # Think + vote prompt (single text segment)
        gi, newlens, new_rope = model.prepare_prompts(
            curr_kvlens=newlens, curr_rope=new_rope,
            prompts=["\n" + full_prompt], tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        for k, v in gi.items():
            if torch.is_tensor(v): gi[k] = v.to(device)
        past_key_values = model.forward_cache_update_text(past_key_values, **gi)

        prompt_len = newlens[0]

        # Save base KV cache for reuse in decision backward (before K-repeat)
        if return_base_kv:
            lm = model.language_model
            extra_inputs = {"mode": "und"} if model.use_moe else {}
            base_cached_kv = (past_key_values, list(newlens), list(new_rope),
                              device, lm, extra_inputs)
        else:
            base_cached_kv = None

        # ── Batched generation: K votes in one packed forward pass ──
        batch_kv = NaiveCache(model.config.llm_config.num_hidden_layers)
        for layer_idx in range(model.config.llm_config.num_hidden_layers):
            orig_k = past_key_values.key_cache[layer_idx]
            orig_v = past_key_values.value_cache[layer_idx]
            batch_kv.key_cache[layer_idx] = orig_k.repeat(num_generations, 1, 1)
            batch_kv.value_cache[layer_idx] = orig_v.repeat(num_generations, 1, 1)

        batch_newlens = newlens * num_generations
        batch_new_rope = new_rope * num_generations
        gen_input = model.prepare_start_tokens(batch_newlens, batch_new_rope, new_token_ids)
        for k, v in gen_input.items():
            if torch.is_tensor(v): gen_input[k] = v.to(device)

        gen_ids, gen_logps = model.generate_text_with_logprobs(
            past_key_values=batch_kv,
            max_length=max_tokens,
            temperature=temperature,
            end_token_id=new_token_ids['eos_token_id'],
            **gen_input,
        )
        # gen_ids: [max_steps+1, K], gen_logps: [max_steps, K]
        del batch_kv

    # Split batch results into K individual samples
    eos_id = new_token_ids['eos_token_id']
    eos_val = eos_id.item() if torch.is_tensor(eos_id) else eos_id
    for b in range(num_generations):
        ids_b = gen_ids[:, b]
        logps_b = gen_logps[:, b]

        # Find EOS position
        eos_pos = (ids_b == eos_val).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            end = eos_pos[0].item()
            token_ids = ids_b[1:end]
            log_probs = logps_b[:max(end - 1, 0)]
        else:
            token_ids = ids_b[1:]
            log_probs = logps_b

        text = tokenizer.decode(ids_b)
        text = text.split('<|im_end|>')[0]
        if '<|im_start|>' in text:
            text = text.split('<|im_start|>')[1]

        results.append({
            'text': text,
            'token_ids': token_ids.cpu(),
            'log_probs': log_probs.cpu(),
            'prompt_len': prompt_len,
        })

    if return_base_kv:
        return results, base_cached_kv
    return results


def batch_sample_vote_with_logprobs(inferencer, all_player_images: List[List[Image.Image]],
                                     all_vote_prompts: List[str],
                                     num_generations: int = 8,
                                     max_tokens: int = 512, temperature: float = 0.7,
                                     return_base_kv: bool = False):
    """Batched voting across multiple games — builds KV caches for all games in
    one packed pass, then generates G*K votes in a single batched forward.

    Args:
        all_player_images: List of G lists, each containing N player PIL images.
        all_vote_prompts: List of G vote prompt strings.
        num_generations: K votes per game.
        return_base_kv: If True, return per-game base KV caches for decision backward.

    Returns:
        List of G results, each is (vote_samples, cached_kv) if return_base_kv
        else just vote_samples (List of K dicts).
    """
    from flow_grpo.bagel.data.data_utils import pil_img2rgb
    from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache

    vlm_transform = build_vlm_image_transform()
    model = inferencer.model
    tokenizer = inferencer.tokenizer
    new_token_ids = inferencer.new_token_ids
    device = next(model.parameters()).device

    num_games = len(all_player_images)
    num_players_per_game = len(all_player_images[0])

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        newlens = [0] * num_games
        new_rope = [0] * num_games

        # Interleaved: [label] [image] for each player, across all games in parallel
        for pid in range(num_players_per_game):
            # Text labels for this player across all games
            prompts = [f"Player {pid+1}'s generated image:"] * num_games
            gi, newlens, new_rope = model.prepare_prompts(
                curr_kvlens=newlens, curr_rope=new_rope,
                prompts=prompts, tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )
            for k, v in gi.items():
                if torch.is_tensor(v): gi[k] = v.to(device)
            past_key_values = model.forward_cache_update_text(past_key_values, **gi)

            # Images for this player across all games
            images_rgb = [pil_img2rgb(all_player_images[g][pid]) for g in range(num_games)]
            gi, newlens, new_rope = model.prepare_vit_images(
                curr_kvlens=newlens, curr_rope=new_rope,
                images=images_rgb, transforms=vlm_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in gi.items():
                if torch.is_tensor(v): gi[k] = v.to(device)
            past_key_values = model.forward_cache_update_vit(past_key_values, **gi)

        # Vote prompts (different per game)
        vote_prompts = ["\n" + p for p in all_vote_prompts]
        gi, newlens, new_rope = model.prepare_prompts(
            curr_kvlens=newlens, curr_rope=new_rope,
            prompts=vote_prompts, tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        for k, v in gi.items():
            if torch.is_tensor(v): gi[k] = v.to(device)
        past_key_values = model.forward_cache_update_text(past_key_values, **gi)

        prompt_lens = list(newlens)  # per-game prompt lengths

        # Compute per-game cache boundaries (used for both KV extraction and K-repeat)
        cum_lens = [0]
        for nl in newlens:
            cum_lens.append(cum_lens[-1] + nl)

        # Save per-game base KV caches for decision backward reuse
        # Extract each game's slice from the packed cache so compute_text_grpo_loss
        # receives a single-game cache (not the full packed multi-game cache).
        base_cached_kvs = [None] * num_games
        if return_base_kv:
            lm = model.language_model
            extra_inputs = {"mode": "und"} if model.use_moe else {}
            num_layers = model.config.llm_config.num_hidden_layers
            for g in range(num_games):
                start, end = cum_lens[g], cum_lens[g + 1]
                game_kv = NaiveCache(num_layers)
                for li in range(num_layers):
                    game_kv.key_cache[li] = past_key_values.key_cache[li][start:end]
                    game_kv.value_cache[li] = past_key_values.value_cache[li][start:end]
                base_cached_kvs[g] = (game_kv, [newlens[g]], [new_rope[g]],
                                       device, lm, extra_inputs)

        # ── Replicate each game's KV cache K times for batched generation ──
        # Current packed cache: game0_kv | game1_kv | ... in contiguous memory
        # Need: game0_copy1 | game0_copy2 | ... | game1_copy1 | game1_copy2 | ...

        batch_kv = NaiveCache(model.config.llm_config.num_hidden_layers)
        K = num_generations
        for layer_idx in range(model.config.llm_config.num_hidden_layers):
            orig_k = past_key_values.key_cache[layer_idx]
            orig_v = past_key_values.value_cache[layer_idx]
            # Extract and repeat each game's slice
            k_parts = []
            v_parts = []
            for g in range(num_games):
                start, end = cum_lens[g], cum_lens[g + 1]
                k_parts.append(orig_k[start:end].repeat(K, 1, 1))
                v_parts.append(orig_v[start:end].repeat(K, 1, 1))
            batch_kv.key_cache[layer_idx] = torch.cat(k_parts, dim=0)
            batch_kv.value_cache[layer_idx] = torch.cat(v_parts, dim=0)

        # Build newlens/new_rope for G*K sequences
        batch_newlens = []
        batch_new_rope = []
        for g in range(num_games):
            batch_newlens.extend([newlens[g]] * K)
            batch_new_rope.extend([new_rope[g]] * K)

        gen_input = model.prepare_start_tokens(batch_newlens, batch_new_rope, new_token_ids)
        for k, v in gen_input.items():
            if torch.is_tensor(v): gen_input[k] = v.to(device)

        gen_ids, gen_logps = model.generate_text_with_logprobs(
            past_key_values=batch_kv,
            max_length=max_tokens,
            temperature=temperature,
            end_token_id=new_token_ids['eos_token_id'],
            **gen_input,
        )
        # gen_ids: [max_steps+1, G*K], gen_logps: [max_steps, G*K]
        del batch_kv

    # Split results by game, then by K
    eos_id = new_token_ids['eos_token_id']
    eos_val = eos_id.item() if torch.is_tensor(eos_id) else eos_id

    all_results = []
    for g in range(num_games):
        game_results = []
        for ki in range(K):
            b = g * K + ki
            ids_b = gen_ids[:, b]
            logps_b = gen_logps[:, b]

            eos_pos = (ids_b == eos_val).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                end = eos_pos[0].item()
                token_ids = ids_b[1:end]
                log_probs = logps_b[:max(end - 1, 0)]
            else:
                token_ids = ids_b[1:]
                log_probs = logps_b

            text = tokenizer.decode(ids_b)
            text = text.split('<|im_end|>')[0]
            if '<|im_start|>' in text:
                text = text.split('<|im_start|>')[1]

            game_results.append({
                'text': text,
                'token_ids': token_ids.cpu(),
                'log_probs': log_probs.cpu(),
                'prompt_len': prompt_lens[g],
            })
        all_results.append(game_results)

    if return_base_kv:
        return [(all_results[g], base_cached_kvs[g]) for g in range(num_games)]
    return all_results


def compute_decision_rewards(vote_samples, spy_player, num_players, extract_vote_fn,
                              lambda_fmt=0.3, beta_acc=1.2, gamma=0.99):
    """Compute per-vote decision rewards (Vision-Zero PBRS style).

    Args:
        vote_samples: List of K dicts from sample_vote_with_logprobs
        spy_player: int, the actual spy player id
        num_players: int
        extract_vote_fn: function to extract vote from text
        lambda_fmt: format reward weight
        beta_acc: accuracy reward weight
        gamma: PBRS discount factor

    Returns:
        List of K float rewards
    """
    rewards = []
    for vs in vote_samples:
        text = vs['text']
        extracted = extract_vote_fn(text)

        # Format check: reasoning content + single valid answer
        import re as _re
        has_answer = extracted is not None and extracted.get("voted_spy") is not None

        # Check for reasoning (flexible: <think> tags OR free text before answer)
        has_reasoning = False
        think_match = _re.search(r'<think>(.*?)</think>', text, _re.DOTALL)
        if think_match and len(think_match.group(1).strip()) > 10:
            has_reasoning = True  # explicit <think> tags with content
        else:
            # No <think> tags — check for free-form reasoning before the answer
            boxed_pos = _re.search(r'\\\\?boxed\{', text)
            answer_pos = _re.search(r'<answer>', text)
            ans_start = None
            if boxed_pos:
                ans_start = boxed_pos.start()
            elif answer_pos:
                ans_start = answer_pos.start()
            if ans_start is not None and len(text[:ans_start].strip()) > 10:
                has_reasoning = True  # substantial text before answer

        # Penalize multiple answers: multiple tags OR multiple numbers in single tag
        num_boxed = len(_re.findall(r'\\\\?boxed\{', text))
        num_answer_tags = len(_re.findall(r'<answer>', text))
        has_multiple_answers = (num_boxed > 1) or (num_answer_tags > 1)
        # extract_vote returns None when answer content has multiple numbers
        # (e.g. \boxed{1, 3}) — also treat as multiple answers
        if not has_multiple_answers and extracted is None and (num_boxed > 0 or num_answer_tags > 0):
            has_multiple_answers = True

        fmt_ok = has_reasoning and has_answer and not has_multiple_answers

        # Accuracy check
        if extracted and isinstance(extracted.get("voted_spy"), int):
            voted = extracted["voted_spy"]
            if voted == spy_player:
                acc = 1.0
            else:
                acc = -1.0
        elif extracted and extracted.get("voted_spy") == "N/A":
            acc = -0.8
        else:
            acc = -1.0

        # PBRS reward
        r_fmt = lambda_fmt * (2 * int(fmt_ok) - 1)
        r_acc = acc * beta_acc
        phi_next = 1.0 if acc > 0 else 0.0
        shaped = gamma * phi_next
        reward = r_fmt + r_acc + shaped

        rewards.append(reward)
    return rewards


def build_vote_kv_cache(model, tokenizer, new_token_ids, player_images, vote_prompt):
    """Build KV cache for voting context (images + prompt). Reusable across K vote samples.

    Returns (past_key_values, newlens, new_rope, device, lm, extra_inputs).
    lm is the (possibly FSDP-wrapped) language model — call through it for FSDP safety.
    """
    from flow_grpo.bagel.data.data_utils import pil_img2rgb
    from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache

    vlm_transform = build_vlm_image_transform()
    device = next(model.parameters()).device
    images_rgb = [pil_img2rgb(img) for img in player_images]

    lm = model.language_model
    extra_inputs = {"mode": "und"} if model.use_moe else {}

    full_prompt = vote_prompt

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        for i, image in enumerate(images_rgb):
            label = f"Player {i+1}'s generated image:"
            gi, newlens, new_rope = model.prepare_prompts(
                curr_kvlens=newlens, curr_rope=new_rope,
                prompts=[label], tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )
            for k, v in gi.items():
                if torch.is_tensor(v): gi[k] = v.to(device)
            past_key_values = model.forward_cache_update_text(past_key_values, **gi)

            gi, newlens, new_rope = model.prepare_vit_images(
                curr_kvlens=newlens, curr_rope=new_rope,
                images=[image], transforms=vlm_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in gi.items():
                if torch.is_tensor(v): gi[k] = v.to(device)
            past_key_values = model.forward_cache_update_vit(past_key_values, **gi)

        gi, newlens, new_rope = model.prepare_prompts(
            curr_kvlens=newlens, curr_rope=new_rope,
            prompts=["\n" + full_prompt], tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        for k, v in gi.items():
            if torch.is_tensor(v): gi[k] = v.to(device)
        past_key_values = model.forward_cache_update_text(past_key_values, **gi)

    return past_key_values, newlens, new_rope, device, lm, extra_inputs


def compute_text_grpo_loss(model, tokenizer, new_token_ids, inferencer,
                            player_images, vote_prompt, vote_sample,
                            advantage, clip_range=0.2, kl_beta=0.04,
                            max_tokens=512, temperature=0.7,
                            cached_kv=None):
    """Compute text GRPO loss for one vote sample (Vision-Zero decision phase).

    Uses batched forward (ONE LLM forward for all tokens) instead of per-token loop.
    Includes KL penalty against frozen ref model (Vision-Zero: beta=0.04).

    Args:
        model: Bagel model (with language_model and optionally language_model_ref)
        vote_sample: dict with 'token_ids', 'log_probs', 'prompt_len'
        advantage: float, the advantage for this vote
        kl_beta: KL penalty coefficient (Vision-Zero: 0.04)
    Returns:
        loss: scalar tensor
        info: dict with metrics
    """
    from flow_grpo.bagel.data.data_utils import pil_img2rgb
    from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache

    vlm_transform = build_vlm_image_transform()
    device = next(model.parameters()).device

    images_rgb = [pil_img2rgb(img) for img in player_images]

    old_log_probs = vote_sample['log_probs'].to(device)
    target_ids = vote_sample['token_ids'].to(device)
    seq_len = len(target_ids)

    if seq_len == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            'text_policy_loss': 0.0, 'text_clipfrac': 0.0}

    # ── Step 1: Use cached KV or build from scratch ──
    if cached_kv is not None:
        # Handle both 6-tuple (from build_vote_kv_cache) and 7-tuple (from batch_sample with game index)
        if len(cached_kv) == 7:
            past_key_values, newlens, new_rope, device, lm, extra_inputs, _game_idx = cached_kv
        else:
            past_key_values, newlens, new_rope, device, lm, extra_inputs = cached_kv
    else:
        past_key_values, newlens, new_rope, device, lm, extra_inputs = \
            build_vote_kv_cache(model, tokenizer, new_token_ids, player_images, vote_prompt)

    # ── Step 2: Batched forward — ALL completion tokens in ONE pass (with grad) ──
    # input_ids: [bos, tok0, tok1, ..., tok_{n-2}]  (shifted right)
    # labels:    [tok0, tok1, ..., tok_{n-1}]        (target_ids)
    bos = new_token_ids['bos_token_id']
    bos_t = bos.unsqueeze(0).to(device) if torch.is_tensor(bos) else torch.tensor([bos], device=device)
    input_ids = torch.cat([bos_t, target_ids[:-1]])  # [seq_len] shifted

    kv_len = newlens[0]
    rope_start = new_rope[0]

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        # FSDP-safe: all calls go through lm() which triggers proper unshard/reshard
        packed_text_embedding = lm(mode="get_embeddings", input_ids=input_ids)  # [seq_len, hidden]
        query_lens = torch.tensor([seq_len], dtype=torch.int, device=device)
        packed_query_indexes = torch.arange(kv_len, kv_len + seq_len, device=device)
        packed_query_position_ids = torch.arange(rope_start, rope_start + seq_len,
                                                  dtype=torch.long, device=device)
        pki = torch.arange(kv_len, device=device)

        output = lm(
            packed_query_sequence=packed_text_embedding,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=torch.tensor(newlens, dtype=torch.int, device=device),
            packed_key_value_indexes=pki,
            update_past_key_values=False,  # Don't need updated KV cache
            is_causal=True,
            **extra_inputs,
        )

        logits = lm(mode="compute_logits", hidden_state=output.packed_query_sequence)  # [seq_len, vocab]
        log_probs_all = F.log_softmax(logits.float() / temperature, dim=-1)
        # Gather log_probs for target tokens
        new_log_probs = log_probs_all.gather(1, target_ids.unsqueeze(1)).squeeze(1)  # [seq_len]

    # ── Step 3: KL penalty from ref model (Vision-Zero: beta=0.04) ──
    per_token_kl = torch.zeros(seq_len, device=device)
    if kl_beta > 0 and hasattr(model, 'language_model_ref'):
        ref_lm = model.language_model_ref
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            # Call through FSDP wrapper (same pattern as _forward_flow with ref_model=True)
            ref_embeddings = ref_lm(mode="get_embeddings", input_ids=input_ids)
            ref_output = ref_lm(
                packed_query_sequence=ref_embeddings.detach(),
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=torch.tensor(newlens, dtype=torch.int, device=device),
                packed_key_value_indexes=pki,
                update_past_key_values=False,
                is_causal=True,
                **extra_inputs,
            )
            ref_logits = ref_lm(mode="compute_logits", hidden_state=ref_output.packed_query_sequence)
            ref_log_probs_all = F.log_softmax(ref_logits.float() / temperature, dim=-1)
            ref_log_probs = ref_log_probs_all.gather(1, target_ids.unsqueeze(1)).squeeze(1)

        # Vision-Zero KL: 0.5 * [exp(ref - new) - (ref - new) - 1]
        # The 0.5 factor matches Vision-Zero's grpo_trainer.py (reverse KL approximation).
        # ref_log_probs is detached (computed under no_grad), gradients flow through new_log_probs
        per_token_kl = 0.5 * (torch.exp(ref_log_probs - new_log_probs) -
                              (ref_log_probs - new_log_probs) - 1)

    # ── Step 4: PPO-clip loss ──
    adv = torch.tensor(advantage, dtype=torch.float32, device=device)
    ratio = torch.exp(new_log_probs - old_log_probs[:seq_len])
    per_token_loss1 = ratio * adv
    per_token_loss2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    # Add KL penalty
    if kl_beta > 0:
        per_token_loss = per_token_loss + kl_beta * per_token_kl

    loss = per_token_loss.mean()

    clipfrac = ((ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)).float().mean().item()

    return loss, {
        'text_policy_loss': loss.item(),
        'text_clipfrac': clipfrac,
        'text_ratio_mean': ratio.mean().item(),
        'text_kl_mean': per_token_kl.mean().item() if kl_beta > 0 else 0.0,
        'text_seq_len': seq_len,
    }
