"""
Debug: Compare native Bagel model.chat() vs our repeated vote function.
Same prompt, same images, multiple temperatures.

CUDA_VISIBLE_DEVICES=2 conda run -n spy_bagel python scripts/debug_vote_comparison.py
"""

import os
import sys
import json
import torch
import time
from copy import deepcopy
from collections import Counter
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from flow_grpo.bagel.data.data_utils import add_special_tokens, pil_img2rgb
from flow_grpo.bagel.data.transforms import ImageTransform
from flow_grpo.bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from flow_grpo.bagel.modeling.qwen2 import Qwen2Tokenizer
from flow_grpo.bagel.modeling.autoencoder import load_ae
from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache
from flow_grpo.bagel.inferencer import InterleaveInferencer
from flow_grpo.spy_game_data import SpyGameDataGenerator
from flow_grpo.spy_game_reward import (
    run_bagel_vote_multi_image,
    run_bagel_vote_multi_image_repeated,
    build_vlm_image_transform,
    tensor_images_to_pil,
)

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.hooks import remove_hook_from_module
from huggingface_hub import snapshot_download

OUT_DIR = "/adialab/usr/shadabk/MedUMM/flow_grpo/test_samples/debug_votes"
TEMPERATURES = [0.3, 0.7, 1.0]
K = 4  # votes per temperature


def load_model(device="cuda:0"):
    model_name = "ByteDance-Seed/BAGEL-7B-MoT"
    dtype = torch.bfloat16
    model_path = snapshot_download(repo_id=model_name)

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    bagel_config = BagelConfig(
        visual_gen=True, visual_und=True,
        llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2, max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, bagel_config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(512, 256, 8)
    vit_transform = ImageTransform(490, 112, 7)

    model = load_checkpoint_and_dispatch(
        model, checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map={"": device}, offload_buffers=False,
        dtype=dtype, force_hooks=True, offload_folder="/adialab/usr/shadabk/MedUMM/.offload"
    )
    for module in model.modules():
        remove_hook_from_module(module)
    model = model.eval()
    torch.cuda.empty_cache()

    vae_model.requires_grad_(False)
    model.requires_grad_(False)
    vae_model.to(device, dtype=dtype)
    model.to(device, dtype=dtype)

    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=vae_transform, vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer


def native_chat_vote(model, tokenizer, new_token_ids, vlm_transform,
                     images_rgb, prompt, max_tokens=512, temperature=0.7):
    """Pure model.chat() — the gold standard from Bagel eval."""
    with torch.no_grad():
        output = model.chat(
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            image_transform=vlm_transform,
            images=images_rgb,
            prompt=prompt,
            max_length=max_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 0.01),
        )
    return output


def native_chat_vote_repeated(model, tokenizer, new_token_ids, vlm_transform,
                               images_rgb, prompt, num_generations=4,
                               max_tokens=512, temperature=0.7):
    """Repeated model.chat() — call K times independently (no caching)."""
    results = []
    for _ in range(num_generations):
        out = native_chat_vote(model, tokenizer, new_token_ids, vlm_transform,
                               images_rgb, prompt, max_tokens, temperature)
        results.append(out)
    return results


def cached_vote_repeated(model, tokenizer, new_token_ids, vlm_transform,
                          images_rgb, prompt, num_generations=4,
                          max_tokens=512, temperature=0.7):
    """Our cached implementation: build KV once, deepcopy + gen_text K times."""
    device = next(model.parameters()).device
    results = []

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        for image in images_rgb:
            generation_input, newlens, new_rope = model.prepare_vit_images(
                curr_kvlens=newlens, curr_rope=new_rope,
                images=[image], transforms=vlm_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            past_key_values = model.forward_cache_update_vit(past_key_values, **generation_input)

        generation_input, newlens, new_rope = model.prepare_prompts(
            curr_kvlens=newlens, curr_rope=new_rope,
            prompts=[prompt], tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)

        for _ in range(num_generations):
            kv_copy = deepcopy(past_key_values)
            gen_input = model.prepare_start_tokens(newlens, new_rope, new_token_ids)
            for k, v in gen_input.items():
                if torch.is_tensor(v):
                    gen_input[k] = v.to(device)
            unpacked_latent = model.generate_text(
                past_key_values=kv_copy,
                max_length=max_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 0.01),
                end_token_id=new_token_ids['eos_token_id'],
                **gen_input,
            )
            output = tokenizer.decode(unpacked_latent[:, 0])
            output = output.split('<|im_end|>')[0]
            if '<|im_start|>' in output:
                output = output.split('<|im_start|>')[1]
            results.append(output)

    return results


def generate_image(inferencer, prompt):
    """Generate one image for testing."""
    import ml_collections
    mock_config = ml_collections.ConfigDict()
    mock_config.sample = ml_collections.ConfigDict()
    mock_config.sample.sde_window_size = 0
    mock_config.sample.sde_window_range = (0, 0)
    mock_config.train = ml_collections.ConfigDict()
    mock_config.train.clip_range_lt = 0.2
    mock_config.train.clip_range_gt = 0.2
    mock_config.train.beta = 0

    with torch.no_grad():
        output = inferencer.interleave_inference(
            input_lists=[prompt], understanding_output=False,
            do_sample=True, text_temperature=1.0,
            image_shapes=(512, 512), cfg_text_scale=4.0,
            cfg_img_scale=1.0, cfg_interval=[0, 1.0],
            num_timesteps=50, timestep_shift=3.0,
            cfg_renorm_min=0.0, cfg_renorm_type="global",
            grpo_config=mock_config,
        )
    result = output[0] if isinstance(output, list) else output
    if isinstance(result, dict) and 'image' in result:
        img = result['image']
    else:
        img = result
    if isinstance(img, torch.Tensor):
        img = tensor_images_to_pil(img.unsqueeze(0) if img.dim() == 3 else img)[0]
    return img


def main():
    device = "cuda:0"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Debug: Native model.chat() vs Cached repeated vote")
    print("=" * 70)

    # Load
    print("\n[1] Loading model...")
    inferencer = load_model(device)
    model = inferencer.model
    tokenizer = inferencer.tokenizer
    new_token_ids = inferencer.new_token_ids
    vlm_transform = build_vlm_image_transform()

    # Generate game + images
    print("\n[2] Generating game and images...")
    game_gen = SpyGameDataGenerator(num_players=4)
    game_data = game_gen.generate_game(epoch=200, sample_idx=7)
    spy_pid = game_data["spy_player"]
    print(f"  Spy: P{spy_pid}")

    pil_images = []
    for pid in range(1, 5):
        prompt = game_gen.format_generation_prompt_simple(game_data, pid)
        img = generate_image(inferencer, prompt)
        img.save(os.path.join(OUT_DIR, f"player_{pid}.png"))
        pil_images.append(img)
        print(f"  P{pid} image generated")

    images_rgb = [pil_img2rgb(img) for img in pil_images]

    # Use Player 2's vote prompt for comparison
    test_pid = 2
    vote_prompt = game_gen.format_voting_prompt(game_data, player_id=test_pid)
    role = "SPY" if test_pid == spy_pid else "CIV"

    # Build the full prompt (same as run_bagel_vote_multi_image builds it)
    full_prompt = ""
    for i in range(len(pil_images)):
        full_prompt += f"Player {i+1}'s generated image:\n"
    full_prompt += "\n" + vote_prompt

    # Save prompt for reference
    with open(os.path.join(OUT_DIR, "prompt.txt"), "w") as f:
        f.write(f"Test player: P{test_pid} [{role}]\n")
        f.write(f"Spy: P{spy_pid}\n\n")
        f.write(f"Vote prompt:\n{vote_prompt}\n\n")
        f.write(f"Full prompt (with player labels):\n{full_prompt}\n")

    print(f"\n[3] Comparing methods — P{test_pid} [{role}], Spy=P{spy_pid}")
    print(f"  Temperatures: {TEMPERATURES}")
    print(f"  K={K} votes per temperature per method")

    results = {}

    for temp in TEMPERATURES:
        print(f"\n{'='*70}")
        print(f"  Temperature = {temp}")
        print(f"{'='*70}")

        # Method A: Native model.chat() — K independent calls
        print(f"\n  [A] Native model.chat() x {K}...")
        t0 = time.time()
        native_results = native_chat_vote_repeated(
            model, tokenizer, new_token_ids, vlm_transform,
            images_rgb, full_prompt, num_generations=K,
            max_tokens=512, temperature=temp,
        )
        t_native = time.time() - t0

        # Method B: Cached KV + deepcopy — our implementation
        print(f"  [B] Cached KV repeated x {K}...")
        t0 = time.time()
        cached_results = cached_vote_repeated(
            model, tokenizer, new_token_ids, vlm_transform,
            images_rgb, full_prompt, num_generations=K,
            max_tokens=512, temperature=temp,
        )
        t_cached = time.time() - t0

        # Method C: Our spy_game_reward function
        print(f"  [C] run_bagel_vote_multi_image_repeated() x {K}...")
        t0 = time.time()
        with torch.no_grad():
            func_results = run_bagel_vote_multi_image_repeated(
                inferencer, pil_images, vote_prompt,
                num_generations=K, max_tokens=512, temperature=temp,
            )
        t_func = time.time() - t0

        results[temp] = {
            'native': native_results,
            'cached': cached_results,
            'func': func_results,
            't_native': t_native,
            't_cached': t_cached,
            't_func': t_func,
        }

        # Print comparison
        print(f"\n  Time: native={t_native:.1f}s  cached={t_cached:.1f}s  func={t_func:.1f}s")
        print(f"  Speedup (cached vs native): {t_native/max(t_cached,0.01):.2f}x")

        for k in range(K):
            n_text = native_results[k][:120].replace('\n', '\\n')
            c_text = cached_results[k][:120].replace('\n', '\\n')
            f_text = func_results[k][:120].replace('\n', '\\n')

            n_vote = game_gen.extract_vote(native_results[k])
            c_vote = game_gen.extract_vote(cached_results[k])
            f_vote = game_gen.extract_vote(func_results[k])

            n_v = n_vote.get("voted_spy", "?") if n_vote else "FAIL"
            c_v = c_vote.get("voted_spy", "?") if c_vote else "FAIL"
            f_v = f_vote.get("voted_spy", "?") if f_vote else "FAIL"

            print(f"\n  Vote {k+1}:")
            print(f"    [A] native  → P{n_v}: {n_text}")
            print(f"    [B] cached  → P{c_v}: {c_text}")
            print(f"    [C] func    → P{f_v}: {f_text}")

    # Save full results
    with open(os.path.join(OUT_DIR, "comparison.txt"), "w") as f:
        f.write(f"Test player: P{test_pid} [{role}], Spy: P{spy_pid}\n")
        f.write(f"Temperatures: {TEMPERATURES}, K={K}\n\n")

        for temp in TEMPERATURES:
            r = results[temp]
            f.write(f"\n{'='*70}\n")
            f.write(f"Temperature = {temp}\n")
            f.write(f"Time: native={r['t_native']:.1f}s  cached={r['t_cached']:.1f}s  func={r['t_func']:.1f}s\n")
            f.write(f"{'='*70}\n")

            for k in range(K):
                f.write(f"\n--- Vote {k+1} ---\n")
                n_vote = game_gen.extract_vote(r['native'][k])
                c_vote = game_gen.extract_vote(r['cached'][k])
                f_vote = game_gen.extract_vote(r['func'][k])

                f.write(f"[A] Native  (voted P{n_vote.get('voted_spy','?') if n_vote else 'FAIL'}):\n")
                f.write(f"{r['native'][k]}\n\n")
                f.write(f"[B] Cached  (voted P{c_vote.get('voted_spy','?') if c_vote else 'FAIL'}):\n")
                f.write(f"{r['cached'][k]}\n\n")
                f.write(f"[C] Func    (voted P{f_vote.get('voted_spy','?') if f_vote else 'FAIL'}):\n")
                f.write(f"{r['func'][k]}\n\n")

            # Summary
            native_votes = [game_gen.extract_vote(t) for t in r['native']]
            cached_votes = [game_gen.extract_vote(t) for t in r['cached']]
            func_votes = [game_gen.extract_vote(t) for t in r['func']]

            def vote_dist(votes):
                targets = []
                for v in votes:
                    if v and isinstance(v.get("voted_spy"), int):
                        targets.append(v["voted_spy"])
                    else:
                        targets.append("FAIL")
                return dict(Counter(targets))

            f.write(f"\nVote distribution:\n")
            f.write(f"  [A] Native: {vote_dist(native_votes)}\n")
            f.write(f"  [B] Cached: {vote_dist(cached_votes)}\n")
            f.write(f"  [C] Func:   {vote_dist(func_votes)}\n")

    print(f"\n\nFull results saved to: {OUT_DIR}/comparison.txt")
    print(f"Images: {OUT_DIR}/player_*.png")
    print(f"GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")


if __name__ == "__main__":
    main()
