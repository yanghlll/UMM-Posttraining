"""
Test script: Verify repeated voting (K=8) produces diverse results,
and that single vs repeated voting are consistent.

Run on GPU 3-4:
  CUDA_VISIBLE_DEVICES=3 python flow_grpo/scripts/test_repeated_voting.py
"""

import os
import sys
import torch
import time
from collections import Counter
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from flow_grpo.bagel.data.data_utils import add_special_tokens
from flow_grpo.bagel.data.transforms import ImageTransform
from flow_grpo.bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from flow_grpo.bagel.modeling.qwen2 import Qwen2Tokenizer
from flow_grpo.bagel.modeling.autoencoder import load_ae
from flow_grpo.bagel.inferencer import InterleaveInferencer
from flow_grpo.spy_game_data import SpyGameDataGenerator
from flow_grpo.spy_game_reward import (
    run_bagel_vote_multi_image,
    run_bagel_vote_multi_image_repeated,
    tensor_images_to_pil,
)

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.hooks import remove_hook_from_module
from huggingface_hub import snapshot_download


def load_model(model_name="ByteDance-Seed/BAGEL-7B-MoT", device="cuda:0"):
    """Load Bagel model (same as train_bagel_spy.py)."""
    dtype = torch.bfloat16

    # Resolve model path (local or HF hub)
    if os.path.exists(model_name):
        model_path = model_name
    else:
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
    return inferencer, model, tokenizer, new_token_ids, vit_transform


def generate_test_images_dummy(num_players):
    """Create simple colored test images for voting test (no GPU gen needed)."""
    import numpy as np
    colors = [(255, 50, 50), (50, 50, 255), (50, 200, 50), (255, 200, 50)]
    images = []
    for i in range(num_players):
        # Create a 512x512 image with colored geometric shapes
        img = np.ones((512, 512, 3), dtype=np.uint8) * 200  # gray bg
        c = colors[i % len(colors)]
        # Draw a colored rectangle (simulate different scenes)
        img[100:300, 100:300] = c
        if i == 0:  # spy — slightly different
            img[150:350, 150:350] = (c[1], c[2], c[0])  # shifted color
        images.append(Image.fromarray(img))
    return images


def generate_test_images_real(inferencer, game_data, num_players):
    """Generate real images using Bagel (slower but more realistic test)."""
    images = []
    for pid in range(1, num_players + 1):
        desc = game_data["player_descriptions"][pid - 1]
        prompt = f"A 3D rendered scene with geometric objects on a flat surface: {desc}"
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = inferencer.interleave_inference(
                input_lists=[prompt],
                understanding_output=False,
                do_sample=True,
                text_temperature=1.0,
                image_shapes=(512, 512),
                cfg_text_scale=4.0,
                num_timesteps=15,
                timestep_shift=3.0,
            )
        if isinstance(output, list) and len(output) > 0:
            img = output[0]
        else:
            img = output
        if isinstance(img, torch.Tensor):
            img = tensor_images_to_pil(img.unsqueeze(0) if img.dim() == 3 else img)[0]
        images.append(img)
    return images


def main():
    model_name = "ByteDance-Seed/BAGEL-7B-MoT"
    device = "cuda:0"
    num_players = 4
    K = 8  # repeated votes

    print("=" * 60)
    print("Test: Repeated Voting (K=8) Diversity & Consistency")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model...")
    t0 = time.time()
    inferencer, model, tokenizer, new_token_ids, vit_transform = load_model(model_name, device)
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Generate game data
    print("\n[2/4] Generating game data and images...")
    game_gen = SpyGameDataGenerator(num_players=num_players)
    game_data = game_gen.generate_game(epoch=0, sample_idx=42)
    print(f"  Spy player: {game_data['spy_player']}")
    for pid in range(1, num_players + 1):
        desc = game_data['player_descriptions'][pid-1]
        print(f"  P{pid}: {desc[:80]}...")

    # Generate images (use dummy for fast voting test)
    t0 = time.time()
    pil_images = generate_test_images_dummy(num_players)
    print(f"  {num_players} dummy images created in {time.time()-t0:.1f}s")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Test voting
    print("\n[3/4] Testing single vote vs repeated vote...")

    # Pick one player for detailed comparison
    test_pid = 1
    vote_prompt = game_gen.format_voting_prompt(game_data, player_id=test_pid)

    # Single votes (K times, each call independent)
    print(f"\n  --- Single votes (K={K} independent calls) ---")
    single_votes = []
    t0 = time.time()
    with torch.no_grad():
        for k in range(K):
            vote_text = run_bagel_vote_multi_image(
                inferencer, pil_images, vote_prompt,
                max_tokens=512, temperature=0.7,
            )
            extracted = game_gen.extract_vote(vote_text)
            single_votes.append(extracted)
            voted = extracted.get("voted_spy", "?") if extracted else "FAIL"
            print(f"    Vote {k+1}: Player {voted}  (text: {vote_text[:100]}...)")
    t_single = time.time() - t0
    print(f"  Time: {t_single:.1f}s ({t_single/K:.1f}s per vote)")

    # Repeated votes (cached ViT context)
    print(f"\n  --- Repeated votes (K={K}, cached ViT context) ---")
    t0 = time.time()
    with torch.no_grad():
        repeated_texts = run_bagel_vote_multi_image_repeated(
            inferencer, pil_images, vote_prompt,
            num_generations=K, max_tokens=512, temperature=0.7,
        )
    t_repeated = time.time() - t0

    repeated_votes = []
    for k, vt in enumerate(repeated_texts):
        extracted = game_gen.extract_vote(vt)
        repeated_votes.append(extracted)
        voted = extracted.get("voted_spy", "?") if extracted else "FAIL"
        print(f"    Vote {k+1}: Player {voted}  (text: {vt[:100]}...)")
    print(f"  Time: {t_repeated:.1f}s ({t_repeated/K:.1f}s per vote)")
    print(f"  Speedup: {t_single/max(t_repeated, 0.01):.2f}x")

    # Analysis
    print("\n[4/4] Analysis")
    print("=" * 60)

    def vote_distribution(votes):
        targets = []
        for v in votes:
            if v and isinstance(v.get("voted_spy"), int):
                targets.append(v["voted_spy"])
            else:
                targets.append("INVALID")
        return Counter(targets)

    single_dist = vote_distribution(single_votes)
    repeated_dist = vote_distribution(repeated_votes)

    print(f"  Single vote distribution:   {dict(single_dist)}")
    print(f"  Repeated vote distribution: {dict(repeated_dist)}")

    # Check diversity
    single_unique = len(set(str(v) for v in single_votes))
    repeated_unique = len(set(str(v) for v in repeated_votes))
    print(f"  Single unique responses:   {single_unique}/{K}")
    print(f"  Repeated unique responses: {repeated_unique}/{K}")

    # Check text diversity (are generated texts actually different?)
    single_texts_set = set()
    repeated_texts_set = set()
    for k in range(K):
        # We don't have single texts saved, but repeated we do
        repeated_texts_set.add(repeated_texts[k][:200])

    print(f"  Repeated unique text prefixes: {len(repeated_texts_set)}/{K}")

    # Verify vote targets are similar between methods
    print(f"\n  Single most common:   {single_dist.most_common(1)}")
    print(f"  Repeated most common: {repeated_dist.most_common(1)}")

    # Full game test: all players vote K times
    print(f"\n  --- Full game: all {num_players} players × K={K} votes ---")
    vote_counts = {i: 0 for i in range(1, num_players + 1)}
    t0 = time.time()
    with torch.no_grad():
        for pid in range(1, num_players + 1):
            vp = game_gen.format_voting_prompt(game_data, player_id=pid)
            vote_texts = run_bagel_vote_multi_image_repeated(
                inferencer, pil_images, vp,
                num_generations=K, max_tokens=512, temperature=0.7,
            )
            pid_targets = []
            for vt in vote_texts:
                ext = game_gen.extract_vote(vt)
                if ext and isinstance(ext.get("voted_spy"), int):
                    v = ext["voted_spy"]
                    if 1 <= v <= num_players:
                        vote_counts[v] += 1
                        pid_targets.append(v)
            print(f"    P{pid} votes: {pid_targets}  (dist: {dict(Counter(pid_targets))})")
    t_full = time.time() - t0

    spy = game_data["spy_player"]
    total_votes = num_players * K
    print(f"\n  Aggregated vote_counts: {vote_counts}")
    print(f"  Spy (P{spy}) received: {vote_counts[spy]}/{total_votes} votes")
    print(f"  Spy caught: {vote_counts[spy] > total_votes // 2}")
    print(f"  Full game voting time: {t_full:.1f}s")

    # Zero-sum reward
    game_outcome = {
        "player_rewards": {i: 0.0 for i in range(1, num_players + 1)},
        "spy_caught": vote_counts[spy] > total_votes // 2,
        "vote_counts": vote_counts,
        "spy_player": spy,
    }
    gen_rewards = game_gen.compute_generation_rewards(game_outcome, beta=0.1, lambda_param=0.1)
    print(f"  Zero-sum rewards: {[f'{r:.4f}' for r in gen_rewards]}")
    print(f"  Sum (should be ~0): {sum(gen_rewards):.6f}")

    print(f"\n  GPU memory peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    print("\nDone!")


if __name__ == "__main__":
    main()
