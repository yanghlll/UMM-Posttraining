"""
Full game sample test: generate images, run voting, save everything for inspection.

Outputs to /adialab/usr/shadabk/MedUMM/flow_grpo/test_samples/:
  game_N/
    game_info.txt          — game setup, prompts, spy assignment
    player_1.png ... player_4.png — generated images
    votes_single.txt       — single vote per player (original interleave_inference)
    votes_repeated_K8.txt  — K=8 repeated votes per player (cached ViT)
    bagel_native.txt       — native Bagel understanding output (no game framing)
    reward_analysis.txt    — vote aggregation + zero-sum rewards

Run:
  CUDA_VISIBLE_DEVICES=3 conda run -n spy_bagel python scripts/test_full_game_samples.py
"""

import os
import sys
import json
import torch
import time
from collections import Counter
from PIL import Image

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
    build_voting_grid,
    build_vlm_image_transform,
)

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.hooks import remove_hook_from_module
from huggingface_hub import snapshot_download


OUT_DIR = "/adialab/usr/shadabk/MedUMM/flow_grpo/test_samples"
NUM_GAMES = 2
NUM_PLAYERS = 4
K = 8  # repeated votes


def load_model(device="cuda:0"):
    model_name = "ByteDance-Seed/BAGEL-7B-MoT"
    dtype = torch.bfloat16

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
    return inferencer


def generate_image(inferencer, prompt):
    """Generate one image using Bagel's native interleave_inference.

    The gen_image path requires grpo_config for SDE window params.
    We pass a mock config so inference works without training setup.
    """
    import ml_collections

    mock_config = ml_collections.ConfigDict()
    mock_config.sample = ml_collections.ConfigDict()
    mock_config.sample.sde_window_size = 0  # 0 = no SDE (standard ODE sampling)
    mock_config.sample.sde_window_range = (0, 0)
    mock_config.train = ml_collections.ConfigDict()
    mock_config.train.clip_range_lt = 0.2
    mock_config.train.clip_range_gt = 0.2
    mock_config.train.beta = 0

    with torch.no_grad():
        output = inferencer.interleave_inference(
            input_lists=[prompt],
            understanding_output=False,
            do_sample=True,
            text_temperature=1.0,
            image_shapes=(512, 512),
            cfg_text_scale=4.0,
            cfg_img_scale=1.0,
            cfg_interval=[0, 1.0],
            num_timesteps=50,
            timestep_shift=3.0,
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            grpo_config=mock_config,
        )
    if isinstance(output, list) and len(output) > 0:
        result = output[0]
    else:
        result = output
    # interleave_inference returns dict with 'image' key for gen mode
    if isinstance(result, dict) and 'image' in result:
        img = result['image']
    else:
        img = result
    if isinstance(img, torch.Tensor):
        img = tensor_images_to_pil(img.unsqueeze(0) if img.dim() == 3 else img)[0]
    return img


def native_bagel_understand(inferencer, images, prompt):
    """Use Bagel's native model.chat() for understanding (matching eval code)."""
    from flow_grpo.bagel.data.data_utils import pil_img2rgb

    vlm_transform = build_vlm_image_transform()
    images_rgb = [pil_img2rgb(img) for img in images]

    with torch.no_grad():
        output = inferencer.model.chat(
            tokenizer=inferencer.tokenizer,
            new_token_ids=inferencer.new_token_ids,
            image_transform=vlm_transform,
            images=images_rgb,
            prompt=prompt,
            max_length=512,
            do_sample=True,
            temperature=0.7,
        )
    return output


def run_one_game(inferencer, game_gen, game_idx, out_base):
    game_dir = os.path.join(out_base, f"game_{game_idx}")
    os.makedirs(game_dir, exist_ok=True)

    game_data = game_gen.generate_game(epoch=100, sample_idx=game_idx)
    spy_pid = game_data["spy_player"]

    # ── Save game info ──
    with open(os.path.join(game_dir, "game_info.txt"), "w") as f:
        f.write(f"=== Game {game_idx} ===\n")
        f.write(f"Spy player: Player {spy_pid}\n")
        f.write(f"Number of players: {NUM_PLAYERS}\n\n")

        f.write("--- Scene Descriptions ---\n")
        for pid in range(1, NUM_PLAYERS + 1):
            role = "SPY" if pid == spy_pid else "CIV"
            desc = game_data["player_descriptions"][pid - 1]
            f.write(f"\nPlayer {pid} [{role}]:\n{desc}\n")

        f.write("\n--- Generation Prompts ---\n")
        for pid in range(1, NUM_PLAYERS + 1):
            role = "SPY" if pid == spy_pid else "CIV"
            prompt = game_gen.format_generation_prompt_simple(game_data, pid)
            f.write(f"\nPlayer {pid} [{role}] prompt:\n{prompt}\n")

        f.write("\n--- Voting Prompts ---\n")
        for pid in range(1, NUM_PLAYERS + 1):
            role = "SPY" if pid == spy_pid else "CIV"
            vote_prompt = game_gen.format_voting_prompt(game_data, player_id=pid)
            f.write(f"\nPlayer {pid} [{role}] vote prompt:\n{vote_prompt}\n")

    print(f"\n  Game {game_idx}: spy=P{spy_pid}")

    # ── Generate images ──
    print(f"  Generating {NUM_PLAYERS} images (50 steps each)...")
    pil_images = []
    for pid in range(1, NUM_PLAYERS + 1):
        prompt = game_gen.format_generation_prompt_simple(game_data, pid)
        t0 = time.time()
        img = generate_image(inferencer, prompt)
        dt = time.time() - t0
        img.save(os.path.join(game_dir, f"player_{pid}.png"))
        role = "SPY" if pid == spy_pid else "CIV"
        print(f"    P{pid}[{role}]: {dt:.1f}s")
        pil_images.append(img)

    # Save voting grid
    grid = build_voting_grid(pil_images)
    grid.save(os.path.join(game_dir, "voting_grid.png"))

    # ── Single votes (original interleave_inference path) ──
    print(f"  Running single votes...")
    with open(os.path.join(game_dir, "votes_single.txt"), "w") as f:
        f.write("=== Single Vote per Player (via interleave_inference) ===\n\n")
        for pid in range(1, NUM_PLAYERS + 1):
            vote_prompt = game_gen.format_voting_prompt(game_data, player_id=pid)
            role = "SPY" if pid == spy_pid else "CIV"

            t0 = time.time()
            with torch.no_grad():
                vote_text = run_bagel_vote_multi_image(
                    inferencer, pil_images, vote_prompt,
                    max_tokens=512, temperature=0.7,
                )
            dt = time.time() - t0

            extracted = game_gen.extract_vote(vote_text)
            voted = extracted.get("voted_spy", "?") if extracted else "FAIL"

            f.write(f"--- Player {pid} [{role}] ---\n")
            f.write(f"Voted for: Player {voted}\n")
            f.write(f"Time: {dt:.1f}s\n")
            f.write(f"Full response:\n{vote_text}\n\n")
            print(f"    P{pid}[{role}] voted: P{voted} ({dt:.1f}s)")

    # ── Repeated votes K=8 (cached ViT context) ──
    print(f"  Running repeated votes (K={K})...")
    vote_counts = {i: 0 for i in range(1, NUM_PLAYERS + 1)}
    with open(os.path.join(game_dir, f"votes_repeated_K{K}.txt"), "w") as f:
        f.write(f"=== Repeated Votes K={K} per Player (cached ViT context) ===\n\n")
        for pid in range(1, NUM_PLAYERS + 1):
            vote_prompt = game_gen.format_voting_prompt(game_data, player_id=pid)
            role = "SPY" if pid == spy_pid else "CIV"

            t0 = time.time()
            with torch.no_grad():
                vote_texts = run_bagel_vote_multi_image_repeated(
                    inferencer, pil_images, vote_prompt,
                    num_generations=K, max_tokens=512, temperature=0.7,
                )
            dt = time.time() - t0

            f.write(f"--- Player {pid} [{role}] ({dt:.1f}s for {K} votes) ---\n")
            pid_targets = []
            for k, vt in enumerate(vote_texts):
                extracted = game_gen.extract_vote(vt)
                voted = extracted.get("voted_spy", "?") if extracted else "FAIL"
                if extracted and isinstance(extracted.get("voted_spy"), int):
                    v = extracted["voted_spy"]
                    if 1 <= v <= NUM_PLAYERS:
                        vote_counts[v] += 1
                        pid_targets.append(v)
                f.write(f"\n  Vote {k+1}: Player {voted}\n")
                f.write(f"  Response: {vt}\n")

            dist = dict(Counter(pid_targets))
            f.write(f"\n  Distribution: {dist}\n\n")
            print(f"    P{pid}[{role}] K={K} votes: {dist} ({dt:.1f}s)")

    # ── Native Bagel understanding (no game framing) ──
    print(f"  Running native Bagel understanding...")
    with open(os.path.join(game_dir, "bagel_native.txt"), "w") as f:
        f.write("=== Native Bagel Understanding (no game framing) ===\n\n")

        # Simple comparison prompt
        simple_prompt = (
            f"Compare these {NUM_PLAYERS} images carefully. "
            f"They are supposed to show the same 3D scene with geometric objects. "
            f"Which image looks different from the others? "
            f"Describe the differences you see."
        )
        f.write(f"Prompt: {simple_prompt}\n\n")

        t0 = time.time()
        native_response = native_bagel_understand(inferencer, pil_images, simple_prompt)
        dt = time.time() - t0
        f.write(f"Response ({dt:.1f}s):\n{native_response}\n\n")
        print(f"    Native response: {native_response[:100]}...")

        # Also try with a different prompt (no think mode — model.chat doesn't support it)
        f.write("--- Detailed analysis prompt ---\n")
        detail_prompt = (
            f"These {NUM_PLAYERS} images show 3D rendered scenes. One image is slightly different. "
            f"Which one and why? Please think step by step."
        )
        f.write(f"Prompt: {detail_prompt}\n\n")
        t0 = time.time()
        detail_response = native_bagel_understand(inferencer, pil_images, detail_prompt)
        dt = time.time() - t0
        f.write(f"Response ({dt:.1f}s):\n{detail_response}\n")
        print(f"    Detail response: {detail_response[:100]}...")

    # ── Reward analysis ──
    spy = game_data["spy_player"]
    total_votes = NUM_PLAYERS * K
    spy_caught = vote_counts[spy] > total_votes // 2
    game_outcome = {
        "player_rewards": {i: 0.0 for i in range(1, NUM_PLAYERS + 1)},
        "spy_caught": spy_caught,
        "vote_counts": vote_counts,
        "spy_player": spy,
    }
    gen_rewards = game_gen.compute_generation_rewards(game_outcome, beta=0.1, lambda_param=0.1)

    with open(os.path.join(game_dir, "reward_analysis.txt"), "w") as f:
        f.write("=== Reward Analysis ===\n\n")
        f.write(f"Spy: Player {spy}\n")
        f.write(f"Total votes: {total_votes} ({NUM_PLAYERS} players x {K} votes)\n")
        f.write(f"Vote counts: {vote_counts}\n")
        f.write(f"Spy (P{spy}) received: {vote_counts[spy]} votes\n")
        f.write(f"Spy caught: {spy_caught}\n\n")

        f.write("--- Zero-sum Rewards (Vision-Zero formula) ---\n")
        for pid in range(1, NUM_PLAYERS + 1):
            role = "SPY" if pid == spy else "CIV"
            f.write(f"  P{pid} [{role}]: {gen_rewards[pid-1]:+.4f}\n")
        f.write(f"\n  Sum: {sum(gen_rewards):.6f} (should be 0)\n")

    print(f"  Rewards: {[f'{r:+.4f}' for r in gen_rewards]}, sum={sum(gen_rewards):.6f}")
    print(f"  Spy caught: {spy_caught}, votes on spy: {vote_counts[spy]}/{total_votes}")
    return game_dir


def main():
    device = "cuda:0"

    print("=" * 60)
    print("Full Game Sample Test")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n[1] Loading model...")
    t0 = time.time()
    inferencer = load_model(device)
    print(f"  Loaded in {time.time()-t0:.1f}s, GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    game_gen = SpyGameDataGenerator(num_players=NUM_PLAYERS)

    for g in range(NUM_GAMES):
        print(f"\n[{g+2}] Running game {g}...")
        t0 = time.time()
        game_dir = run_one_game(inferencer, game_gen, g, OUT_DIR)
        print(f"  Game {g} done in {time.time()-t0:.1f}s → {game_dir}")

    print(f"\nAll done! Results in: {OUT_DIR}")
    print(f"GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")


if __name__ == "__main__":
    main()
