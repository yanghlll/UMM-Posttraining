"""
Test: Compare grid-based voting vs multi-prompt (individual image) voting.

Generates images for 2 games, then runs both voting methods on each game.
Saves everything: game info, images, prompts, full vote responses, rewards.

Run:
  CUDA_VISIBLE_DEVICES=2 conda run -n spy_bagel python scripts/test_multi_prompt_vote.py 2>&1 | tee test_samples/multi_prompt_vote/run.log
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
    run_bagel_votes_cached,
    run_bagel_votes_multi_prompt,
    run_bagel_vote_multi_image,
    tensor_images_to_pil,
    build_voting_grid,
    build_vlm_image_transform,
)

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.hooks import remove_hook_from_module
from huggingface_hub import snapshot_download


OUT_DIR = "/adialab/usr/shadabk/MedUMM/flow_grpo/test_samples/multi_prompt_vote"
NUM_GAMES = 2
NUM_PLAYERS = 4
MAX_VOTE_TOKENS = 256


def load_model(device="cuda:0"):
    dtype = torch.bfloat16
    model_path = snapshot_download(repo_id="ByteDance-Seed/BAGEL-7B-MoT")

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    bagel_config = BagelConfig(
        visual_gen=True, visual_und=True,
        llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
        vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh',
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
    for m in model.modules():
        remove_hook_from_module(m)
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
    import ml_collections
    mc = ml_collections.ConfigDict()
    mc.sample = ml_collections.ConfigDict()
    mc.sample.sde_window_size = 0
    mc.sample.sde_window_range = (0, 0)
    mc.train = ml_collections.ConfigDict()
    mc.train.clip_range_lt = 0.2
    mc.train.clip_range_gt = 0.2
    mc.train.beta = 0

    with torch.no_grad():
        out = inferencer.interleave_inference(
            input_lists=[prompt], understanding_output=False,
            do_sample=True, text_temperature=1.0,
            image_shapes=(512, 512), cfg_text_scale=4.0,
            cfg_img_scale=1.0, cfg_interval=[0, 1.0],
            num_timesteps=50, timestep_shift=3.0,
            cfg_renorm_min=0.0, cfg_renorm_type="global",
            grpo_config=mc,
        )
    r = out[0] if isinstance(out, list) else out
    if isinstance(r, dict):
        r = r['image']
    if isinstance(r, torch.Tensor):
        r = tensor_images_to_pil(r.unsqueeze(0) if r.dim() == 3 else r)[0]
    return r


def run_one_game(inferencer, game_gen, game_idx, out_base):
    game_dir = os.path.join(out_base, f"game_{game_idx}")
    os.makedirs(game_dir, exist_ok=True)

    game_data = game_gen.generate_game(epoch=200, sample_idx=game_idx)
    spy_pid = game_data["spy_player"]
    N = NUM_PLAYERS

    # ── Save game info ──
    with open(os.path.join(game_dir, "game_info.txt"), "w") as f:
        f.write(f"=== Game {game_idx} ===\n")
        f.write(f"Spy player: Player {spy_pid}\n")
        f.write(f"Number of players: {N}\n\n")

        f.write("--- Scene Descriptions ---\n")
        for pid in range(1, N + 1):
            role = "SPY" if pid == spy_pid else "CIV"
            desc = game_data["player_descriptions"][pid - 1]
            f.write(f"\nPlayer {pid} [{role}]:\n{desc}\n")

        f.write("\n--- Voting Prompts ---\n")
        for pid in range(1, N + 1):
            role = "SPY" if pid == spy_pid else "CIV"
            vp = game_gen.format_voting_prompt(game_data, player_id=pid)
            f.write(f"\n{'='*40}\nPlayer {pid} [{role}] vote prompt:\n{'='*40}\n{vp}\n")

    print(f"\n  Game {game_idx}: spy=P{spy_pid}")

    # ── Generate images ──
    print(f"  Generating {N} images...")
    pil_images = []
    for pid in range(1, N + 1):
        prompt = game_gen.format_generation_prompt_simple(game_data, pid)
        t0 = time.time()
        img = generate_image(inferencer, prompt)
        dt = time.time() - t0
        img.save(os.path.join(game_dir, f"player_{pid}.png"))
        role = "SPY" if pid == spy_pid else "CIV"
        print(f"    P{pid}[{role}]: {dt:.1f}s")
        pil_images.append(img)

    # Save voting grid for visual reference
    grid = build_voting_grid(pil_images)
    grid.save(os.path.join(game_dir, "voting_grid.png"))

    # Build vote prompts for all players
    vote_prompts = [
        game_gen.format_voting_prompt(game_data, player_id=pid)
        for pid in range(1, N + 1)
    ]

    # ════════════════════════════════════════════════════════════════
    # Method A: OLD grid-based voting (run_bagel_votes_cached)
    #   — encodes single grid image, each prompt sees the same grid
    # ════════════════════════════════════════════════════════════════
    print(f"  [A] Grid-based voting (old)...")
    vote_counts_A = {i: 0 for i in range(1, N + 1)}
    with open(os.path.join(game_dir, "votes_A_grid.txt"), "w") as f:
        f.write("=== Method A: Grid-based voting (OLD — single grid image) ===\n")
        f.write(f"max_tokens={MAX_VOTE_TOKENS}\n\n")

        t0 = time.time()
        vote_texts_A = run_bagel_votes_cached(
            inferencer, grid, vote_prompts,
            max_tokens=MAX_VOTE_TOKENS, temperature=0.7,
        )
        dt_A = time.time() - t0

        for pid_idx, pid in enumerate(range(1, N + 1)):
            role = "SPY" if pid == spy_pid else "CIV"
            vt = vote_texts_A[pid_idx]
            extracted = game_gen.extract_vote(vt)
            voted = extracted.get("voted_spy", "?") if extracted else "FAIL"
            if extracted and isinstance(extracted.get("voted_spy"), int):
                v = extracted["voted_spy"]
                if 1 <= v <= N:
                    vote_counts_A[v] += 1

            f.write(f"--- Player {pid} [{role}] → voted P{voted} ---\n")
            f.write(f"Full response:\n{vt}\n\n")
            print(f"    P{pid}[{role}] → P{voted}")

        f.write(f"\nTotal time: {dt_A:.1f}s\n")
        f.write(f"Vote counts: {vote_counts_A}\n")
        f.write(f"Spy (P{spy_pid}) received: {vote_counts_A[spy_pid]} votes\n")

    print(f"    Time: {dt_A:.1f}s, votes on spy: {vote_counts_A[spy_pid]}/{N}")

    # ════════════════════════════════════════════════════════════════
    # Method B: NEW multi-prompt voting (individual images)
    #   — encodes N images separately into KV cache, each with own ViT tokens
    # ════════════════════════════════════════════════════════════════
    print(f"  [B] Multi-prompt voting (new, individual images)...")
    vote_counts_B = {i: 0 for i in range(1, N + 1)}
    with open(os.path.join(game_dir, "votes_B_multi_prompt.txt"), "w") as f:
        f.write("=== Method B: Multi-prompt voting (NEW — N individual images) ===\n")
        f.write(f"max_tokens={MAX_VOTE_TOKENS}\n\n")

        t0 = time.time()
        vote_texts_B = run_bagel_votes_multi_prompt(
            inferencer, pil_images, vote_prompts,
            max_tokens=MAX_VOTE_TOKENS, temperature=0.7,
        )
        dt_B = time.time() - t0

        for pid_idx, pid in enumerate(range(1, N + 1)):
            role = "SPY" if pid == spy_pid else "CIV"
            vt = vote_texts_B[pid_idx]
            extracted = game_gen.extract_vote(vt)
            voted = extracted.get("voted_spy", "?") if extracted else "FAIL"
            if extracted and isinstance(extracted.get("voted_spy"), int):
                v = extracted["voted_spy"]
                if 1 <= v <= N:
                    vote_counts_B[v] += 1

            f.write(f"--- Player {pid} [{role}] → voted P{voted} ---\n")
            f.write(f"Full response:\n{vt}\n\n")
            print(f"    P{pid}[{role}] → P{voted}")

        f.write(f"\nTotal time: {dt_B:.1f}s\n")
        f.write(f"Vote counts: {vote_counts_B}\n")
        f.write(f"Spy (P{spy_pid}) received: {vote_counts_B[spy_pid]} votes\n")

    print(f"    Time: {dt_B:.1f}s, votes on spy: {vote_counts_B[spy_pid]}/{N}")

    # ════════════════════════════════════════════════════════════════
    # Method C: interleave_inference (Bagel native understanding path)
    #   — text and images truly interleaved: "Player 1:" <img1> "Player 2:" <img2> ...
    # ════════════════════════════════════════════════════════════════
    print(f"  [C] interleave_inference (native Bagel understanding)...")
    vote_counts_C = {i: 0 for i in range(1, N + 1)}
    with open(os.path.join(game_dir, "votes_C_interleave.txt"), "w") as f:
        f.write("=== Method C: interleave_inference (Bagel native — true interleave) ===\n")
        f.write(f"max_tokens={MAX_VOTE_TOKENS}\n\n")

        t_C_total = 0.0
        for pid in range(1, N + 1):
            role = "SPY" if pid == spy_pid else "CIV"
            vote_prompt = vote_prompts[pid - 1]

            # Build interleaved input: text, img, text, img, ..., prompt
            input_lists = []
            for i, img in enumerate(pil_images):
                input_lists.append(f"Player {i+1}'s generated image:")
                input_lists.append(img)
            input_lists.append(vote_prompt)

            t0 = time.time()
            with torch.no_grad():
                out = inferencer.interleave_inference(
                    input_lists=input_lists,
                    understanding_output=True,
                    do_sample=True,
                    text_temperature=0.7,
                    max_think_token_n=MAX_VOTE_TOKENS,
                )
            dt = time.time() - t0
            t_C_total += dt

            vt = out[0] if isinstance(out, list) else str(out)
            extracted = game_gen.extract_vote(vt)
            voted = extracted.get("voted_spy", "?") if extracted else "FAIL"
            if extracted and isinstance(extracted.get("voted_spy"), int):
                v = extracted["voted_spy"]
                if 1 <= v <= N:
                    vote_counts_C[v] += 1

            f.write(f"--- Player {pid} [{role}] → voted P{voted} ({dt:.1f}s) ---\n")
            f.write(f"Full response:\n{vt}\n\n")
            print(f"    P{pid}[{role}] → P{voted} ({dt:.1f}s)")

        f.write(f"\nTotal time: {t_C_total:.1f}s\n")
        f.write(f"Vote counts: {vote_counts_C}\n")
        f.write(f"Spy (P{spy_pid}) received: {vote_counts_C[spy_pid]} votes\n")

    print(f"    Time: {t_C_total:.1f}s, votes on spy: {vote_counts_C[spy_pid]}/{N}")

    # ── Reward analysis ──
    with open(os.path.join(game_dir, "reward_comparison.txt"), "w") as f:
        f.write("=== Reward Comparison ===\n\n")
        f.write(f"Spy: Player {spy_pid}\n\n")

        for label, vc in [("A (grid)", vote_counts_A),
                           ("B (multi-prompt)", vote_counts_B),
                           ("C (interleave)", vote_counts_C)]:
            game_outcome = {
                "player_rewards": {i: 0.0 for i in range(1, N + 1)},
                "spy_caught": vc[spy_pid] > N // 2,
                "vote_counts": vc,
                "spy_player": spy_pid,
            }
            gen_rewards = game_gen.compute_generation_rewards(
                game_outcome, beta=0.1, lambda_param=0.1)

            f.write(f"--- Method {label} ---\n")
            f.write(f"  Vote counts: {vc}\n")
            f.write(f"  Spy caught: {vc[spy_pid] > N // 2}\n")
            for pid in range(1, N + 1):
                role = "SPY" if pid == spy_pid else "CIV"
                f.write(f"  P{pid} [{role}]: reward={gen_rewards[pid-1]:+.4f}\n")
            f.write(f"  Sum: {sum(gen_rewards):.6f}\n\n")

    # ── Summary ──
    with open(os.path.join(game_dir, "summary.txt"), "w") as f:
        f.write(f"Game {game_idx} | Spy=P{spy_pid}\n\n")
        f.write(f"{'Method':<25} {'Time':>8} {'Spy votes':>10} {'Caught':>8}\n")
        f.write(f"{'-'*55}\n")
        for label, vc, dt in [("A: grid (old)", vote_counts_A, dt_A),
                                ("B: multi-prompt (new)", vote_counts_B, dt_B),
                                ("C: interleave (native)", vote_counts_C, t_C_total)]:
            caught = "YES" if vc[spy_pid] > N // 2 else "NO"
            f.write(f"{label:<25} {dt:>7.1f}s {vc[spy_pid]:>5}/{N:<4} {caught:>8}\n")

        f.write(f"\nDetailed vote targets:\n")
        for label, vc in [("A", vote_counts_A), ("B", vote_counts_B), ("C", vote_counts_C)]:
            f.write(f"  {label}: {vc}\n")

    print(f"\n  Summary saved → {game_dir}/summary.txt")
    return game_dir


def main():
    device = "cuda:0"

    print("=" * 60)
    print("Grid vs Multi-Prompt vs Interleave Voting Comparison")
    print(f"Output: {OUT_DIR}")
    print(f"Games: {NUM_GAMES}, Players: {NUM_PLAYERS}")
    print(f"Max vote tokens: {MAX_VOTE_TOKENS}")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n[1] Loading model...")
    t0 = time.time()
    inferencer = load_model(device)
    print(f"  Loaded in {time.time()-t0:.1f}s, GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    game_gen = SpyGameDataGenerator(num_players=NUM_PLAYERS)

    for g in range(NUM_GAMES):
        print(f"\n{'='*60}")
        print(f"[{g+2}] Running game {g}...")
        print(f"{'='*60}")
        t0 = time.time()
        game_dir = run_one_game(inferencer, game_gen, g, OUT_DIR)
        print(f"  Game {g} done in {time.time()-t0:.1f}s")

    print(f"\n{'='*60}")
    print(f"All done! Results in: {OUT_DIR}")
    print(f"GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
