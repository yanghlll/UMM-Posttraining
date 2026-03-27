"""
Simulate 2 complete spy games end-to-end, saving everything:
  - All prompts (generation + voting)
  - All generated images (per player)
  - All vote responses (full text)
  - Game outcomes and rewards
"""
import torch, time, os, json
from PIL import Image
from flow_grpo.bagel.data.data_utils import add_special_tokens
from flow_grpo.bagel.data.transforms import ImageTransform
from flow_grpo.bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from flow_grpo.bagel.modeling.qwen2 import Qwen2Tokenizer
from flow_grpo.bagel.modeling.autoencoder import load_ae
from flow_grpo.bagel.inferencer import InterleaveInferencer
from flow_grpo.spy_game_data import TextFileGameDataGenerator
from flow_grpo.spy_game_reward import tensor_images_to_pil, run_bagel_vote_multi_image
from huggingface_hub import snapshot_download
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from accelerate.hooks import remove_hook_from_module
import ml_collections

OUTPUT_DIR = "simulate_game_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dtype = torch.bfloat16
device = "cuda:0"

# ── Load model ──
print("Loading model...")
m = snapshot_download(repo_id='ByteDance-Seed/BAGEL-7B-MoT')
lc = Qwen2Config.from_json_file(os.path.join(m, 'llm_config.json'))
lc.qk_norm = True; lc.tie_word_embeddings = False; lc.layer_module = 'Qwen2MoTDecoderLayer'
vc = SiglipVisionConfig.from_json_file(os.path.join(m, 'vit_config.json'))
vc.rope = False; vc.num_hidden_layers -= 1
vae, vaec = load_ae(local_path=os.path.join(m, 'ae.safetensors'))
bc = BagelConfig(visual_gen=True, visual_und=True, llm_config=lc, vit_config=vc,
                  vae_config=vaec, vit_max_num_patch_per_side=70,
                  connector_act='gelu_pytorch_tanh', latent_patch_size=2, max_latent_size=64)
with init_empty_weights():
    model = Bagel(Qwen2ForCausalLM(lc), SiglipVisionModel(vc), bc)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vc, meta=True)
tok, ntid, _ = add_special_tokens(Qwen2Tokenizer.from_pretrained(m))
model = load_checkpoint_and_dispatch(model, checkpoint=os.path.join(m, 'ema.safetensors'),
    device_map={'': device}, offload_buffers=False, dtype=dtype, force_hooks=True,
    offload_folder='/adialab/usr/shadabk/MedUMM/.offload')
for mod in model.modules():
    remove_hook_from_module(mod)
model.eval(); vae.to(device, dtype=dtype)

inf = InterleaveInferencer(model=model, vae_model=vae, tokenizer=tok,
    vae_transform=ImageTransform(512, 256, 8), vit_transform=ImageTransform(490, 112, 7),
    new_token_ids=ntid)

gc = ml_collections.ConfigDict()
gc.sample = ml_collections.ConfigDict()
gc.sample.sde_window_size = 3; gc.sample.sde_window_range = (0, 7)
gc.train = ml_collections.ConfigDict()
gc.train.clip_range = 1e-4; gc.train.clip_range_lt = 1e-5; gc.train.clip_range_gt = 1e-5
gc.train.beta = 0; gc.train.adv_clip_max = 5; gc.train.timestep_fraction = 1
ih = dict(cfg_img_scale=1.0, cfg_interval=[0, 1.0], timestep_shift=3.0,
          cfg_renorm_min=0.0, cfg_renorm_type='global', image_shapes=(512, 512))

gen = TextFileGameDataGenerator('dataset/ocr', 'train', prompt_type='ocr', num_players=4)
print("Model loaded.\n")

# ── Simulate 2 games ──
for gi in range(2):
    game = gen.generate_game(0, gi)
    spy = game['spy_player']
    N = game['num_players']
    game_dir = os.path.join(OUTPUT_DIR, f"game_{gi+1}")
    os.makedirs(game_dir, exist_ok=True)

    log = []
    log.append(f"{'='*80}")
    log.append(f"GAME {gi+1}")
    log.append(f"{'='*80}")
    log.append(f"Spy: Player {spy}")
    log.append(f"Original description: {game['original_description']}")
    log.append(f"Modified description: {game['modified_description']}")
    log.append(f"Modification: {game['diff_metadata'].get('change', 'N/A')}")
    log.append("")

    print(f"=== GAME {gi+1} (spy=Player {spy}) ===")
    print(f"  Original: {game['original_description'][:80]}...")
    print(f"  Modified: {game['modified_description'][:80]}...")
    print(f"  Change: {game['diff_metadata'].get('change', 'N/A')}")

    # ── Phase 1: Image generation ──
    log.append("=" * 40)
    log.append("PHASE 1: IMAGE GENERATION")
    log.append("=" * 40)

    imgs_tensor = []
    pil_imgs = []
    for pid in range(1, N + 1):
        role = "SPY" if pid == spy else "CIV"
        prompt = gen.format_generation_prompt_simple(game, pid)

        log.append(f"\n--- Player {pid} ({role}) ---")
        log.append(f"Generation prompt:\n{prompt}")

        print(f"  Generating P{pid}({role})...", end=" ", flush=True)
        t0 = time.time()
        with torch.no_grad():
            out = inf(text=prompt, noise_level=1.3, num_timesteps=15,
                      cfg_text_scale=4.0, grpo_config=gc, **ih)
        dt = time.time() - t0
        print(f"{dt:.1f}s")

        img_t = out['image']
        imgs_tensor.append(img_t)
        img_np = (img_t * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
        pil = Image.fromarray(img_np)
        pil_imgs.append(pil)

        fname = f"player{pid}_{role}.jpg"
        pil.save(os.path.join(game_dir, fname))
        log.append(f"Image saved: {fname} ({dt:.1f}s)")

    # ── Phase 2: Voting ──
    log.append("\n" + "=" * 40)
    log.append("PHASE 2: VOTING")
    log.append("=" * 40)

    votes = []
    for pid in range(1, N + 1):
        role = "SPY" if pid == spy else "CIV"
        vote_prompt = gen.format_voting_prompt(game, player_id=pid)

        log.append(f"\n--- Player {pid} ({role}) voting ---")
        log.append(f"Vote prompt:\n{vote_prompt}")

        # Show the interleaved input structure
        log.append(f"\nInterleaved input to model:")
        for i in range(N):
            log.append(f"  [text] \"Player {i+1}'s generated image:\"")
            log.append(f"  [image] player{i+1}_{'SPY' if i+1==spy else 'CIV'}.jpg")
        log.append(f"  [text] <vote prompt above>")

        print(f"  Voting P{pid}({role})...", end=" ", flush=True)
        t0 = time.time()
        vote_text = run_bagel_vote_multi_image(
            inf, pil_imgs, vote_prompt, max_tokens=512, temperature=0.7)
        dt = time.time() - t0
        ntok = len(tok.encode(vote_text))
        print(f"{dt:.1f}s, {ntok} tokens")

        vi = gen.extract_vote(vote_text)
        voted = vi.get('voted_spy') if vi else None
        correct = (voted == spy) if isinstance(voted, int) else False

        log.append(f"\nModel response ({ntok} tokens, {dt:.1f}s):")
        log.append(vote_text)
        log.append(f"\nExtracted vote: {voted}")
        log.append(f"Correct: {correct}")

        votes.append(vi)

        # Save vote text
        with open(os.path.join(game_dir, f"vote_player{pid}_{role}.txt"), 'w') as f:
            f.write(f"=== Player {pid} ({role}) Vote ===\n\n")
            f.write(f"Prompt:\n{vote_prompt}\n\n")
            f.write(f"Response ({ntok} tokens):\n{vote_text}\n\n")
            f.write(f"Extracted vote: {voted}\n")
            f.write(f"Correct: {correct}\n")

    # ── Phase 3: Rewards ──
    log.append("\n" + "=" * 40)
    log.append("PHASE 3: REWARDS")
    log.append("=" * 40)

    outcome = gen.calculate_game_rewards(game, votes)
    gen_rewards = gen.compute_generation_rewards(outcome)

    log.append(f"\nVote counts: {outcome['vote_counts']}")
    log.append(f"Spy caught: {outcome['spy_caught']}")
    log.append(f"Player rewards (voting): {outcome['player_rewards']}")
    log.append(f"Generation rewards: {gen_rewards}")

    print(f"\n  Vote counts: {outcome['vote_counts']}")
    print(f"  Spy caught: {outcome['spy_caught']}")
    print(f"  Gen rewards: {gen_rewards}")

    # Summary table
    log.append(f"\n{'='*60}")
    log.append(f"SUMMARY")
    log.append(f"{'='*60}")
    log.append(f"{'Player':<10} {'Role':<6} {'Voted':<8} {'Correct':<8} {'GenReward':<10}")
    log.append(f"{'-'*42}")
    for pid in range(1, N + 1):
        role = "SPY" if pid == spy else "CIV"
        vi = votes[pid - 1]
        voted = vi.get('voted_spy') if vi else 'INVALID'
        correct = "✓" if (isinstance(voted, int) and voted == spy) else "✗"
        reward = gen_rewards[pid - 1]
        log.append(f"Player {pid:<3} {role:<6} {str(voted):<8} {correct:<8} {reward:<10.1f}")

    # Save full log
    with open(os.path.join(game_dir, "game_log.txt"), 'w') as f:
        f.write('\n'.join(log))

    print(f"\n  All files saved to {game_dir}/\n")

# Save index
with open(os.path.join(OUTPUT_DIR, "README.txt"), 'w') as f:
    f.write("Spy Game Simulation Output\n")
    f.write("=" * 40 + "\n\n")
    f.write("Each game_N/ folder contains:\n")
    f.write("  - player{1-4}_{SPY|CIV}.jpg : generated images\n")
    f.write("  - vote_player{1-4}_{SPY|CIV}.txt : full vote prompt + response\n")
    f.write("  - game_log.txt : complete game log with all details\n")

print(f"All output saved to {OUTPUT_DIR}/")
