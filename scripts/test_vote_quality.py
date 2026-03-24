"""Test voting quality with new Vision-Zero style prompts.
Runs 4 games, 4 players each, reports token lengths and answer quality.
"""
import torch, time, os, re
from PIL import Image
from flow_grpo.bagel.data.data_utils import add_special_tokens, pil_img2rgb
from flow_grpo.bagel.data.transforms import ImageTransform
from flow_grpo.bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from flow_grpo.bagel.modeling.qwen2 import Qwen2Tokenizer
from flow_grpo.bagel.modeling.autoencoder import load_ae
from flow_grpo.bagel.inferencer import InterleaveInferencer
from flow_grpo.spy_game_data import TextFileGameDataGenerator
from flow_grpo.spy_game_reward import (
    build_voting_grid_tensor, grid_tensor_to_pil, run_bagel_votes_batch
)
from huggingface_hub import snapshot_download
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from accelerate.hooks import remove_hook_from_module
import ml_collections

dtype = torch.bfloat16
device = "cuda:0"

# Load model
model_local_dir = snapshot_download(repo_id='ByteDance-Seed/BAGEL-7B-MoT')
llm_config = Qwen2Config.from_json_file(os.path.join(model_local_dir, 'llm_config.json'))
llm_config.qk_norm = True; llm_config.tie_word_embeddings = False
llm_config.layer_module = 'Qwen2MoTDecoderLayer'
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_local_dir, 'vit_config.json'))
vit_config.rope = False; vit_config.num_hidden_layers -= 1
vae_model, vae_config = load_ae(local_path=os.path.join(model_local_dir, 'ae.safetensors'))
bagel_config = BagelConfig(
    visual_gen=True, visual_und=True, llm_config=llm_config,
    vit_config=vit_config, vae_config=vae_config,
    vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh',
    latent_patch_size=2, max_latent_size=64,
)
with init_empty_weights():
    lm = Qwen2ForCausalLM(llm_config); vm = SiglipVisionModel(vit_config)
    model = Bagel(lm, vm, bagel_config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
tokenizer, new_token_ids, _ = add_special_tokens(
    Qwen2Tokenizer.from_pretrained(model_local_dir))
vit_transform = ImageTransform(490, 112, 7)
model = load_checkpoint_and_dispatch(
    model, checkpoint=os.path.join(model_local_dir, 'ema.safetensors'),
    device_map={'': device}, offload_buffers=False, dtype=dtype,
    force_hooks=True, offload_folder='/tmp/offload')
for m in model.modules():
    remove_hook_from_module(m)
model.eval(); vae_model.to(device, dtype=dtype)

grpo_config = ml_collections.ConfigDict()
grpo_config.sample = ml_collections.ConfigDict()
grpo_config.sample.sde_window_size = 3; grpo_config.sample.sde_window_range = (0, 7)
grpo_config.train = ml_collections.ConfigDict()
grpo_config.train.clip_range = 1e-4; grpo_config.train.clip_range_lt = 1e-5
grpo_config.train.clip_range_gt = 1e-5; grpo_config.train.beta = 0.0
grpo_config.train.adv_clip_max = 5.0; grpo_config.train.timestep_fraction = 1.0
ihyper = dict(cfg_img_scale=1.0, cfg_interval=[0, 1.0], timestep_shift=3.0,
              cfg_renorm_min=0.0, cfg_renorm_type='global', image_shapes=(512, 512))

inferencer = InterleaveInferencer(
    model=model, vae_model=vae_model, tokenizer=tokenizer,
    vae_transform=ImageTransform(512, 256, 8), vit_transform=vit_transform,
    new_token_ids=new_token_ids)

gen = TextFileGameDataGenerator('dataset/ocr', 'train', prompt_type='ocr', num_players=4)

N_GAMES = 4
all_vote_tokens = []
all_has_answer = []
all_has_think = []
all_truncated = []
all_correct = []
total_vote_time = 0

print(f"Running {N_GAMES} games × 4 players = {N_GAMES*4} votes\n")

for g in range(N_GAMES):
    game = gen.generate_game(0, g)
    spy = game['spy_player']
    prompts = [gen.format_generation_prompt_simple(game, pid) for pid in range(1, 5)]

    # Generate images
    images = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        for pid in range(4):
            out = inferencer(text=prompts[pid], noise_level=1.3, num_timesteps=15,
                             cfg_text_scale=4.0, grpo_config=grpo_config, **ihyper)
            images.append(out['image'])

    # Build grid
    grid_t = build_voting_grid_tensor(images, cell_size=512)
    grid_pil = grid_tensor_to_pil(grid_t, num_players=4, cell_size=512)

    # Batch vote
    vote_prompts = [gen.format_voting_prompt(game, player_id=pid) for pid in range(1, 5)]
    torch.cuda.synchronize()
    t0 = time.time()
    vote_texts = run_bagel_votes_batch(
        model=model, tokenizer=tokenizer, new_token_ids=new_token_ids,
        vit_transform=vit_transform, grid_images=[grid_pil]*4,
        vote_prompts=vote_prompts, max_tokens=512, temperature=0.7)
    torch.cuda.synchronize()
    t_vote = time.time() - t0
    total_vote_time += t_vote

    print(f"Game {g+1} (spy=P{spy}): vote time={t_vote:.1f}s")

    for pid in range(4):
        text = vote_texts[pid]
        n_tokens = len(tokenizer.encode(text))
        has_think = '<think>' in text and '</think>' in text
        has_answer = '<answer>' in text and '</answer>' in text
        truncated = not has_answer and n_tokens >= 500
        role = "SPY" if pid + 1 == spy else "CIV"

        # Extract vote
        vote_info = gen.extract_vote(text)
        voted = vote_info.get('voted_spy', None) if vote_info else None
        correct = (voted == spy) if isinstance(voted, int) else False

        all_vote_tokens.append(n_tokens)
        all_has_answer.append(has_answer)
        all_has_think.append(has_think)
        all_truncated.append(truncated)
        all_correct.append(correct)

        status = "✓" if has_answer else ("TRUNCATED" if truncated else "no_answer")
        vote_str = str(voted) if voted else "INVALID"
        correct_str = "✓" if correct else "✗"
        print(f"  P{pid+1}({role}): {n_tokens:3d} tok | think={has_think} | answer={has_answer} "
              f"| {status} | vote={vote_str} {correct_str}")

import numpy as np
print(f"\n{'='*60}")
print(f"SUMMARY ({N_GAMES} games × 4 players = {len(all_vote_tokens)} votes)")
print(f"{'='*60}")
print(f"Token length: mean={np.mean(all_vote_tokens):.0f}, "
      f"median={np.median(all_vote_tokens):.0f}, "
      f"min={np.min(all_vote_tokens)}, max={np.max(all_vote_tokens)}")
print(f"Has <think>: {sum(all_has_think)}/{len(all_has_think)} "
      f"({sum(all_has_think)/len(all_has_think)*100:.0f}%)")
print(f"Has <answer>: {sum(all_has_answer)}/{len(all_has_answer)} "
      f"({sum(all_has_answer)/len(all_has_answer)*100:.0f}%)")
print(f"Truncated (hit 512 limit): {sum(all_truncated)}/{len(all_truncated)} "
      f"({sum(all_truncated)/len(all_truncated)*100:.0f}%)")
print(f"Correct votes: {sum(all_correct)}/{len(all_correct)} "
      f"({sum(all_correct)/len(all_correct)*100:.0f}%)")
print(f"Total vote time: {total_vote_time:.1f}s "
      f"({total_vote_time/len(all_vote_tokens):.1f}s/vote, "
      f"{total_vote_time/sum(all_vote_tokens)*1000:.1f}ms/token)")
