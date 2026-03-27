"""
Standalone OCR Accuracy evaluation for Bagel model.

Usage (single GPU):
    python scripts/eval_bagel_ocr.py \
        --model_path ByteDance-Seed/BAGEL-7B-MoT \
        --dataset dataset/ocr \
        --output_dir eval_output/ocr_bagel

Usage (multi-GPU via accelerate):
    accelerate launch --config_file scripts/accelerate_configs/fsdp.yaml \
        --num_machines 1 --num_processes 8 --main_process_port 29501 \
        scripts/eval_bagel_ocr.py \
        --model_path ByteDance-Seed/BAGEL-7B-MoT \
        --dataset dataset/ocr
"""

import argparse
import contextlib
import json
import os
import time
from collections import defaultdict
from concurrent import futures
from functools import partial

import numpy as np
import torch
from PIL import Image

from accelerate import Accelerator, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from flow_grpo.bagel.data.data_utils import add_special_tokens
from flow_grpo.bagel.data.transforms import ImageTransform
from flow_grpo.bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from flow_grpo.bagel.modeling.qwen2 import Qwen2Tokenizer
from flow_grpo.bagel.modeling.autoencoder import load_ae
from flow_grpo.bagel.inferencer import InterleaveInferencer
from flow_grpo.ocr import OcrScorer

from torch.utils.data import Dataset, DataLoader
from huggingface_hub import snapshot_download

import tqdm as _tqdm
tqdm = partial(_tqdm.tqdm, dynamic_ncols=True)

logger = get_logger(__name__)


class TextPromptDataset(Dataset):
    def __init__(self, dataset_dir, split='test', max_samples=None):
        file_path = os.path.join(dataset_dir, f'{split}.txt')
        with open(file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines() if line.strip()]
        if max_samples is not None:
            self.prompts = self.prompts[:max_samples]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}

    @staticmethod
    def collate_fn(examples):
        return [ex["prompt"] for ex in examples]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bagel model OCR Accuracy")
    parser.add_argument("--model_path", type=str, default="ByteDance-Seed/BAGEL-7B-MoT",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA checkpoint (optional)")
    parser.add_argument("--dataset", type=str, default="dataset/ocr",
                        help="Path to OCR dataset directory containing test.txt")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate (default: test)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to evaluate (default: all)")
    parser.add_argument("--output_dir", type=str, default="eval_output/ocr_bagel",
                        help="Directory to save generated images and results")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Image resolution for generation")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of diffusion inference steps")
    parser.add_argument("--guidance_scale", type=float, default=4.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--timestep_shift", type=float, default=3.0,
                        help="Timestep shift for Bagel")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_images", action="store_true",
                        help="Save generated images to output_dir")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU (Bagel generates one image at a time, this controls dataloader)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Accelerator setup ──
    accelerator = Accelerator(mixed_precision="bf16")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Eval config: {vars(args)}")

    set_seed(args.seed, device_specific=True)

    inference_dtype = torch.bfloat16

    # ── Load Bagel model ──
    model_path = args.model_path
    if not os.path.exists(model_path):
        model_local_dir = snapshot_download(repo_id=model_path)
    else:
        model_local_dir = model_path

    llm_config = Qwen2Config.from_json_file(os.path.join(model_local_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vae_model, vae_config = load_ae(local_path=os.path.join(model_local_dir, "ae.safetensors"))

    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=False,
        llm_config=llm_config,
        vit_config=None,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        model = Bagel(language_model, None, bagel_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_local_dir)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(512, 256, 8)
    vit_transform = ImageTransform(490, 112, 7)

    logger.info(f"Loading checkpoint on device cuda:{accelerator.local_process_index}")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_local_dir, "ema.safetensors"),
        device_map={"": f"cuda:{accelerator.local_process_index}"},
        offload_buffers=False,
        dtype=inference_dtype,
        force_hooks=True,
        offload_folder="/adialab/usr/shadabk/MedUMM/.offload"
    )
    model = model.eval()
    model.requires_grad_(False)
    vae_model.requires_grad_(False)
    vae_model.to(accelerator.device, dtype=inference_dtype)

    # Load LoRA if provided
    if args.lora_path:
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(
            model.language_model, args.lora_path
        )
        logger.info(f"Loaded LoRA from {args.lora_path}")

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids
    )

    # Build a minimal config-like object for inferencer compatibility
    import ml_collections
    grpo_config = ml_collections.ConfigDict()
    grpo_config.sample = ml_collections.ConfigDict()
    grpo_config.sample.sde_window_size = 2
    grpo_config.sample.sde_window_range = (0, args.num_steps // 2)

    inference_hyper = dict(
        cfg_img_scale=1.0,
        cfg_interval=[0, 1.0],
        timestep_shift=args.timestep_shift,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(args.resolution, args.resolution),
    )

    # ── Dataset ──
    dataset = TextPromptDataset(args.dataset, split=args.split, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=TextPromptDataset.collate_fn,
        shuffle=False,
        num_workers=4,
    )
    dataloader = accelerator.prepare(dataloader)

    logger.info(f"Evaluating {len(dataset)} prompts on {accelerator.num_processes} GPUs")

    # ── OCR Scorer ──
    ocr_scorer = OcrScorer(use_gpu=False)
    executor = futures.ThreadPoolExecutor(max_workers=4)

    # ── Evaluation loop ──
    all_scores = []
    all_prompts = []
    all_results = []
    sample_idx = 0

    autocast = accelerator.autocast

    for batch_prompts in tqdm(
        dataloader,
        desc="Generating & scoring",
        disable=not accelerator.is_local_main_process,
    ):
        images = []
        with autocast():
            for prompt in batch_prompts:
                with torch.no_grad():
                    output_dict = inferencer(
                        text=prompt,
                        noise_level=0,
                        grpo_config=grpo_config,
                        accelerator=accelerator,
                        cfg_text_scale=args.guidance_scale,
                        num_timesteps=args.num_steps,
                        **inference_hyper,
                    )
                images.append(output_dict['image'])

        # Convert to numpy for OCR (NCHW float [0,1] → NHWC uint8)
        images_tensor = torch.stack(images, dim=0)
        images_np = (images_tensor * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images_np = images_np.transpose(0, 2, 3, 1)  # NCHW → NHWC

        # Score OCR
        scores = ocr_scorer(
            [Image.fromarray(img) for img in images_np],
            batch_prompts
        )

        # Save images if requested
        if args.save_images and accelerator.is_local_main_process:
            img_dir = os.path.join(args.output_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            for i, img_np in enumerate(images_np):
                pil = Image.fromarray(img_np)
                pil.save(os.path.join(img_dir, f"{sample_idx + i:05d}.png"))

        # Gather results across GPUs
        scores_tensor = torch.tensor(scores, device=accelerator.device, dtype=torch.float32)
        gathered_scores = accelerator.gather(scores_tensor).cpu().numpy()

        # Gather prompts via tokenizer encoding
        prompt_ids = tokenizer(
            batch_prompts,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(accelerator.device)
        gathered_prompt_ids = accelerator.gather(prompt_ids).cpu().numpy()
        gathered_prompts = tokenizer.batch_decode(gathered_prompt_ids, skip_special_tokens=True)

        if accelerator.is_main_process:
            for prompt, score in zip(gathered_prompts, gathered_scores):
                all_prompts.append(prompt)
                all_scores.append(float(score))
                all_results.append({"prompt": prompt, "ocr_accuracy": float(score)})

        sample_idx += len(batch_prompts) * accelerator.num_processes

    # ── Aggregate & report ──
    if accelerator.is_main_process:
        scores_array = np.array(all_scores)
        mean_acc = scores_array.mean()
        median_acc = np.median(scores_array)
        perfect_ratio = (scores_array == 1.0).mean()
        zero_ratio = (scores_array == 0.0).mean()

        report = {
            "model": args.model_path,
            "lora_path": args.lora_path,
            "dataset": args.dataset,
            "split": args.split,
            "num_samples": len(all_scores),
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "resolution": args.resolution,
            "mean_ocr_accuracy": float(mean_acc),
            "median_ocr_accuracy": float(median_acc),
            "perfect_accuracy_ratio": float(perfect_ratio),
            "zero_accuracy_ratio": float(zero_ratio),
            "std_ocr_accuracy": float(scores_array.std()),
        }

        print("\n" + "=" * 60)
        print("OCR Accuracy Evaluation Results")
        print("=" * 60)
        print(f"  Model:            {args.model_path}")
        if args.lora_path:
            print(f"  LoRA:             {args.lora_path}")
        print(f"  Dataset:          {args.dataset} ({args.split})")
        print(f"  Num samples:      {len(all_scores)}")
        print(f"  Num steps:        {args.num_steps}")
        print(f"  Guidance scale:   {args.guidance_scale}")
        print(f"  Resolution:       {args.resolution}")
        print("-" * 60)
        print(f"  Mean OCR Acc:     {mean_acc:.4f}")
        print(f"  Median OCR Acc:   {median_acc:.4f}")
        print(f"  Std OCR Acc:      {scores_array.std():.4f}")
        print(f"  Perfect (=1.0):   {perfect_ratio:.2%}")
        print(f"  Zero (=0.0):      {zero_ratio:.2%}")
        print("=" * 60)

        # Save detailed results
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)

        with open(os.path.join(args.output_dir, "detailed_results.jsonl"), "w") as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
