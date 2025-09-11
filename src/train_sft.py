"""QLoRA SFT training script for a 7B model (e.g., Mistral-7B)."""

import argparse
import os
import math
import time
from typing import Dict, List, Optional
from loguru import logger
import torch
from torch.nn.utils.rnn import pad_sequence
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import login


class VramPeakCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        if not getattr(state, "is_local_process_zero", True):
            return
        if torch.cuda.is_available() and wandb.run is not None:
            peak_gb = torch.cuda.max_memory_reserved() / (1024**3)
            wandb.log({"train/peak_vram_gb": peak_gb}, step=state.global_step)


class EvalPerplexityCallback(TrainerCallback):
    """Logs eval/perplexity from eval_loss."""

    def on_evaluate(self, args, state, control, **kwargs):
        print("eval starts")
        if not getattr(state, "is_local_process_zero", True):
            return
        if wandb.run is None:
            return
        print("EvalPerplexityCallback.on_evaluate", kwargs.get("metrics", {}))
        metrics = kwargs.get("metrics", {})
        if "eval_loss" in metrics:
            ppl = (
                math.exp(metrics["eval_loss"])
                if metrics["eval_loss"] < 20
                else float("inf")
            )
            wandb.log({"eval/perplexity": ppl}, step=state.global_step)


class TokensPerSecAndStepTimeCallback(TrainerCallback):
    """Estimates and logs training tokens/sec at each logging step."""

    def __init__(self, seq_len_estimate: int):
        self.seq_len = seq_len_estimate
        self.last_t = None
        self.last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_t = time.time()
        self.last_step = 0

    def on_log(self, args, state, control, **kwargs):
        if not getattr(state, "is_local_process_zero", True):
            return
        if wandb.run is None:
            return
        now = time.time()
        steps = state.global_step - self.last_step
        dt = max(now - (self.last_t or now), 1e-9)

        world = getattr(args, "world_size", 1)
        toks = (
            steps
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * world
            * self.seq_len
        )
        if steps > 0:
            step_time_ms = (dt / steps) * 1000.0
            wandb.log(
                {"train/step_time_ms": step_time_ms, "train/tokens_per_sec": toks / dt},
                step=state.global_step,
            )
        self.last_t, self.last_step = now, state.global_step


def parse_args():
    parser = argparse.ArgumentParser(description="Train QLoRA SFT model")
    parser.add_argument(
        "--train_file", type=str, required=True, help="JSONL with {prompt,response}"
    )
    parser.add_argument(
        "--eval_file", type=str, default=None, help="Optional eval JSONL"
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.05,
        help="If no eval_file, take this ratio from train",
    )

    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")

    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="If >0, overrides epochs"
    )
    parser.add_argument(
        "--grad_ckpt", type=bool, default=False, help="Use gradient checkpointing"
    )
    parser.add_argument(
        "--ds_config", type=str, default=None, help="DeepSpeed config JSON file path"
    )
    parser.add_argument(
        "--ddp_find_unused_parameters", type=bool, default=False, help="For DDP"
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # W&B
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="sft_run")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def init_hf_hub():
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    if hf_token is None:
        logger.warning(
            "HuggingFace token not found in HUGGINGFACE_HUB_TOKEN env variable. Skipping login."
        )
    else:
        login(token=hf_token)
        logger.info("Logged in to HuggingFace Hub.")


def in_distributed_mode() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataloader_num_workers > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    init_hf_hub()

    set_seed(args.seed)

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    logger.info(f"Loading tokenizer for the base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"Creating the model {args.model_name} with 4-bit quantization")
    bf16_supported = (
        torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16_supported else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=None if in_distributed_mode() else "auto",
        dtype=torch.bfloat16 if bf16_supported else torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    # turn off KV cache for training
    model.config.use_cache = False
    # set model pad token id
    model.config.pad_token_id = tokenizer.pad_token_id
    # gradient checkpointing
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    # Prepare for k-bit training and wrap with LoRA
    # Common Mistral/LLaMA MLP+Attention proj layers
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable params: {n_train} / {n_total} ({100*n_train/n_total:.4f}%)")
    assert n_train > 0, "No trainable parameters found!"

    # Load dataset(s)
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    dataset = load_dataset("json", data_files=data_files)

    def _has_nonempty_response(ex):
        r = ex.get("response")
        return isinstance(r, str) and len(r.strip()) > 0

    dataset = dataset.filter(_has_nonempty_response)

    def preprocess_function(example):
        # 1) Build strings using your template
        prompt = example["prompt"]
        response = example["response"]

        prompt_text = f"<|user|>\n{prompt}\n<|assistant|>\n"
        resp_text = f"{response}"

        # 2) Ensure EOS so the model learns to stop
        if tokenizer.eos_token is not None and not resp_text.endswith(tokenizer.eos_token):
            resp_text = resp_text + tokenizer.eos_token

        # 3) Tokenize separately (no special tokens)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        resp_ids   = tokenizer(resp_text,   add_special_tokens=False)["input_ids"]

        # 4) Truncate to max_length, preferring to KEEP the response
        max_length = args.max_seq_len
        total_len = len(prompt_ids) + len(resp_ids)
        if total_len > max_length:
            overflow = total_len - max_length
            if overflow <= len(prompt_ids):
                # Drop from the LEFT of the prompt first
                prompt_ids = prompt_ids[overflow:]
            else:
                # Prompt fully dropped; trim remaining overflow from the LEFT of the response
                # (keep the tail so EOS is preserved)
                rem = overflow - len(prompt_ids)
                prompt_ids = []
                resp_ids = resp_ids[rem:]

        # 5) Concatenate and build labels/masks (mask prompt with -100)
        input_ids = prompt_ids + resp_ids
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids)) + resp_ids[:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataset = dataset.map(
        preprocess_function, remove_columns=dataset["train"].column_names
    )

    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(
            test_size=args.eval_ratio, seed=args.seed, shuffle=True
        )

    eval_split = "validation" if "validation" in dataset else "test"

    logger.info(f"Train dataset size: {len(dataset['train'])}")
    if "validation" in dataset:
        logger.info(f"Validation dataset size: {len(dataset['validation'])}")

    class CollatorKeepLabels:
        """Pads input_ids/attention_mask/labels while preserving -100 on labels."""
        def __init__(self, pad_id: int, label_pad_id: int = -100, right_pad: bool = True):
            self.pad_id = pad_id
            self.label_pad_id = label_pad_id
            self.right_pad = right_pad

        def __call__(self, features):
            ids  = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
            attn = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
            labs = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

            if self.right_pad:
                # Standard HF style: right padding
                batch_ids  = pad_sequence(ids,  batch_first=True, padding_value=self.pad_id)
                batch_attn = pad_sequence(attn, batch_first=True, padding_value=0)
                batch_labs = pad_sequence(labs, batch_first=True, padding_value=self.label_pad_id)
            else:
                # Left-padding: reverse, pad, reverse back
                def left_pad(seq_list, pad_val):
                    rev = [torch.flip(s, dims=[0]) for s in seq_list]
                    padded = pad_sequence(rev, batch_first=True, padding_value=pad_val)
                    return torch.flip(padded, dims=[1])

                batch_ids  = left_pad(ids,  self.pad_id)
                batch_attn = left_pad(attn, 0)
                batch_labs = left_pad(labs, self.label_pad_id)

            return {
                "input_ids": batch_ids,
                "attention_mask": batch_attn,
                "labels": batch_labs,
            }

    data_collator = CollatorKeepLabels(
        pad_id=tokenizer.pad_token_id,
        label_pad_id=-100,
        right_pad=True,  # set False if you really want left-padding
    )

    # TrainingArguments
    use_bf16 = bf16_supported
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if args.grad_ckpt else None
        ),
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        optim="paged_adamw_8bit",
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        logging_strategy="steps",
        eval_steps=args.eval_steps,
        do_eval=True,
        report_to=["wandb"] if args.wandb_project else [],
        run_name=args.run_name,
        seed=args.seed,
        deepspeed=args.ds_config,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        dataloader_num_workers=args.dataloader_num_workers,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[eval_split],
        data_collator=data_collator,
        peft_config=lora_cfg,
        callbacks=(
            [
                VramPeakCallback(),
                EvalPerplexityCallback(),
                TokensPerSecAndStepTimeCallback(seq_len_estimate=args.max_seq_len),
            ]
            if args.wandb_project
            else []
        ),
    )

    def simple_label_check(dataloader, num_batches=5, min_ok_ratio=0.8):
        """Quick sanity check across a few batches."""
        total = 0
        good = 0

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            labels = batch["labels"]
            # Count how many samples in this batch have at least one supervised token
            supervised = (labels != -100).any(dim=1)
            total += labels.size(0)
            good += supervised.sum().item()

        ratio = good / total if total else 0
        print(
            f"[check] {good}/{total} samples had supervised tokens (ratio={ratio:.2f})"
        )

        assert ratio >= min_ok_ratio, (
            f"Too many empty/fully masked samples (only {ratio:.2%} good). "
            f"Check your data formatting or collator."
        )

    # Run before trainer.train()
    simple_label_check(trainer.get_train_dataloader(), num_batches=5, min_ok_ratio=0.8)

    logger.info("Starting training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()

    # Save PEFT model and tokenizer
    logger.info(f"Saving the model and tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Efficiency stats
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_reserved() / (1024**3)
        logger.info("Peak reserved VRAM: {:.2f} GB", peak_gb)

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()

    logger.info("Training is done.")


if __name__ == "__main__":
    main()
