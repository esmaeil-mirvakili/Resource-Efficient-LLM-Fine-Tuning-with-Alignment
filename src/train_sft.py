"""QLoRA SFT training script for a 7B model (e.g., Mistral-7B)."""

import argparse
import os
import math
import time
from typing import Dict, List
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
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import login


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
    parser.add_argument("--hf_cache_path", type=str, default="/opt/ml/.cache/huggingface")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=6)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
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
        "--grad_ckpt", type=bool, default=True, help="Use gradient checkpointing"
    )
    parser.add_argument(
        "--ds_config", type=str, default=None, help="DeepSpeed config JSON file path"
    )
    parser.add_argument(
        "--ddp_find_unused_parameters", type=bool, default=True, help="For DDP"
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument(
        "--assistance_only_loss",
        type=bool,
        default=True,
        help="Use collator to mask prompt tokens in loss",
    )
    parser.add_argument(
        "--use_flash_attn", type=bool, default=True, help="Use flash attention"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help='Path to a checkpoint directory to resume from, or "auto" to pick the latest in output_dir',
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # W&B
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="sft_run")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


class VramPeakCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        if not getattr(state, "is_local_process_zero", True):
            return
        if torch.cuda.is_available() and wandb.run is not None:
            peak_gb = torch.cuda.max_memory_reserved() / (1024**3)
            wandb.log(
                {
                    "train/peak_vram_gb": peak_gb,
                    "global_step": state.global_step,
                }
            )


class EvalPerplexityCallback(TrainerCallback):
    """Logs eval/perplexity from eval_loss."""

    def on_evaluate(self, args, state, control, **kwargs):
        if not getattr(state, "is_local_process_zero", True):
            return
        if wandb.run is None:
            return
        metrics = kwargs.get("metrics", {})
        if "eval_loss" in metrics:
            ppl = (
                math.exp(metrics["eval_loss"])
                if metrics["eval_loss"] < 20
                else float("inf")
            )
            wandb.log({"eval/perplexity": ppl, "global_step": state.global_step})


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
                {
                    "train/step_time_ms": step_time_ms,
                    "train/tokens_per_sec": toks / dt,
                    "global_step": state.global_step,
                },
            )
        self.last_t, self.last_step = now, state.global_step


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


def get_lora_target_modules(model_name: str) -> List[str]:
    if model_name == "mistralai/Mistral-7B-v0.1":
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif model_name == "mistralai/Mixtral-8x7B-v0.1":
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "w1",
            "w2",
            "w3",
        ]
    logger.warning(
        f"Model {model_name} not recognized for automatic lora target module selection. Choosing default set."
    )
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataloader_num_workers > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    init_hf_hub()

    set_seed(args.seed)

    ckpt_to_resume = None
    if args.resume_from:
        if args.resume_from.lower() == "auto":
            ckpt_to_resume = get_last_checkpoint(args.output_dir)
            if ckpt_to_resume is None:
                logger.warning("No checkpoint found in output_dir; starting fresh.")
        else:
            ckpt_to_resume = args.resume_from
            if not os.path.isdir(ckpt_to_resume):
                raise FileNotFoundError(f"--resume_from path not found: {ckpt_to_resume}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    logger.info(f"Loading tokenizer for the base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.hf_cache_path)
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
        attn_implementation="flash_attention_2" if args.use_flash_attn else None,
        cache_dir=args.hf_cache_path,
    )
    # turn off KV cache for training
    model.config.use_cache = False
    # set model pad token id
    model.config.pad_token_id = tokenizer.pad_token_id
    # keep existing BOS/EOS; typically bos=1, eos=2 for Mistral
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    # gradient checkpointing
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    target_modules = get_lora_target_modules(args.model_name)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load dataset(s)
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    dataset = load_dataset("json", data_files=data_files)

    def _has_nonempty_response(ex):
        r = ex.get("response")
        return isinstance(r, str) and len(r.strip()) > 0

    dataset = dataset.filter(_has_nonempty_response)
    dataset = dataset.shuffle(seed=args.seed)

    def format_example(example) -> str:
        response = example.get("response", "")
        prompt = example.get("prompt", "")
        if response is None:
            response = ""
        return f"<|user|>\n{prompt}\n<|assistant|>\n{response}".strip()

    def tokenize_function(ex: Dict[str, str]) -> Dict[str, List[int]]:
        text = format_example(ex)
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
            return_attention_mask=True,
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

    def make_completion_collator(tokenizer, response_template="<|assistant|>\n"):
        """
        Data collator for SFT with causal LMs.

        Pads a batch of examples and sets labels for all prompt tokens to -100
        (ignore index), so the loss is computed only on the assistant’s response.
        This prevents the model from wasting capacity predicting the prompt.
        """
        resp_ids = tokenizer.encode(response_template, add_special_tokens=False)

        def find_sublist_start(seq_ids, pattern):
            if not pattern:
                return 0
            # naive sublist search
            for j in range(0, len(seq_ids) - len(pattern) + 1):
                if seq_ids[j : j + len(pattern)] == pattern:
                    return j + len(pattern)
            return 0  # fallback if tag not found

        def collate_fn(batch):
            # tensors for input_ids
            input_ids_list = [
                torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch
            ]

            # attention masks: use provided if present, else all-ones
            if "attention_mask" in batch[0]:
                attn_list = [
                    torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch
                ]
            else:
                attn_list = [
                    torch.ones(len(ids), dtype=torch.long) for ids in input_ids_list
                ]

            # pad
            input_ids = pad_sequence(
                input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            attention_mask = pad_sequence(attn_list, batch_first=True, padding_value=0)

            # labels = input_ids, but mask prompt tokens with -100 (ignore index)
            labels = input_ids.clone()
            for i, seq in enumerate(input_ids_list):
                seq_ids = seq.tolist()
                start = find_sublist_start(seq_ids, resp_ids)
                labels[i, :start] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        return collate_fn

    dataset = dataset.map(
        tokenize_function, 
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(
            test_size=args.eval_ratio, seed=args.seed, shuffle=True
        )

    eval_split = "validation" if "validation" in dataset else "test"

    logger.info(f"Train dataset size: {len(dataset['train'])}")
    if "validation" in dataset:
        logger.info(f"Validation dataset size: {len(dataset['validation'])}")

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
        peft_config=lora_cfg,
        processing_class=tokenizer,
        data_collator=(
            make_completion_collator(tokenizer) if args.assistance_only_loss else None
        ),
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

    logger.info("Starting training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=ckpt_to_resume)

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
