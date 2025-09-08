"""QLoRA SFT training script for a 7B model (e.g., Mistral-7B)."""

import argparse
import os
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
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import login


def format_example(prompt: str, response: Optional[str]) -> str:
    # Minimal, deterministic template
    # Note: For training, both prompt and response are included; inference code should stop at assistant tag.
    if response is None:
        response = ""
    return f"<|user|>\n{prompt}\n<|assistant|>\n{response}".strip()


def tokenize_function(tokenizer: AutoTokenizer, max_len: int):
    def _fn(ex: Dict[str, str]) -> Dict[str, List[int]]:
        text = format_example(ex["prompt"], ex.get("response"))
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_attention_mask=True,
        )
        # Let SFTTrainer handle labels from the same ids (causal LM)
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

    return _fn


class WandbVramLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and wandb.run is not None:
            peak_gb = torch.cuda.max_memory_reserved() / (1024**3)
            wandb.log({"train/peak_vram_gb": peak_gb}, step=state.global_step)


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
    parser.add_argument("--grad_ckpt", action="store_true", default=False, help="Use gradient checkpointing")

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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    init_hf_hub()

    set_seed(args.seed)

    logger.info(f"Loading tokenizer for the base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
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
        device_map="auto",
        trust_remote_code=True,
    )

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
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    # turn off KV cache for training
    model.config.use_cache = False
    # set model pad token id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load dataset(s)
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    raw = load_dataset("json", data_files=data_files)

    if "validation" not in raw:
        raw = raw["train"].train_test_split(test_size=args.eval_ratio, seed=args.seed)

    # Tokenize
    tok_fn = tokenize_function(tokenizer, args.max_seq_len)
    tokenized = raw.map(tok_fn, remove_columns=raw["train"].column_names)
    logger.info(f"Train dataset size: {len(tokenized['train'])}")
    if "validation" in tokenized:
        logger.info(f"Validation dataset size: {len(tokenized['validation'])}")

    # TrainingArguments
    use_bf16 = bf16_supported
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        do_eval=True,
        report_to=["wandb"] if args.wandb_project else [],
        run_name=args.run_name,
        seed=args.seed,
    )

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

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

    eval_split = "validation" if "validation" in tokenized else "test"

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[eval_split],
        data_collator=make_completion_collator(tokenizer),
        callbacks=[WandbVramLoggingCallback()] if args.wandb_project else [],
    )

    logger.info("Starting training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()

    # Save PEFT model and tokenizer
    logger.info(f"Saving the model and tokenizer to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Efficiency stats
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_reserved() / (1024**3)
        logger.info("Peak reserved VRAM: {:.2f} GB", peak_gb)

    logger.info("Training is done.")


if __name__ == "__main__":
    main()
