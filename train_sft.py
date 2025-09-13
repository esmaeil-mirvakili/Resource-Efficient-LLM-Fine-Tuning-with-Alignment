"""QLoRA SFT training script for a 7B model (e.g., Mistral-7B)."""

import os
import math
import json
from loguru import logger
import torch
from torch.nn.utils.rnn import pad_sequence
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
import hydra
from utils import hydra_arg_fix, align_tokenizer_and_model, VramPeakCallback


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


def flash_attn_available() -> str | None:
    try:
        import flash_attn

        return True
    except Exception:
        logger.warning("flash-attn not available; falling back to default attention.")
        return False


def is_bfloat16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def preprocess_dataset(dataset_config):
    os.makedirs(dataset_config.dataset_output_path, exist_ok=True)
    dataset_paths = {
        split: os.path.join(dataset_config.dataset_output_path, f"{split}.jsonl")
        for split in dataset_config.splits
    }
    for split in dataset_config.splits:
        if os.path.exists(dataset_paths[split]):
            logger.info(
                f"Dataset file {dataset_paths[split]} already exists. Skipping preprocessing."
            )
            continue
        logger.info(
            f"Preprocessing {split} dataset and saving to {dataset_paths[split]}"
        )
        logger.info(f"Loading dataset: {dataset_config.dataset_name} (split={split})")
        dataset = load_dataset(dataset_config.dataset_name, split=split)

        cols = set(dataset.column_names)
        column_map = dataset_config.column_map or {}
        if not any(col in cols for col in column_map.values()):
            raise ValueError(
                f"Could not infer the provided column mapping for dataset with columns {sorted(cols)}."
            )
        logger.info(f"Using columns: {column_map}")

        # Normalize + filter
        def normalize_record(record):
            normalized_record = {}
            for col in column_map.keys():
                value = record.get(column_map[col], None)
                if value is None:
                    value = ""
                value = str(value).strip()
                if not value:
                    return None
                # Basic whitespace normalization
                value = "\n".join(line.rstrip() for line in value.splitlines()).strip()
                normalized_record[col] = value
            return normalized_record

        normalized = dataset.map(
            normalize_record,
            remove_columns=[
                c for c in dataset.column_names if c not in column_map.keys()
            ],
        )
        normalized = normalized.filter(lambda r: r is not None)

        normalized = Dataset.from_dict(
            {k: list(normalized[k]) for k in normalized.features}
        )

        if getattr(dataset_config, "deduplicate", None):
            # Deduplicate by prompt
            before = len(normalized)
            seen = set()
            keep_idx = []
            for i, v in enumerate(normalized[dataset_config.deduplicate]):
                if v not in seen:
                    seen.add(v)
                    keep_idx.append(i)
            normalized = normalized.select(keep_idx)
            logger.info("Dropped {} duplicate prompts", before - len(normalized))
        written_cnt = 0
        with open(dataset_paths[split], "w", encoding="utf-8") as f:
            for r in normalized:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                written_cnt += 1
            logger.info(f"Wrote {written_cnt} rows in {dataset_paths[split]}")

    return dataset_paths


def prepare_dataset(dataset_config, example_template, seed):
    data_files = preprocess_dataset(dataset_config)

    # Load dataset
    dataset = load_dataset("json", data_files=data_files)

    if getattr(dataset_config, "shuffle", None):
        dataset = dataset.shuffle(seed=seed)

    if getattr(dataset_config, "limit", -1) > 0:
        limit = dataset_config.limit
        if "train" in dataset:
            dataset["train"] = dataset["train"].select(
                range(min(limit, len(dataset["train"])))
            )

    def format_example(example) -> str:
        return {dataset_config.data_text_field: example_template.format(**example)}

    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names["train"],
    )

    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(
            test_size=dataset_config.eval_ratio, seed=seed, shuffle=True
        )

    return dataset


def prepare_model_tokenizer(model_config):
    logger.info(f"Loading tokenizer for the base model: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        use_fast=model_config.tokenizer.use_fast,
        trust_remote_code=getattr(model_config.tokenizer, "trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = model_config.tokenizer.padding_side

    logger.info(f"Creating the model {model_config.model_name} with 4-bit quantization")
    bnb_config = BitsAndBytesConfig(
        bnb_4bit_compute_dtype=(
            torch.bfloat16
            if getattr(model_config, "use_bf16", None) and is_bfloat16_supported()
            else torch.float16
        ),
        **model_config.bnb_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map=None if in_distributed_mode() else "auto",
        dtype=(
            torch.bfloat16
            if getattr(model_config, "use_bf16", None) and is_bfloat16_supported()
            else torch.float16
        ),
        trust_remote_code=getattr(model_config.model, "trust_remote_code", False),
        attn_implementation=(
            getattr(model_config.model, "attn_implementation", None)
            if flash_attn_available()
            else None
        ),
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    align_tokenizer_and_model(tokenizer, model)

    lora_config = LoraConfig(
        **model_config.lora_config,
    )

    return model, tokenizer, lora_config


def tokenize_dataset(dataset, dataset_config, tokenizer, tokenizer_config):

    def tokenize_function(ex):
        text = ex[dataset_config.data_text_field]
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=tokenizer_config.seq_max_length,
            padding=False,
            return_attention_mask=True,
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

    dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    return dataset


def prepare_trainer(
    trainer_config,
    lora_config,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    data_collator=None,
):
    # gradient checkpointing
    if getattr(trainer_config.training_arguments, "gradient_checkpointing", None):
        model.gradient_checkpointing_enable()
    # TrainingArguments
    training_args = SFTConfig(
        **trainer_config.training_arguments,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=(
            [
                VramPeakCallback(),
                EvalPerplexityCallback(),
            ]
        ),
    )

    if getattr(trainer_config, "early_stopping", None):
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=trainer_config.early_stopping.early_stopping_patience,
                early_stopping_threshold=trainer_config.early_stopping.early_stopping_threshold,
            )
        )

    return trainer


def train(trainer, trainer_config):
    ckpt_to_resume = None
    if getattr(trainer_config, "resume_from", None):
        if getattr(trainer_config, "resume_from", "").lower() == "auto":
            ckpt_to_resume = get_last_checkpoint(
                trainer_config.training_arguments.output_dir
            )
            if ckpt_to_resume is None:
                logger.warning("No checkpoint found in output_dir; starting fresh.")
        else:
            ckpt_to_resume = trainer_config.resume_from
            if not os.path.isdir(ckpt_to_resume):
                raise FileNotFoundError(
                    f"--resume_from path not found: {ckpt_to_resume}"
                )
    logger.info("Starting training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=ckpt_to_resume)


@hydra.main(version_base="1.3", config_path="configs/sft", config_name="sft_config")
def main(config=None):
    init_hf_hub()
    seed = config.get("seed", 42)
    set_seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if getattr(config, "wandb", None):
        os.environ.setdefault("WANDB_PROJECT", config.wandb.project)
        wandb.init(
            project=config.wandb.project,
            name=config.wandb.run_name,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    model, tokenizer, lora_config = prepare_model_tokenizer(config.model)

    dataset = prepare_dataset(config.dataset, config.model.example_template, seed)

    eval_split = "validation" if "validation" in dataset else "test"

    logger.info(f"Train dataset size: {len(dataset['train'])}")
    if "validation" in dataset:
        logger.info(f"Validation dataset size: {len(dataset['validation'])}")

    dataset = tokenize_dataset(
        dataset, config.dataset, tokenizer, config.model.tokenizer
    )

    def make_completion_collator(tokenizer, completion_template):
        """
        Data collator for SFT with causal LMs.

        Pads a batch of examples and sets labels for all prompt tokens to -100
        (ignore index), so the loss is computed only on the assistant’s response.
        This prevents the model from wasting capacity predicting the prompt.
        """
        resp_ids = tokenizer.encode(completion_template, add_special_tokens=False)

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

    trainer = prepare_trainer(
        config.trainer,
        lora_config,
        model,
        tokenizer,
        dataset["train"],
        dataset[eval_split],
        data_collator=(
            make_completion_collator(tokenizer, config.model.completion_template)
            if getattr(config.model, "completion_only_loss", None)
            else None
        ),
    )

    train(trainer, config.trainer)

    # Save PEFT model and tokenizer
    logger.info(f"Saving the model and tokenizer to {config.trainer.output_dir}")
    trainer.save_model(config.trainer.training_arguments.output_dir)
    tokenizer.save_pretrained(config.trainer.training_arguments.output_dir)

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
    hydra_arg_fix()
    main()
