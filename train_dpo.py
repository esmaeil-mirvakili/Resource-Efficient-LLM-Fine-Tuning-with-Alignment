import os
import json
from datasets import load_dataset, Dataset
from loguru import logger
from utils import (
    config_to_plain,
    hydra_arg_fix,
    align_tokenizer_and_model,
    create_hf_gitignore_file,
    VramPeakCallback,
)
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig
from datasets import load_dataset, Dataset
from trl import DPOTrainer, DPOConfig
from peft import get_peft_model
from huggingface_hub import login
import hydra


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


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
        required_keys = {"chosen", "rejected"}

        if not required_keys.issubset(column_map.keys()):
            raise ValueError(
                f"column_map must contain {required_keys}, got {column_map.keys()}"
            )

        if not any(col in cols for col in column_map.values()):
            raise ValueError(
                f"Could not infer the provided column mapping for dataset with columns {sorted(cols)}."
            )
        logger.info(f"Using columns: {column_map}")

        # Normalize and filter
        def normalize_record(record):
            normalized_record = {}
            for col in required_keys:
                value = record.get(column_map[col], None)
                if value is None:
                    return None
                value = str(value).strip()
                if not value:
                    return None
                # Basic whitespace normalization
                value = "\n".join(line.rstrip() for line in value.splitlines()).strip()
                normalized_record[col] = value
            # skip degenerate pairs
            if normalized_record["chosen"] == normalized_record["rejected"]:
                return None
            return normalized_record

        normalized = dataset.map(
            normalize_record,
            remove_columns=[
                c for c in dataset.column_names if c not in set(column_map.values())
            ],
        )
        normalized = normalized.filter(lambda r: r is not None)

        # Convert to a clean Dataset
        normalized = Dataset.from_dict({k: list(normalized[k]) for k in required_keys})

        if getattr(dataset_config, "deduplicate", None):
            key = dataset_config.deduplicate
            if key not in {"chosen", "rejected"}:
                raise ValueError(
                    f"deduplicate must be one of chosen/rejected, got {key}"
                )
            before = len(normalized)
            seen = set()
            keep_idx = []
            for i, v in enumerate(normalized[key]):
                if v not in seen:
                    seen.add(v)
                    keep_idx.append(i)
            normalized = normalized.select(keep_idx)
            logger.info("Dropped {} duplicate {}", before - len(normalized), key)

        written_cnt = 0
        with open(dataset_paths[split], "w", encoding="utf-8") as f:
            for r in normalized:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                written_cnt += 1
            logger.info(f"Wrote {written_cnt} rows in {dataset_paths[split]}")

    return dataset_paths


def prepare_dataset(dataset_config, seed):
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

    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(
            test_size=dataset_config.eval_ratio, seed=seed, shuffle=True
        )

    return dataset


def prepare_model_tokenizer_dpo(model_config):
    logger.info(f"Loading tokenizer for base/SFT model: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        use_fast=model_config.tokenizer.use_fast,
        trust_remote_code=getattr(model_config.tokenizer, "trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = model_config.tokenizer.padding_side

    logger.info(
        f"Creating the POLICY model {model_config.model_name} with 4-bit quantization"
    )
    bnb_config = BitsAndBytesConfig(
        bnb_4bit_compute_dtype=(
            torch.bfloat16
            if getattr(model_config, "use_bf16", None) and is_bfloat16_supported()
            else torch.float16
        ),
        **config_to_plain(model_config.bnb_config),
    )
    policy = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
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
    policy.config.use_cache = False
    policy.config.pad_token_id = tokenizer.pad_token_id

    align_tokenizer_and_model(tokenizer, policy)

    # Apply LoRA to POLICY only
    lora_config = LoraConfig(**config_to_plain(model_config.lora_config))
    policy = get_peft_model(policy, lora_config)

    # Reference model (frozen). Default to same checkpoint unless config provides ref_model_name.
    ref_name = getattr(model_config, "model_name", None) or model_config.model_name
    logger.info(f"Loading REFERENCE model: {ref_name}")
    ref = AutoModelForCausalLM.from_pretrained(
        ref_name,
        quantization_config=bnb_config,  # quantized ref is fine; it’s frozen
        device_map="auto",
        dtype=(
            torch.bfloat16
            if getattr(model_config, "use_bf16", None) and is_bfloat16_supported()
            else torch.float16
        ),
        trust_remote_code=model_config.model.trust_remote_code,
        attn_implementation=(
            model_config.model.attn_implementation if flash_attn_available() else None
        ),
    )
    ref.config.use_cache = False
    ref.config.pad_token_id = tokenizer.pad_token_id
    align_tokenizer_and_model(tokenizer, ref)

    # Freeze the reference
    ref.requires_grad_(False)
    ref.eval()

    return policy, ref, tokenizer


def prepare_trainer_dpo(
    trainer_config,
    policy_model,
    ref_model,
    tokenizer,
    train_dataset,
    eval_dataset,
):
    os.makedirs(trainer_config.training_arguments.output_dir, exist_ok=True)
    if (
        getattr(trainer_config.training_arguments, "push_to_hub", False)
        and is_main_process()
    ):
        create_hf_gitignore_file(trainer_config.training_arguments.output_dir)
    if in_distributed_mode() and getattr(
        trainer_config.training_arguments, "deepspeed", None
    ):
        setattr(
            trainer_config.training_arguments,
            "load_best_model_at_end",
            False,
        )
    # gradient checkpointing
    if getattr(trainer_config.training_arguments, "gradient_checkpointing", None):
        policy_model.gradient_checkpointing_enable()

    # You can use HF TrainingArguments (works with TRL) for parity with SFT config
    training_args = DPOConfig(
        **config_to_plain(trainer_config.training_arguments),
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.add_callback(VramPeakCallback())
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

    logger.info("Starting DPO training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=ckpt_to_resume)


@hydra.main(version_base="1.3", config_path="configs/dpo", config_name="dpo_config")
def main(config=None):
    if is_main_process():
        init_hf_hub()
    seed = config.get("seed", 42)
    set_seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if getattr(config, "wandb", None) and is_main_process():
        os.environ.setdefault("WANDB_PROJECT", config.wandb.project)
        wandb.init(project=config.wandb.project, name=config.wandb.run_name)
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    policy, ref, tokenizer = prepare_model_tokenizer_dpo(config.model)

    dataset = prepare_dataset(config.dataset, seed)

    eval_split = "validation" if "validation" in dataset else "test"
    logger.info(f"Train pairs: {len(dataset['train'])}")
    if "validation" in dataset:
        logger.info(f"Validation pairs: {len(dataset['validation'])}")

    trainer = prepare_trainer_dpo(
        config.trainer,
        policy,
        ref,
        tokenizer,
        dataset["train"],
        dataset[eval_split],
    )

    train(trainer, config.trainer)

    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        # Save PEFT model and tokenizer
        logger.info(
            f"Saving the model and tokenizer to {config.trainer.training_arguments.output_dir}"
        )
        trainer.save_model(config.trainer.training_arguments.output_dir)
        tokenizer.save_pretrained(config.trainer.training_arguments.output_dir)
    trainer.accelerator.wait_for_everyone()

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_reserved() / (1024**3)
        logger.info("Peak reserved VRAM: {:.2f} GB", peak_gb)

    del policy, ref, trainer
    torch.cuda.empty_cache()
    logger.info("DPO training is done.")


if __name__ == "__main__":
    hydra_arg_fix()
    main()
