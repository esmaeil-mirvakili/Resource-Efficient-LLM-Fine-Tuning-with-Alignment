import sys
import os
import torch
from transformers import TrainerCallback
import wandb
from omegaconf import OmegaConf


def config_to_plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


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


def align_tokenizer_and_model(tokenizer, model):
    to_set = {}
    if tokenizer.pad_token_id is not None:
        to_set["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        to_set["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        to_set["bos_token_id"] = tokenizer.bos_token_id

    for k, v in to_set.items():
        setattr(model.config, k, v)
        if hasattr(model, "generation_config") and model.generation_config is not None:
            setattr(model.generation_config, k, v)


def hydra_arg_fix():
    if len(sys.argv) <= 1:
        return
    hydra_formatted_args = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a.startswith("--"):
            # support --key=value
            if "=" in a:
                k, v = a[2:].split("=", 1)
                hydra_formatted_args.append(f"{k}={v}")
                i += 1
            else:
                # support --key value   (value might be missing)
                k = a[2:]
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                    v = sys.argv[i + 1]
                    hydra_formatted_args.append(f"{k}={v}")
                    i += 2
                else:
                    # bare flag => true
                    hydra_formatted_args.append(f"{k}=true")
                    i += 1
        else:
            hydra_formatted_args.append(a)
            i += 1
    sys.argv = hydra_formatted_args


def create_hf_gitignore_file(path):
    with open(os.path.join(path, ".gitignore"), "w") as f:
        f.write("*.sagemaker-uploading\n")
        f.write("*.sagemaker-uploaded\n")
        f.write("*.tmp\n")
        f.write("*.lock\n")
