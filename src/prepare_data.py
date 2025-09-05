"""
Download and prepare SFT and preference data from a public dataset.
"""
import argparse
import os
import json
from typing import Dict, Iterable
from functools import partial
from datasets import load_dataset, Dataset
from loguru import logger


def write_jsonl(records: Iterable[Dict], path: str) -> None:
    """Write a sequence of record of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    logger.info(f"Wrote {n} rows in {path}")


def normalize_record(rec: Dict, input_cols: Dict[str, str]) -> Dict[str, str] | None:
    """Map a source row to {prompt, response} and lightly clean.
    Returns None if the row should be dropped.
    """
    prompt_text = rec.get(input_cols["prompt"], "")
    response_text = rec.get(input_cols["response"], "")

    if prompt_text is None:
        prompt_text = ""
    if response_text is None:
        response_text = ""

    prompt = str(prompt_text).strip()
    response = str(response_text).strip()

    # Drop obviously empty or degenerate examples
    if not prompt or not response:
        return None
    if len(response.split()) < 2:  # ultra-short answers are often noise
        return None

    # Basic whitespace normalization
    prompt = "\n".join(line.rstrip() for line in prompt.splitlines()).strip()
    response = "\n".join(line.rstrip() for line in response.splitlines()).strip()

    return {"prompt": prompt, "response": response}


def load_and_prepare_sft_dataset(
    dataset_name: str,
    split: str,
    limit: int | None,
    shuffle: bool,
    seed: int,
    drop_duplicates: bool,
) -> Dataset:
    """Load a public dataset and convert to {prompt, response} pairs.
    Currently optimized for databricks/databricks-dolly-15k
    """
    logger.info(f"Loading dataset: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split)

    # Column guessing: try common instruction/response fields
    candidate_mappings = {"prompt": ["instruction", "question", "prompt", "input"],
                          "response": ["response", "answer", "response", "output"]}
    cols = set(ds.column_names)
    chosen_map = {"prompt": None, "response": None}
    for prompt_map in candidate_mappings["prompt"]:
        if prompt_map in cols:
            chosen_map["prompt"] = prompt_map
            break
    for res_map in candidate_mappings["response"]:
        if res_map in cols:
            chosen_map["response"] = res_map
            break
    if chosen_map["prompt"] is None or chosen_map["response"] is None:
        raise ValueError(
            f"Could not infer prompt/response columns for dataset with columns {sorted(cols)}."
        )
    logger.info(
        f"Using columns: prompt={chosen_map['prompt']}, response={chosen_map['response']}"
    )

    # Normalize + filter
    normalized = ds.map(
        partial(normalize_record, input_cols=chosen_map),
        remove_columns=[c for c in ds.column_names if c not in chosen_map.values()],
    )
    normalized = normalized.filter(lambda r: r is not None)

    normalized = Dataset.from_dict(
        {k: list(normalized[k]) for k in normalized.features}
    )

    if shuffle:
        normalized = normalized.shuffle(seed=seed)
    if drop_duplicates:
        # Deduplicate by prompt
        before = len(normalized)
        seen = set()
        keep_idx = []
        for i, p in enumerate(normalized["prompt"]):
            if p not in seen:
                seen.add(p)
                keep_idx.append(i)
        normalized = normalized.select(keep_idx)
        logger.info("Dropped {} duplicate prompts", before - len(normalized))
    if limit is not None:
        normalized = normalized.select(range(min(limit, len(normalized))))

    logger.info("Prepared {} examples", len(normalized))
    return normalized


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare SFT and preference data from a public dataset"
    )
    parser.add_argument("--task", type=str, choices=["sft", "prefs"],
                        default="sft", help="Path to write SFT JSONL output")
    parser.add_argument("--output", type=str, default="data/", help="Folder path to write JSONL output")
    parser.add_argument("--dataset_name", type=str, default="databricks/databricks-dolly-15k",
                        help="HF dataset name or local path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before limiting")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--no_dedup", action="store_true", help="Disable de-duplication by prompt")
    return parser.parse_args()


def main():
    args = parse_args()
    records = None
    if args.task == "sft":
        ds = load_and_prepare_sft_dataset(
            dataset_name=args.dataset_name,
            split=args.split,
            limit=args.limit,
            shuffle=args.shuffle,
            seed=args.seed,
            drop_duplicates=not args.no_dedup,
        )
        records = ( {"prompt": p, "response": r} for p, r in zip(ds["prompt"], ds["response"]))
    elif args.task == "pref":
        raise NotImplementedError("Preference data preparation not yet implemented")
    if records:
        write_jsonl(records, os.path.join(args.output, f"{args.task}_{args.split}.jsonl"))

if __name__ == "__main__":
    main()
