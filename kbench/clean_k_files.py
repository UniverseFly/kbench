from fire import Fire
from pathlib import Path
from datasets import load_dataset
from tqdm.auto import tqdm

ORGANIZATION = "runtimeverification"


def get_stats(text: str) -> dict:
    num_lines = len(text.splitlines())
    max_line_length = max(len(line) for line in text.splitlines())
    return {"num_lines": num_lines, "max_line_length": max_line_length}


def map_fn(x):
    # Remove copyright string
    x["content"] = (
        x["content"]
        .replace("// Copyright (c) Runtime Verification, Inc. All Rights Reserved.", "")
        .strip()
    )
    return {
        **x,
        **get_stats(x["content"]),
    }


def main(
    dataset_path: str = "/scratch/yuxiang-data/kfiles/raw_dataset.jsonl",
    save_path: str = "/scratch/yuxiang-data/kfiles/cleaned_dataset.jsonl",
    min_lines: int = 20,
    max_lines: int = 500,
    max_characters_per_line: int = 100,
    num_proc: int = 64,
):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(map_fn, num_proc=num_proc)
    dataset = dataset.filter(
        lambda x: x["num_lines"] <= max_lines
        and x["max_line_length"] <= max_characters_per_line
        and x["num_lines"] >= min_lines
    )
    print(f"Saving {len(dataset)} items to {save_path}")
    dataset.to_json(save_path, lines=True)


if __name__ == "__main__":
    Fire(main)
