from fire import Fire
from pathlib import Path
from datasets import load_dataset
import random

PROMPT = Path("kbench/kbench_prompt.txt").read_text().strip()


def get_stats(text: str) -> dict:
    num_lines = len(text.splitlines())
    max_line_length = max(len(line) for line in text.splitlines())
    return {"num_lines": num_lines, "max_line_length": max_line_length}


def map_fn(x):
    # Remove copyright string
    random.seed(x["repo_id"] + x["path"])
    part = random.choice(["top", "middle", "bottom"])
    content = x["content"]
    if part == "top":
        partial_content = content[: len(content) // 3]
    elif part == "middle":
        partial_content = content[len(content) // 3 : (2 * len(content)) // 3]
    else:
        partial_content = content[(2 * len(content)) // 3 :]
    assert len(partial_content.strip()) > 0
    num_lines = len(content.splitlines())
    prompt = PROMPT.format(part=part, context=partial_content, num_lines=num_lines)
    return {
        "repo_id": x["repo_id"],
        "path": x["path"],
        "prompt": prompt,
        "ground_truth": content,
        "part": part,
    }


def main(
    dataset_path: str = "/scratch/yuxiang-data/kfiles/cleaned_dataset.jsonl",
    save_path: str = "/scratch/yuxiang-data/kfiles/kbench.jsonl",
    num_proc: int = 64,
):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(
        map_fn, num_proc=num_proc, remove_columns=dataset.column_names
    )
    print(f"Saving {len(dataset)} items to {save_path}")
    dataset.to_json(save_path, lines=True)


if __name__ == "__main__":
    Fire(main)
