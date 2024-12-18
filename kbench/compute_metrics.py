"""Compute the metrics given inference results"""

from fire import Fire
from datasets import load_dataset, disable_progress_bars
import Levenshtein
import re


def calculate_edit_similarity(predictions: list[str], references: list[str]) -> float:
    total_similarity = 0.0
    for pred, ref in zip(predictions, references):
        distance = Levenshtein.distance(pred, ref)
        try:
            similarity = 1 - (distance / max(len(pred), len(ref)))
        except ZeroDivisionError as e:
            similarity = 1
        total_similarity += similarity
    return total_similarity / len(predictions)


def sanitize(text: str) -> str:
    """Extract the first ```k...``` code block from text"""
    code_blocks = re.findall(r"```k(.+?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    if "```k" in text:
        return text[text.index("```k") + 4 :].strip()
    return text


def map_fn(x: dict):
    x["code_output"] = sanitize(x["response"])
    x["edit_similarity"] = calculate_edit_similarity(
        [x["code_output"]], [x["ground_truth"]]
    )
    return x


def main(
    result_path: str = "/scratch/yuxiang-data/kbench-results/meta-llama__Llama-3.1-8B-Instruct",
):
    results = load_dataset("json", data_files=f"{result_path}/*.jsonl", split="train")
    results = results.map(map_fn, num_proc=64)

    disable_progress_bars()

    # Avg similarity
    avg_similarity = sum(results["edit_similarity"]) / len(results)
    print(f"ES: {avg_similarity:.4f}")

    print()

    # by repo_name
    all_repo_names = set(x["repo_id"] for x in results)
    for repo_name in all_repo_names:
        repo_results = results.filter(lambda x: x["repo_id"] == repo_name)
        avg_similarity = sum(repo_results["edit_similarity"]) / len(repo_results)
        print(f"[{repo_name} ({(len(repo_results))})] ES: {avg_similarity:.4f}")

    print()

    # by part
    all_parts = set(x["part"] for x in results)
    for part in all_parts:
        part_results = results.filter(lambda x: x["part"] == part)
        avg_similarity = sum(part_results["edit_similarity"]) / len(part_results)
        print(f"[{part} ({len(part_results)})] ES: {avg_similarity:.4f}")


if __name__ == "__main__":
    Fire(main)
