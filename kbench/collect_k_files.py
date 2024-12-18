from fire import Fire
from pathlib import Path
from datasets import Dataset
from tqdm.auto import tqdm

ORGANIZATION = "runtimeverification"


def main(
    clone_path: str = "/scratch/yuxiang-data/krepos",
    save_path: str = "/scratch/yuxiang-data/kfiles/raw_dataset.jsonl",
):
    
    items: list[dict] = []
    for repo_root in Path(clone_path).iterdir():
        if not repo_root.is_dir():
            continue

        all_paths = repo_root.glob("**/*.k")
        for path in tqdm(all_paths):
            repo_name = repo_root.name
            repo_id = f"{ORGANIZATION}/{repo_name}"
            relative_path = path.relative_to(repo_root).as_posix()
            content = path.read_text(encoding="utf-8")
            d = {
                "repo_id": repo_id,
                "path": relative_path,
                "content": content,
            }
            items.append(d)
    dataset = Dataset.from_list(items)
    print(f"Saving {len(dataset)} items to {save_path}")
    dataset.to_json(save_path, lines=True)


if __name__ == "__main__":
    Fire(main)
