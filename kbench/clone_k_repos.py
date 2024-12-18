from fire import Fire
import os
import subprocess


def main(
    repos_path: str = "krepos.txt", clone_path: str = "/scratch/yuxiang-data/krepos"
):
    with open(repos_path, "r") as f:
        repos = f.read().splitlines()

    # Clone all repos
    os.makedirs(clone_path, exist_ok=True)
    for repo in repos:
        url = f"https://github.com/{repo}"
        subprocess.run(["git", "clone", "--depth", "1", url], cwd=clone_path)


if __name__ == "__main__":
    Fire(main)
