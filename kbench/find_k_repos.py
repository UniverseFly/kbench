from ghapi.all import GhApi
from fire import Fire
import re

ORG_NAME = "runtimeverification"
# contains "semantics" or is exactly "k"
PATTERN = re.compile(r"^(k)$|semantics", re.IGNORECASE)


def list_repos_with_regex(api: GhApi, per_page=100):
    page = 1
    matching_repos = list[str]()

    while True:
        repos = api.repos.list_for_org(org=ORG_NAME, per_page=per_page, page=page)
        if not repos:
            break

        for repo in repos:
            if PATTERN.search(repo.name):
                matching_repos.append(repo.full_name)

        page += 1
    return matching_repos

# Fire main
def main(
    save_path: str = "krepos.txt"
):
    api = GhApi()
    repos = list_repos_with_regex(api, PATTERN)
    with open(save_path, "w") as f:
        f.write("\n".join(repos))

if __name__ == "__main__":
    Fire(main)