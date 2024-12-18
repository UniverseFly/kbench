# kbench

This repo consists of two parts:

1. Code to crawl and construct kbench from `runtimeverification/...` repositories.
2. Code to run inference and evaluation on kbench.

## Crawl and Construct kbench

Follow the following commands (Please check the code file for detailed arguments. They're omitted for simplicity):

```bash
# Install required packages..

# Then run the following commands
## 1. find and clone all k-related repos
python kbench/find_k_repos.py
python kbench/clone_k_repos.py
## 2. collect all k files
python kbench/collect_k_files.py
## 3. clean the k files
python kbench/clean_k_files.py
## 4. construct kbench
python kbench/construct_kbench.py
```

## Inference and Evaluation

```bash
# Make sure you have launched an inference server (e.g., with vLLM OpenAI compatible server)
# 1. Inference
python kbench/inference.py
# 2. Compute metrics
python kbench/compute_metrics.py
```
