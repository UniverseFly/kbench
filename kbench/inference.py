from fire import Fire
from datasets import load_dataset, Dataset
from litellm import batch_completion
from tqdm.auto import tqdm


def main(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    dataset_path: str = "/scratch/yuxiang-data/kfiles/kbench.jsonl",
    result_root: str = "/scratch/yuxiang-data/kbench-results",
    api_base="http://0.0.0.0:8000/v1",
    api_key="EMPTY",
    temperature=0.0,
    max_tokens=4096,
    chunk_size=32,
):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    for i in tqdm(range(0, len(dataset), chunk_size)):
        chunk = dataset.select(range(i, min(i + chunk_size, len(dataset))))
        all_messages = [[{"role": "user", "content": x["prompt"]}] for x in chunk]
        responses = batch_completion(
            model=f"openai/{model_name}",
            api_base=api_base,
            api_key=api_key,
            messages=all_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        string_responses = [
            response.choices[0].message.content for response in responses
        ]
        assert len(string_responses) == len(chunk)
        results = [
            dict(
                repo_id=x["repo_id"],
                path=x["path"],
                prompt=x["prompt"],
                ground_truth=x["ground_truth"],
                part=x["part"],
                response=y,
            )
            for x, y in zip(chunk, string_responses)
        ]
        result_dataset = Dataset.from_list(results)
        fname = model_name.replace("/", "__")
        chunk_index = i // chunk_size
        results_path = f"{result_root}/{fname}/chunk_{chunk_index:04d}.jsonl"
        print(f"Saving {len(result_dataset)} items to {results_path}")
        result_dataset.to_json(results_path, lines=True)


if __name__ == "__main__":
    Fire(main)
