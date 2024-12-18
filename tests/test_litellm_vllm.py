import litellm
from litellm import batch_completion

# litellm.set_verbose = True

response = batch_completion(
    model="openai/meta-llama/Llama-3.1-8B-Instruct",
    api_base="http://0.0.0.0:8000/v1",
    api_key="EMPTY",
    messages=[
        [{"role": "user", "content": "Who are you?"}],
        [{"role": "user", "content": "Can you write a python code?"}],
    ],
    temperature=0.0,
    max_tokens=50,
)

print(response)
