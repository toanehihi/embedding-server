"""Run: uv run --group integration python examples/litellm_client.py --output-type hybrid"""

import argparse
import os

from litellm import embedding

parser = argparse.ArgumentParser()
parser.add_argument("--output-type", choices=["dense", "sparse", "hybrid"], default="dense")
args = parser.parse_args()

response = embedding(
    model="openai/bge-m3",
    api_base=os.environ.get("BGE_API_BASE", "http://127.0.0.1:8000/v1"),
    api_key=os.environ["BGE_API_KEY"],
    input=["Áo thun cotton màu trắng", "White cotton T-shirt"],
    encoding_format="float",
    extra_body={"output_type": args.output_type},
    timeout=120,
)

print("Vectors:", len(response.data))
first = response.data[0]
if args.output_type != "sparse":
    print("Dense dimensions:", len(first["embedding"]))
if args.output_type != "dense":
    print("Sparse:", first["sparse_embedding"])
print("Usage:", response.usage)
