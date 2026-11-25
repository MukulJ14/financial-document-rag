import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_NEMO_EMBED_3_1B")
)

response = client.embeddings.create(
    model = "nvidia/nemotron-3.5-embed-1b",
    input = "NVIDIA develops accelerated computing platforms.",
    extra_body={"input_type": "passage"}
)

embedding = response.data[0].embedding

print("Embedding generated successfully!")
print("Vector dimensions:", len(embedding))
print("First five values:", embedding[:5])