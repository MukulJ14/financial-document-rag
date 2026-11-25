import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("NVIDIA_NEMO_120B")

if not api_key:
    raise ValueError("NVIDIA Reasoning Agent API key not found")


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

model = "nvidia/nemotron-3.5-super-vl-120b-a12b"


# def generate_answer(question, retrieved_chunks):
