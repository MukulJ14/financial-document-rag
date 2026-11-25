import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


# Checking if API keys exists or not
api_key = os.getenv("NVIDIA_NEMO_EMBED_3_1B")

if not api_key:
    raise ValueError("NVIDIA embedding API key not found")


# Creating client to use NVIDIA models
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


# function to embed documents
def embed_documents(chunks):

    if not isinstance(chunks, list):
        raise TypeError("Expected 'chunks' to be a list")

    if not chunks:
        raise ValueError("'chunks' cannot be empty")

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise TypeError("Each chunk must be a dictionary")
        
        if "text" not in chunk:
            raise ValueError("Each chunk must contain a 'text' key")
        
        if not isinstance(chunk["text"], str):
            raise TypeError("Chunk text must be a string.")

        if not chunk["text"].strip():
            raise ValueError("Chunk text cannot be empty or whitespace-only")


    texts = [chunk['text'] for chunk in chunks]


    response = client.embeddings.create(
        model="nvidia/nemotron-3.5-embed-1b",
        input=texts,
        extra_body={"input_type": "passage"}
    )

    embeddings = [item.embedding for item in response.data]

    return embeddings


# function to embed the question
def embed_query(question):

    if not isinstance(question, str):
        raise TypeError("Question must be a string")

    if len(question.strip()) == 0:
        raise ValueError("Query provided is empty")

    response = client.embeddings.create(
        model="nvidia/nemotron-3.5-embed-1b",
        input=question,
        extra_body={"input_type": "query"}
    )

    return response.data[0].embedding
