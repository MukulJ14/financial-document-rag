import numpy as np
import faiss
from pathlib import Path
from src.embeddings import embed_query
import pickle


def save_chunks(chunks):
    directory = Path("data/processed")
    directory.mkdir(parents=True, exist_ok=True)
        
    file_path = directory / "chunks.pkl"

    with open(str(file_path), "wb") as file:
        pickle.dump(chunks, file)


def load_chunks():
    file_path = Path("data/processed/chunks.pkl")

    if not file_path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {file_path}")

    with open(str(file_path), "rb") as file:
        chunks = pickle.load(file)

    return chunks


def create_vector_store(vectors):
    vectors_array = np.asarray(vectors, dtype=np.float32)

    index = faiss.IndexFlatIP(vectors_array.shape[1])
    index.add(vectors_array)

    return index



def save_vector_store(vector_store):
    directory = Path("data/processed")
    directory.mkdir(parents=True, exist_ok=True)

    file_path = directory / "vector_store.faiss"

    faiss.write_index(vector_store, str(file_path))

    print("Index saved!")


def load_vector_store():
    file_path = Path("data/processed/vector_store.faiss")

    if not file_path.is_file():
        raise FileNotFoundError(f"Index file not found: {file_path}")


    index = faiss.read_index(str(file_path))

    return index


def retrieve_chunks(question, vector_store):
    query_vector = embed_query(question)
    query_vector = np.asarray(query_vector, dtype=np.float32)
    query_vector = query_vector.reshape(1, -1).copy()

    k = 3

    distances, indices = vector_store.search(query_vector, k)

    chunks = load_chunks()

    chunks_array = []

    for i in indices[0]:
        if i == -1:
            continue
        
        chunks_array.append(chunks[i])


    return chunks_array
