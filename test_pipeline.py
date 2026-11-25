from src.ingestion import load_pdf
from src.chunking import chunk_documents
from src.embeddings import embed_documents
from src.retrieval import save_chunks, load_chunks, save_vector_store, load_vector_store, create_vector_store, retrieve_chunks


pdf_path = "data/raw/annual_report_FY2025.pdf"

pages = load_pdf(pdf_path)
chunks = chunk_documents(pages)
vectors = embed_documents(chunks)

# print("No. of pages", len(pages))
# print("No. of chunks", len(chunks))
# print("No. of vectors", len(vectors))

# save_chunks(chunks)

# index = create_vector_store(vectors)
# save_vector_store(index)

index = load_vector_store()
question = "What was my company's revenue?"

retrieved_chunks = retrieve_chunks(question, index)
print(retrieved_chunks)