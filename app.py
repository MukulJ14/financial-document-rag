from src.ingestion import load_pdf
from src.chunking import chunk_documents
from src.embeddings import embed_documents, embed_query

pdf_path = "data/raw/annual_report_FY2025.pdf"

pages = load_pdf(pdf_path)
chunks = chunk_documents(pages)
vectors = embed_documents(chunks)

print("No. of pages", len(pages))
print("No. of chunks", len(chunks))
print("No. of vectors", len(vectors))

query_vector = embed_query("What was the company's revenue?")

print("Query dimensions:", len(query_vector))