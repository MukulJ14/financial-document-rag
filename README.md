# Financial Document Intelligence – RAG System

A Retrieval-Augmented Generation (RAG) system that enables users to interact with financial documents through natural-language queries.

- Extracts text and metadata from financial PDFs using PyMuPDF.
- Splits documents into manageable chunks using Recursive Character Text Splitter.
- Generates 2048-dimensional embeddings using NVIDIA Nemotron 3.5 Embed 1B.
- Stores and indexes embeddings using FAISS for semantic similarity search.
- Retrieves relevant document passages based on user queries.
- Uses NVIDIA Nemotron 3.5 Super VL 120B A12B for context-aware answer generation.
- Preserves document metadata, including source and page numbers, for traceability.
- Implements modular components for ingestion, chunking, embeddings, retrieval, and generation.

**Current Status:** Core RAG pipeline implemented.

**Next Phase:** Retrieval and generation evaluation, performance optimization, and end-to-end testing.
