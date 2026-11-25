from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(pages):

    if not pages:
        raise ValueError("Document provided is empty")
    
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
    )

    chunks = []


    for page in pages:

        if not page['text'].strip():
            print(f"Warning: Skipping Page {page['page']} since has no text")
            continue

        page_chunks = splitter.split_text(page['text'])
        
        for chunk in page_chunks:
            chunks.append(
                {
                    'text': chunk,
                    "page" : page['page'],
                    "source" : page['source']
                }
            )

    return chunks
