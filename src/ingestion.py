import pymupdf
from pathlib import Path


def load_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):

            text = page.get_text()

            if not text.strip():
                print(f"Warning: Page {page_number} contains no extractable text.")

            pages.append(
                {
                    "text": text,
                    "page" : page_number,
                    "source" : pdf_path.name
                }
            )

    return pages
