from pathlib import Path

from pdf_semantic_rag import DEFAULT_PDF_PATH, ingest_pdf_knowledge


if __name__ == "__main__":
    result = ingest_pdf_knowledge(
        pdf_path=Path(DEFAULT_PDF_PATH),
        replace_document=True,
    )
    print(result)
