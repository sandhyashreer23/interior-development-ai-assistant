from PyPDF2 import PdfReader
from langchain_core.documents import Document

def read_pdf(uploaded_file):

    pdf = PdfReader(uploaded_file)

    docs = []

    for i, page in enumerate(pdf.pages):

        try:
            text = page.extract_text()

            if text is None:
                continue

            text = text.strip()

            if len(text) == 0:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": i + 1
                    }
                )
            )

        except Exception:
            continue

    return docs